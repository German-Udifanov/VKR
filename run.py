from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.calendar_mod import build_employee_unavailability
from app.data_io import load_all_tables, validate_tables
from app.export import export_comparison_excel, export_daily_excel
from app.kpi import compute_all_kpi
from app.ml_model import apply_ml_to_route
from app.optimizer import SolveConfig, solve_hybrid
from app.planning import expand_orders_to_operations
from app.utils import RuntimeConfig, ask_yes_no, ensure_dir, now_str
from app.checks import run_checks


def parse_cli(project_root: Path, argv: list[str] | None = None) -> RuntimeConfig:
    p = argparse.ArgumentParser(
        description="Построение расписания: данные из Excel, решатель CP-SAT.",
    )
    p.add_argument(
        "--input-xlsx",
        type=Path,
        default=project_root / "data_out" / "synthetic_medium_150_new.xlsx",
        help="Входной Excel с заказами, сотрудниками, календарём и т.д.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Raspisanie",
        help="Каталог для выходных xlsx.",
    )
    p.add_argument("--step-min", type=int, default=30, help="Дискретизация времени, минуты.")
    p.add_argument("--time-limit-sec", type=int, default=500, help="Лимит времени решателя (сек).")
    p.add_argument("--workers", type=int, default=8, help="Число потоков OR-Tools.")
    p.add_argument("--w-tardiness", type=int, default=80, help="Вес штрафа за опоздания в целевой функции.")
    p.add_argument("--w-balance", type=int, default=30, help="Вес баланса загрузки (Gini).")
    p.add_argument("--w-idle-day", type=int, default=2000, help="Вес «пустого» рабочего дня при остатке работы.")
    p.add_argument("--w-low-work-day", type=int, default=15, help="Вес низкой загрузки при остатке работы.")
    p.add_argument(
        "--min-work-hours-per-day",
        type=float,
        default=6.0,
        help="Минимум часов работы в день (для штрафа low-work).",
    )
    p.add_argument(
        "--w-emp-idle-day",
        type=int,
        default=120,
        help="Вес: сотрудник в рабочий по графику день без операции при наличии работы.",
    )
    p.add_argument(
        "--two-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Двухфазная оптимизация: сначала общая цель, затем усиление простоев при потолке по опозданиям.",
    )
    p.add_argument(
        "--tardiness-cap-margin",
        type=float,
        default=0.08,
        help="Допустимое ухудшение взвешенной просрочки во 2-й фазе: множитель (1+margin) к фазе 1.",
    )
    p.add_argument(
        "--two-pass-phase1-time-frac",
        type=float,
        default=0.55,
        help="Доля лимита времени на 1-ю фазу при two-pass (0..1).",
    )
    p.add_argument("--phase2-idle-mult", type=float, default=2.0, help="Множитель w_idle_day во 2-й фазе.")
    p.add_argument("--phase2-low-work-mult", type=float, default=2.0, help="Множитель w_low_work_day во 2-й фазе.")
    p.add_argument("--phase2-emp-idle-mult", type=float, default=1.5, help="Множитель w_emp_idle_day во 2-й фазе.")

    args = p.parse_args(argv)

    if args.step_min <= 0:
        p.error("--step-min должен быть > 0")
    if args.time_limit_sec <= 0:
        p.error("--time-limit-sec должен быть > 0")
    if args.workers <= 0:
        p.error("--workers должен быть > 0")
    if not (0.0 < args.two_pass_phase1_time_frac < 1.0):
        p.error("--two-pass-phase1-time-frac должен быть в (0, 1)")
    if args.tardiness_cap_margin < 0:
        p.error("--tardiness-cap-margin не может быть отрицательным")

    return RuntimeConfig(
        input_xlsx=args.input_xlsx,
        output_dir=args.output_dir,
        step_min=args.step_min,
        time_limit_sec=args.time_limit_sec,
        workers=args.workers,
        w_tardiness=args.w_tardiness,
        w_balance=args.w_balance,
        w_idle_day=args.w_idle_day,
        w_low_work_day=args.w_low_work_day,
        min_work_hours_per_day=args.min_work_hours_per_day,
        w_emp_idle_day=args.w_emp_idle_day,
        two_pass=args.two_pass,
        tardiness_cap_margin=args.tardiness_cap_margin,
        two_pass_phase1_time_frac=args.two_pass_phase1_time_frac,
        phase2_idle_mult=args.phase2_idle_mult,
        phase2_low_work_mult=args.phase2_low_work_mult,
        phase2_emp_idle_mult=args.phase2_emp_idle_mult,
    )


def main(argv: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parent
    cfg = parse_cli(project_root, argv if argv is not None else sys.argv[1:])

    if not cfg.input_xlsx.exists():
        raise FileNotFoundError(f"Не найден файл с данными: {cfg.input_xlsx}")

    print(f"Файл данных: {cfg.input_xlsx}")
    print(
        f"Лимит времени решателя: {cfg.time_limit_sec} сек (workers={cfg.workers}, step={cfg.step_min} мин); "
        f"w_tard={cfg.w_tardiness}, w_balance={cfg.w_balance}, w_idle_day={cfg.w_idle_day}, "
        f"w_low_work={cfg.w_low_work_day}, w_emp_idle={cfg.w_emp_idle_day}, two_pass={cfg.two_pass}"
    )
    print("Загружаем данные...")
    tables = load_all_tables(cfg.input_xlsx)
    validate_tables(tables)
    print(
        f"Данные загружены: employees={len(tables.employees)}, machines={len(tables.machines)}, orders={len(tables.orders)}"
    )
    print("Таблица готова к созданию расписания.")

    if not ask_yes_no("Создаем? (ДА/НЕТ): "):
        print("Отменено пользователем.")
        return

    print("Готовим ограничения (смены/обед/отпуска/простои)...")
    emp_unav = build_employee_unavailability(tables.employees, tables.calendar, tables.shifts, tables.vacations)

    def _solve_one(route_df, label: str):
        ops = expand_orders_to_operations(tables.orders, route_df, step_min=cfg.step_min)
        print(f"Строим расписание ({label})...")
        t0 = time.perf_counter()
        sched = solve_hybrid(
            ops,
            orders_df=tables.orders,
            employees_df=tables.employees,
            machines_df=tables.machines,
            employee_unavailability=emp_unav,
            downtime_df=tables.downtime,
            calendar_df=tables.calendar,
            vacations_df=tables.vacations,
            cfg=SolveConfig(
                step_min=cfg.step_min,
                time_limit_sec=cfg.time_limit_sec,
                workers=cfg.workers,
                w_tardiness=cfg.w_tardiness,
                w_balance=max(1, int(cfg.w_balance)),
                w_idle_day=cfg.w_idle_day,
                w_low_work_day=cfg.w_low_work_day,
                min_work_hours_per_day=cfg.min_work_hours_per_day,
                w_emp_idle_day=cfg.w_emp_idle_day,
                two_pass=cfg.two_pass,
                tardiness_cap_margin=cfg.tardiness_cap_margin,
                two_pass_phase1_time_frac=cfg.two_pass_phase1_time_frac,
                phase2_idle_mult=cfg.phase2_idle_mult,
                phase2_low_work_mult=cfg.phase2_low_work_mult,
                phase2_emp_idle_mult=cfg.phase2_emp_idle_mult,
            ),
        )
        elapsed = time.perf_counter() - t0
        return sched, elapsed

    print("Формируем операции по маршруту...")
    route_base = tables.operations
    schedule_base, t_base = _solve_one(route_base, "без ML")
    if schedule_base.empty:
        print(f"Не удалось построить расписание (без ML). Время: {t_base:.2f} сек")
        return
    kpi_base = compute_all_kpi(
        schedule_base,
        tables.orders,
        employees_df=tables.employees,
        calendar_df=tables.calendar,
        vacations_df=tables.vacations,
    )
    checks_base = run_checks(schedule_df=schedule_base, employee_unavailability=emp_unav)
    for r in checks_base:
        print(f"[check] {r.name}: {'OK' if r.ok else 'FAIL'} (count={r.count})")
    print(
        f"[без ML] time={t_base:.2f}s balance_gini={kpi_base.balance_score:.3f} late_rate={kpi_base.late_rate:.2%} "
        f"avg_delay_h={kpi_base.avg_delay_hours:.2f} pattern_idle={kpi_base.pattern_idle_rate:.1%}"
    )

    do_ml = ask_yes_no("Построить также вариант С ML для сравнения? (ДА/НЕТ): ")
    if not do_ml:
        ensure_dir(cfg.output_dir)
        out_path = cfg.output_dir / f"raspisanie_{now_str()}.xlsx"
        print("Экспортируем Excel (единый файл, по дням)...")
        export_daily_excel(
            out_path,
            schedule_df=schedule_base,
            employees_df=tables.employees,
            calendar_df=tables.calendar,
            vacations_df=tables.vacations,
            kpi_sheets={
                "KPI_workload_total": kpi_base.workload_by_employee,
                "KPI_workload_day": kpi_base.workload_by_employee_day,
                "KPI_workload_week": kpi_base.workload_by_employee_week,
            },
        )
        print("Расписание составлено.")
        print(f"Файл: {out_path}")
        return

    print("ML: корректируем длительности операций...")
    route_ml = apply_ml_to_route(route_base, tables.orders, seed=42)
    schedule_ml, t_ml = _solve_one(route_ml, "с ML")
    if schedule_ml.empty:
        print(f"Не удалось построить расписание (с ML). Время: {t_ml:.2f} сек")
        # still export base
        ensure_dir(cfg.output_dir)
        out_path = cfg.output_dir / f"raspisanie_{now_str()}_base.xlsx"
        export_daily_excel(
            out_path,
            schedule_df=schedule_base,
            employees_df=tables.employees,
            calendar_df=tables.calendar,
            vacations_df=tables.vacations,
        )
        print(f"Сохранен базовый вариант: {out_path}")
        return
    kpi_ml = compute_all_kpi(
        schedule_ml,
        tables.orders,
        employees_df=tables.employees,
        calendar_df=tables.calendar,
        vacations_df=tables.vacations,
    )
    checks_ml = run_checks(schedule_df=schedule_ml, employee_unavailability=emp_unav)
    for r in checks_ml:
        print(f"[check] {r.name}: {'OK' if r.ok else 'FAIL'} (count={r.count})")
    print(
        f"[с ML]  time={t_ml:.2f}s balance_gini={kpi_ml.balance_score:.3f} late_rate={kpi_ml.late_rate:.2%} "
        f"avg_delay_h={kpi_ml.avg_delay_hours:.2f} pattern_idle={kpi_ml.pattern_idle_rate:.1%}"
    )

    ensure_dir(cfg.output_dir)
    out_path = cfg.output_dir / f"raspisanie_compare_{now_str()}.xlsx"
    print("Экспортируем Excel сравнения (2 расписания + KPI)...")
    export_comparison_excel(
        out_path,
        schedule_base=schedule_base,
        schedule_ml=schedule_ml,
        employees_df=tables.employees,
        calendar_df=tables.calendar,
        vacations_df=tables.vacations,
        kpi_base={
            "balance_gini": float(kpi_base.balance_score),
            "late_rate": float(kpi_base.late_rate),
            "avg_delay_h": float(kpi_base.avg_delay_hours),
            "pattern_idle": float(kpi_base.pattern_idle_rate),
            "late_orders": int(kpi_base.late_orders),
            "orders_total": int(len(tables.orders)),
            "workload_total": kpi_base.workload_by_employee,
            "workload_day": kpi_base.workload_by_employee_day,
            "workload_week": kpi_base.workload_by_employee_week,
        },
        kpi_ml={
            "balance_gini": float(kpi_ml.balance_score),
            "late_rate": float(kpi_ml.late_rate),
            "avg_delay_h": float(kpi_ml.avg_delay_hours),
            "pattern_idle": float(kpi_ml.pattern_idle_rate),
            "late_orders": int(kpi_ml.late_orders),
            "orders_total": int(len(tables.orders)),
            "workload_total": kpi_ml.workload_by_employee,
            "workload_day": kpi_ml.workload_by_employee_day,
            "workload_week": kpi_ml.workload_by_employee_week,
        },
        time_base_sec=t_base,
        time_ml_sec=t_ml,
    )
    print("Готово. Файл сравнения:")
    print(out_path)


if __name__ == "__main__":
    main()

