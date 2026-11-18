"""Общий конвейер: загрузка данных → решатель → KPI → Excel (без ML)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


from app.calendar_mod import build_employee_unavailability
from app.checks import CheckResult, run_checks
from app.data_io import Tables, load_all_tables, validate_tables
from app.export import export_daily_excel
from app.kpi import KPI, compute_all_kpi
from app.optimizer import SolveConfig, solve_hybrid
from app.planning import expand_orders_to_operations
from app.utils import RuntimeConfig, ensure_dir, now_str


@dataclass(frozen=True)
class ScheduleRunResult:
    schedule_df: pd.DataFrame
    kpi: KPI
    tables: Tables
    emp_unavailability: dict[int, list[tuple[datetime, datetime]]]
    solve_seconds: float
    excel_path: Path
    checks: list[CheckResult]


def _solve_config(cfg: RuntimeConfig) -> SolveConfig:
    return SolveConfig(
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
    )


def run_schedule_without_ml(cfg: RuntimeConfig) -> ScheduleRunResult:
    tables = load_all_tables(cfg.input_xlsx)
    validate_tables(tables)
    emp_unav = build_employee_unavailability(
        tables.employees, tables.calendar, tables.shifts, tables.vacations
    )
    ops = expand_orders_to_operations(tables.orders, tables.operations, step_min=cfg.step_min)
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
        cfg=_solve_config(cfg),
    )
    elapsed = time.perf_counter() - t0
    if sched.empty:
        raise RuntimeError("Решатель вернул пустое расписание (проверьте данные и лимит времени).")

    kpi = compute_all_kpi(
        sched,
        tables.orders,
        employees_df=tables.employees,
        calendar_df=tables.calendar,
        vacations_df=tables.vacations,
    )
    ensure_dir(cfg.output_dir)
    out_path = cfg.output_dir / f"raspisanie_{now_str()}.xlsx"
    export_daily_excel(
        out_path,
        schedule_df=sched,
        employees_df=tables.employees,
        calendar_df=tables.calendar,
        vacations_df=tables.vacations,
        kpi_sheets={
            "KPI_workload_total": kpi.workload_by_employee,
            "KPI_workload_day": kpi.workload_by_employee_day,
            "KPI_workload_week": kpi.workload_by_employee_week,
        },
    )
    checks = run_checks(schedule_df=sched, employee_unavailability=emp_unav)
    return ScheduleRunResult(
        schedule_df=sched,
        kpi=kpi,
        tables=tables,
        emp_unavailability=emp_unav,
        solve_seconds=elapsed,
        excel_path=out_path,
        checks=list(checks),
    )
