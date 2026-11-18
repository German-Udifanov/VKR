"""Проверка входного xlsx и согласованности с тем, что ожидает код."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from app.calendar_mod import resolve_shift_type
from app.data_io import load_all_tables, parse_skills, validate_tables


def main() -> None:
    path = _ROOT / "data_out" / "synthetic_medium_150_new.xlsx"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).resolve() if Path(sys.argv[1]).is_absolute() else (_ROOT / sys.argv[1]).resolve()

    t = load_all_tables(path)
    validate_tables(t)
    issues: list[str] = []

    mtypes = set(t.machines["machine_type"].astype(str))
    op_types = set(t.operations["machine_type"].astype(str))
    missing_mt = op_types - mtypes
    if missing_mt:
        issues.append(f"operations.machine_type без machines: {sorted(missing_mt)}")

    products_orders = set(t.orders["product_type"].astype(str))
    products_ops = set(t.operations["product_type"].astype(str))
    missing_routes = products_orders - products_ops
    if missing_routes:
        issues.append(f"orders.product_type без маршрута operations: {sorted(missing_routes)}")

    for _, op in t.operations.iterrows():
        sk = str(op["required_skill"]).upper()
        lvl = int(op["min_skill_level"])
        ok = any(parse_skills(r["skills"]).get(sk, 0) >= lvl for _, r in t.employees.iterrows())
        if not ok:
            nm = op.get("operation_name", "")
            issues.append(f"Ни у кого нет {sk}>={lvl} (op {nm}, product {op['product_type']})")
            break

    lookup_keys = set(
        zip(
            t.shifts["shift_pattern"].astype(str),
            t.shifts["shift_type"].astype(str).str.upper(),
        )
    )
    bad_shift = []
    for _, e in t.employees.iterrows():
        st = resolve_shift_type(
            str(e["preferred_shift_type"]),
            bool(e["can_work_night"]),
            str(e["shift_pattern"]),
            int(e["employee_id"]),
        )
        key = (str(e["shift_pattern"]), st)
        if key not in lookup_keys:
            bad_shift.append((int(e["employee_id"]), key))
    if bad_shift:
        issues.append(
            f"Сотрудники без строки в shifts для (pattern, shift_type): {len(bad_shift)} шт, "
            f"примеры {bad_shift[:5]} — в коде им выставляется полная недоступность на весь горизонт"
        )

    # default run.py path
    from run import parse_cli

    cfg = parse_cli(_ROOT, [])
    if cfg.input_xlsx.resolve() != path.resolve():
        issues.append(f"Дефолт run.py: {cfg.input_xlsx} (проверяли: {path})")

    print("Файл:", path)
    print("Строки:", {k: len(getattr(t, k)) for k in ["employees", "machines", "orders", "operations", "downtime", "vacations"]})
    print("validate_tables: OK")
    if issues:
        print("Замечания:")
        for i in issues:
            print(" -", i)
        sys.exit(1)
    print("Расширенные проверки: OK")


if __name__ == "__main__":
    main()
