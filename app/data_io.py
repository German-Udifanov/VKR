from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Tables:
    employees: pd.DataFrame
    machines: pd.DataFrame
    orders: pd.DataFrame
    vacations: pd.DataFrame
    operations: pd.DataFrame
    calendar: pd.DataFrame
    shifts: pd.DataFrame
    downtime: pd.DataFrame


REQUIRED_SHEETS = (
    "employees",
    "machines",
    "orders",
    "vacations",
    "operations",
    "calendar",
    "shifts",
    "downtime",
)

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "employees": [
        "employee_id",
        "name",
        "age",
        "experience_years",
        "skills",
        "speed",
        "quality",
        "shift_pattern",
        "preferred_shift_type",
        "can_work_night",
    ],
    "machines": [
        "machine_id",
        "machine_type",
        "required_skill",
        "min_skill_level",
        "cycle_time_min",
        "operators_required",
    ],
    "orders": [
        "order_id",
        "created_at",
        "due_at",
        "product_type",
        "quantity",
        "priority",
        "late_penalty",
        "order_class",
    ],
    "vacations": [
        "employee_id",
        "date_from",
        "date_to",
    ],
    "operations": [
        "product_type",
        "op_seq",
        "operation_name",
        "machine_type",
        "required_skill",
        "min_skill_level",
        "duration_min_per_unit",
        "operators_required",
    ],
    "calendar": [
        "date",
        "is_workday_rf",
    ],
    "shifts": [
        "shift_pattern",
        "shift_type",
        "start_time",
        "end_time",
        "break_start",
        "break_end",
        "work_minutes",
    ],
    "downtime": [
        "machine_id",
        "from_dt",
        "to_dt",
        "reason",
    ],
}


def load_all_tables(xlsx_path: str | Path) -> Tables:
    p = Path(xlsx_path)
    xls = pd.ExcelFile(p)
    missing = [s for s in REQUIRED_SHEETS if s not in xls.sheet_names]
    if missing:
        raise ValueError(f"В файле нет листов: {missing}")

    def read(name: str) -> pd.DataFrame:
        return pd.read_excel(p, sheet_name=name)

    employees = read("employees")
    machines = read("machines")
    orders = read("orders")
    vacations = read("vacations")
    operations = read("operations")
    calendar = read("calendar")
    shifts = read("shifts")
    downtime = read("downtime")

    # Normalize basic dtypes
    orders["created_at"] = pd.to_datetime(orders["created_at"])
    orders["due_at"] = pd.to_datetime(orders["due_at"])
    downtime["from_dt"] = pd.to_datetime(downtime["from_dt"])
    downtime["to_dt"] = pd.to_datetime(downtime["to_dt"])
    vacations["date_from"] = pd.to_datetime(vacations["date_from"]).dt.date
    vacations["date_to"] = pd.to_datetime(vacations["date_to"]).dt.date
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.date
    employees["can_work_night"] = employees["can_work_night"].astype(str).str.upper().map({"YES": True, "NO": False})
    # Cycle anchors (for 2/2 and 3/3): if not present in Excel, create deterministic anchors.
    # Important: anchor period depends on shift_pattern (2/2 -> 4, 3/3 -> 6).
    if "cycle_anchor" not in employees.columns:
        def _anchor(row) -> int:
            eid = int(row["employee_id"])
            pat = str(row.get("shift_pattern", "2/2"))
            if pat == "2/2":
                return eid % 4
            if pat == "3/3":
                return eid % 6
            return 0
        employees["cycle_anchor"] = employees.apply(_anchor, axis=1)
    employees["cycle_anchor"] = pd.to_numeric(employees["cycle_anchor"], errors="coerce").fillna(0).astype(int)

    return Tables(
        employees=employees,
        machines=machines,
        orders=orders,
        vacations=vacations,
        operations=operations,
        calendar=calendar,
        shifts=shifts,
        downtime=downtime,
    )


def parse_skills(skills_raw: str) -> dict[str, int]:
    text = str(skills_raw or "").strip()
    if not text:
        return {}
    out: dict[str, int] = {}
    for chunk in text.split(";"):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            continue
        s, lvl = item.split(":", 1)
        s = s.strip().upper()
        try:
            out[s] = int(lvl.strip())
        except ValueError:
            continue
    return out


def validate_tables(t: Tables) -> None:
    # 1) Non-empty core sheets
    if t.employees.empty:
        raise ValueError("employees пустой")
    if t.machines.empty:
        raise ValueError("machines пустой")
    if t.orders.empty:
        raise ValueError("orders пустой")
    if t.operations.empty:
        raise ValueError("operations пустой")
    if "cycle_anchor" not in t.employees.columns:
        raise ValueError("Нет колонки employees.cycle_anchor (якорь цикла).")

    # 2) Required columns
    missing_msgs = []
    for sheet, cols in REQUIRED_COLUMNS.items():
        df = getattr(t, sheet)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            missing_msgs.append(f"{sheet}: {missing}")
    if missing_msgs:
        raise ValueError("Не хватает колонок:\n" + "\n".join(missing_msgs))

    # 3) Uniqueness of IDs
    if t.employees["employee_id"].duplicated().any():
        raise ValueError("employees.employee_id должен быть уникальным")
    if t.machines["machine_id"].duplicated().any():
        raise ValueError("machines.machine_id должен быть уникальным")
    if t.orders["order_id"].duplicated().any():
        raise ValueError("orders.order_id должен быть уникальным")

    # 4) Value ranges & enums
    def _range(df: pd.DataFrame, col: str, lo: float, hi: float, label: str) -> None:
        s = pd.to_numeric(df[col], errors="coerce")
        bad = df[s.isna() | (s < lo) | (s > hi)]
        if not bad.empty:
            raise ValueError(f"{label} вне диапазона [{lo}..{hi}] (строк: {len(bad)})")

    _range(t.employees, "age", 16, 80, "employees.age")
    _range(t.employees, "experience_years", 0, 70, "employees.experience_years")
    _range(t.employees, "speed", 1, 10, "employees.speed")
    _range(t.employees, "quality", 1, 10, "employees.quality")
    _range(t.machines, "min_skill_level", 1, 6, "machines.min_skill_level")
    _range(t.operations, "min_skill_level", 1, 6, "operations.min_skill_level")
    _range(t.orders, "priority", 1, 5, "orders.priority")
    _range(t.orders, "quantity", 1, 1_000_000, "orders.quantity")

    allowed_patterns = {"2/2", "3/3", "5/2"}
    if not set(t.employees["shift_pattern"].astype(str)).issubset(allowed_patterns):
        raise ValueError("employees.shift_pattern должен быть одним из: 2/2, 3/3, 5/2")
    allowed_pref = {"DAY", "NIGHT", "ANY"}
    if not set(t.employees["preferred_shift_type"].astype(str).str.upper()).issubset(allowed_pref):
        raise ValueError("employees.preferred_shift_type должен быть DAY/NIGHT/ANY")

    # cycle_anchor bounds depend on pattern
    emp = t.employees.copy()
    emp["shift_pattern"] = emp["shift_pattern"].astype(str)
    emp["cycle_anchor"] = pd.to_numeric(emp["cycle_anchor"], errors="coerce").fillna(0).astype(int)
    bad_22 = emp[(emp["shift_pattern"] == "2/2") & ((emp["cycle_anchor"] < 0) | (emp["cycle_anchor"] > 3))]
    bad_33 = emp[(emp["shift_pattern"] == "3/3") & ((emp["cycle_anchor"] < 0) | (emp["cycle_anchor"] > 5))]
    bad_52 = emp[(emp["shift_pattern"] == "5/2") & (emp["cycle_anchor"] != 0)]
    if not bad_22.empty:
        raise ValueError("employees.cycle_anchor для 2/2 должен быть 0..3")
    if not bad_33.empty:
        raise ValueError("employees.cycle_anchor для 3/3 должен быть 0..5")
    if not bad_52.empty:
        raise ValueError("employees.cycle_anchor для 5/2 должен быть 0")

    # 5) Date/time sanity
    if (pd.to_datetime(t.orders["due_at"]) < pd.to_datetime(t.orders["created_at"])).any():
        raise ValueError("orders: due_at раньше created_at (есть некорректные строки)")
    if (pd.to_datetime(t.downtime["to_dt"]) <= pd.to_datetime(t.downtime["from_dt"])).any():
        raise ValueError("downtime: to_dt должен быть больше from_dt")
    if (pd.to_datetime(t.vacations["date_to"]) < pd.to_datetime(t.vacations["date_from"])).any():
        raise ValueError("vacations: date_to раньше date_from")

    # 6) Referential integrity
    emp_ids = set(t.employees["employee_id"].astype(int).tolist())
    bad_v = t.vacations[~t.vacations["employee_id"].astype(int).isin(emp_ids)]
    if not bad_v.empty:
        raise ValueError(f"vacations: неизвестные employee_id (строк: {len(bad_v)})")
    mach_ids = set(t.machines["machine_id"].astype(int).tolist())
    bad_d = t.downtime[~t.downtime["machine_id"].astype(int).isin(mach_ids)]
    if not bad_d.empty:
        raise ValueError(f"downtime: неизвестные machine_id (строк: {len(bad_d)})")

    # 7) Skills parsing sanity (format)
    for idx, raw in enumerate(t.employees["skills"].astype(str).tolist()):
        try:
            parse_skills(raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"employees.skills: ошибка парсинга в строке {idx}: {exc}") from exc

