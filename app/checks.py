from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    count: int


def _overlaps(df: pd.DataFrame, *, id_col: str) -> int:
    if df.empty:
        return 0
    tmp = df.copy()
    tmp["start_dt"] = pd.to_datetime(tmp["start_dt"])
    tmp["end_dt"] = pd.to_datetime(tmp["end_dt"])
    tmp = tmp.sort_values([id_col, "start_dt", "end_dt"])
    conflicts = 0
    for _, grp in tmp.groupby(id_col):
        prev_end = None
        for _, r in grp.iterrows():
            if prev_end is not None and r["start_dt"] < prev_end:
                conflicts += 1
            prev_end = max(prev_end, r["end_dt"]) if prev_end is not None else r["end_dt"]
    return conflicts


def check_no_overlap_machine(schedule_df: pd.DataFrame) -> CheckResult:
    c = _overlaps(schedule_df, id_col="machine_id")
    return CheckResult("no_overlap_machine", c == 0, c)


def check_no_overlap_employee(schedule_df: pd.DataFrame) -> CheckResult:
    c = _overlaps(schedule_df, id_col="employee_id")
    return CheckResult("no_overlap_employee", c == 0, c)


def check_precedence(schedule_df: pd.DataFrame) -> CheckResult:
    if schedule_df.empty:
        return CheckResult("precedence", True, 0)
    df = schedule_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["end_dt"] = pd.to_datetime(df["end_dt"])
    bad = 0
    for _, grp in df.sort_values(["order_id", "op_seq"]).groupby("order_id"):
        prev_end = None
        for _, r in grp.iterrows():
            if prev_end is not None and r["start_dt"] < prev_end:
                bad += 1
            prev_end = r["end_dt"]
    return CheckResult("precedence", bad == 0, bad)


def check_employee_unavailability(
    schedule_df: pd.DataFrame, unav: dict[int, list[tuple[datetime, datetime]]]
) -> CheckResult:
    if schedule_df.empty:
        return CheckResult("employee_unavailability", True, 0)
    df = schedule_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["end_dt"] = pd.to_datetime(df["end_dt"])
    bad = 0
    for eid, grp in df.groupby("employee_id"):
        blocks = unav.get(int(eid), [])
        if not blocks:
            continue
        for _, r in grp.iterrows():
            for sdt, edt in blocks:
                if pd.Timestamp(sdt) < r["end_dt"] and pd.Timestamp(edt) > r["start_dt"]:
                    bad += 1
                    break
    return CheckResult("employee_unavailability", bad == 0, bad)


def run_checks(
    *,
    schedule_df: pd.DataFrame,
    employee_unavailability: dict[int, list[tuple[datetime, datetime]]],
) -> list[CheckResult]:
    return [
        check_no_overlap_machine(schedule_df),
        check_no_overlap_employee(schedule_df),
        check_precedence(schedule_df),
        check_employee_unavailability(schedule_df, employee_unavailability),
    ]

