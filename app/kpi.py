from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from app.calendar_mod import _is_work_day_for_pattern


@dataclass(frozen=True)
class KPI:
    workload_by_employee: pd.DataFrame
    workload_by_employee_day: pd.DataFrame
    workload_by_employee_week: pd.DataFrame
    balance_score: float
    late_orders: int
    late_rate: float
    avg_delay_hours: float
    pattern_work_slots: int = 0
    pattern_idle_slots: int = 0
    pattern_idle_rate: float = 0.0


def compute_workload_by_employee(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(columns=["employee_id", "busy_hours"])
    df = schedule_df.copy()
    df["busy_hours"] = (pd.to_datetime(df["end_dt"]) - pd.to_datetime(df["start_dt"])).dt.total_seconds() / 3600.0
    out = df.groupby("employee_id", as_index=False)["busy_hours"].sum().sort_values("busy_hours", ascending=False)
    return out.reset_index(drop=True)

def compute_workload_by_employee_day(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(columns=["date", "employee_id", "busy_hours"])
    df = schedule_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["end_dt"] = pd.to_datetime(df["end_dt"])
    df["date"] = df["start_dt"].dt.date
    df["busy_hours"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 3600.0
    out = df.groupby(["date", "employee_id"], as_index=False)["busy_hours"].sum()
    return out.sort_values(["date", "employee_id"]).reset_index(drop=True)


def compute_workload_by_employee_week(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(columns=["week", "employee_id", "busy_hours"])
    df = schedule_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["end_dt"] = pd.to_datetime(df["end_dt"])
    df["week"] = df["start_dt"].dt.to_period("W").astype(str)
    df["busy_hours"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 3600.0
    out = df.groupby(["week", "employee_id"], as_index=False)["busy_hours"].sum()
    return out.sort_values(["week", "employee_id"]).reset_index(drop=True)


def gini(values: np.ndarray) -> float:
    x = np.array(values, dtype=float)
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    g = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    return float(max(0.0, min(1.0, g)))


def compute_balance_score(workload_df: pd.DataFrame) -> float:
    if workload_df.empty:
        return 0.0
    return gini(workload_df["busy_hours"].to_numpy())


def compute_deadline_kpi(schedule_df: pd.DataFrame, orders_df: pd.DataFrame) -> tuple[int, float, float]:
    if schedule_df.empty:
        return 0, 0.0, 0.0
    last_ops = schedule_df.sort_values(["order_id", "op_seq"]).groupby("order_id").tail(1)[["order_id", "end_dt"]]
    merged = last_ops.merge(orders_df[["order_id", "due_at"]], on="order_id", how="left")
    delay_h = ((pd.to_datetime(merged["end_dt"]) - pd.to_datetime(merged["due_at"])).dt.total_seconds() / 3600.0).clip(lower=0.0)
    late = int((delay_h > 0).sum())
    rate = float(late / max(1, len(merged)))
    avg = float(delay_h.mean())
    return late, rate, avg


def compute_pattern_idle_rate(
    schedule_df: pd.DataFrame,
    employees_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    vacations_df: pd.DataFrame | None = None,
) -> tuple[int, int, float]:
    """
    Доля «простоев по графику»: рабочий по смене день без ни одной операции (по дате старта), / все рабочие по графику слоты.
    """
    if schedule_df.empty or employees_df.empty or calendar_df.empty:
        return 0, 0, 0.0
    cal = calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    base_date = cal["date"].min()
    rf_map = {row["date"]: int(row["is_workday_rf"]) for _, row in cal.iterrows()}
    vac = vacations_df if vacations_df is not None else pd.DataFrame()

    df = schedule_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    d_min = df["start_dt"].dt.date.min()
    d_max = df["start_dt"].dt.date.max()

    busy_slots: set[tuple[int, object]] = set()
    for _, r in df.iterrows():
        busy_slots.add((int(r["employee_id"]), pd.Timestamp(r["start_dt"]).date()))

    work_slots = 0
    idle_slots = 0
    d_cur = d_min
    while d_cur <= d_max:
        irf = int(rf_map.get(d_cur, 1))
        for _, er in employees_df.iterrows():
            eid = int(er["employee_id"])
            pat = str(er.get("shift_pattern", "2/2")).strip()
            anchor = int(er.get("cycle_anchor", 0))
            if not _is_work_day_for_pattern(
                d_cur, pattern=pat, cycle_anchor=anchor, is_workday_rf=irf, base_date=base_date
            ):
                continue
            on_vac = False
            if not vac.empty and "employee_id" in vac.columns:
                for _, vr in vac[vac["employee_id"].astype(int) == eid].iterrows():
                    d0 = pd.to_datetime(vr["date_from"]).date()
                    d1 = pd.to_datetime(vr["date_to"]).date()
                    if d0 <= d_cur <= d1:
                        on_vac = True
                        break
            if on_vac:
                continue
            work_slots += 1
            if (eid, d_cur) not in busy_slots:
                idle_slots += 1
        d_cur = d_cur + timedelta(days=1)

    rate = float(idle_slots / max(1, work_slots))
    return work_slots, idle_slots, rate


def compute_all_kpi(
    schedule_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    *,
    employees_df: pd.DataFrame | None = None,
    calendar_df: pd.DataFrame | None = None,
    vacations_df: pd.DataFrame | None = None,
) -> KPI:
    wl = compute_workload_by_employee(schedule_df)
    wl_day = compute_workload_by_employee_day(schedule_df)
    wl_week = compute_workload_by_employee_week(schedule_df)
    balance = compute_balance_score(wl)
    late, rate, avg = compute_deadline_kpi(schedule_df, orders_df)
    p_work, p_idle, p_rate = (0, 0, 0.0)
    if employees_df is not None and calendar_df is not None and not calendar_df.empty:
        p_work, p_idle, p_rate = compute_pattern_idle_rate(
            schedule_df, employees_df, calendar_df, vacations_df=vacations_df
        )
    return KPI(
        workload_by_employee=wl,
        workload_by_employee_day=wl_day,
        workload_by_employee_week=wl_week,
        balance_score=balance,
        late_orders=late,
        late_rate=rate,
        avg_delay_hours=avg,
        pattern_work_slots=p_work,
        pattern_idle_slots=p_idle,
        pattern_idle_rate=p_rate,
    )

