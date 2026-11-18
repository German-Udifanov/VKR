from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd


def _time_from(v) -> time:
    if isinstance(v, time):
        return v
    return datetime.strptime(str(v), "%H:%M").time()


def _make_interval(day: date, t0: time, t1: time) -> tuple[datetime, datetime]:
    start = datetime.combine(day, t0)
    end_day = day if t1 > t0 else (day + timedelta(days=1))
    end = datetime.combine(end_day, t1)
    return start, end


def _make_break_interval(
    *,
    shift_day: date,
    shift_start: time,
    shift_end: time,
    break_start: time,
    break_end: time,
) -> tuple[datetime, datetime]:
    """
    For night shifts, breaks like 01:00-02:00 must be placed on the next calendar day
    relative to the shift start day.
    """
    crosses_midnight = shift_end <= shift_start
    break_day = shift_day
    if crosses_midnight and break_start < shift_start:
        break_day = shift_day + timedelta(days=1)
    return _make_interval(break_day, break_start, break_end)


def _is_work_day_for_pattern(
    day: date,
    *,
    pattern: str,
    cycle_anchor: int,
    is_workday_rf: int,
    base_date: date,
) -> bool:
    if pattern == "5/2":
   
        return int(day.weekday()) < 5
    day_idx = (day - base_date).days
    if pattern == "2/2":
        idx = (int(day_idx) - int(cycle_anchor)) % 4
        return idx in (0, 1)
    if pattern == "3/3":
        idx = (int(day_idx) - int(cycle_anchor)) % 6
        return idx in (0, 1, 2)
    return False


def resolve_shift_type(preferred_shift_type: str, can_work_night: bool, pattern: str, employee_id: int) -> str:
    if pattern == "5/2":
        return "DAY"
    pref = str(preferred_shift_type).upper()
    if pref == "DAY":
        return "DAY"
    if pref == "NIGHT":
        return "NIGHT" if can_work_night else "DAY"
    # ANY
    if not can_work_night:
        return "DAY"
    return "NIGHT" if (employee_id % 10) < 4 else "DAY"


def build_employee_unavailability(
    employees_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    vacations_df: pd.DataFrame,
) -> dict[int, list[tuple[datetime, datetime]]]:
    """
    Returns employee -> list of unavailable windows:
    - OFFSHIFT (everything outside working windows)
    - LUNCH (break window inside shift)
    - VACATION (full days)
    """
    cal = calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    start_base = datetime.combine(cal["date"].min(), time(0, 0))
    # add +2 days to correctly cover the last night shift crossing midnight
    end_base = datetime.combine(cal["date"].max() + timedelta(days=2), time(0, 0))
    base_date = cal["date"].min()

    # shifts lookup
    lookup: dict[tuple[str, str], dict[str, time]] = {}
    for _, row in shifts_df.iterrows():
        key = (str(row["shift_pattern"]), str(row["shift_type"]).upper())
        lookup[key] = {
            "start": _time_from(row["start_time"]),
            "end": _time_from(row["end_time"]),
            "break_start": _time_from(row["break_start"]),
            "break_end": _time_from(row["break_end"]),
        }

    out: dict[int, list[tuple[datetime, datetime]]] = {}
    for _, e in employees_df.iterrows():
        eid = int(e["employee_id"])
        pattern = str(e["shift_pattern"])
        cycle_anchor = int(e.get("cycle_anchor", 0))
        shift_type = resolve_shift_type(str(e["preferred_shift_type"]), bool(e["can_work_night"]), pattern, eid)
        sh = lookup.get((pattern, shift_type))
        if sh is None:
            out[eid] = [(start_base, end_base)]
            continue

        work_windows: list[tuple[datetime, datetime]] = []
        lunch_windows: list[tuple[datetime, datetime]] = []
        for _, c in cal.iterrows():
            d = c["date"]
            if not _is_work_day_for_pattern(
                d,
                pattern=pattern,
                cycle_anchor=cycle_anchor,
                is_workday_rf=int(c["is_workday_rf"]),
                base_date=base_date,
            ):
                continue
            ws, we = _make_interval(d, sh["start"], sh["end"])
            bs, be = _make_break_interval(
                shift_day=d,
                shift_start=sh["start"],
                shift_end=sh["end"],
                break_start=sh["break_start"],
                break_end=sh["break_end"],
            )
            work_windows.append((ws, we))
            lunch_windows.append((bs, be))

        # complement to OFFSHIFT
        work_windows.sort()
        off: list[tuple[datetime, datetime]] = []
        cursor = start_base
        for ws, we in work_windows:
            if cursor < ws:
                off.append((cursor, ws))
            cursor = max(cursor, we)
        if cursor < end_base:
            off.append((cursor, end_base))

        # vacations
        vac = vacations_df[vacations_df["employee_id"].astype(int) == eid]
        vac_windows: list[tuple[datetime, datetime]] = []
        for _, v in vac.iterrows():
            d0 = pd.to_datetime(v["date_from"]).date()
            d1 = pd.to_datetime(v["date_to"]).date()
            cur = d0
            while cur <= d1:
                vac_windows.append((datetime.combine(cur, time(0, 0)), datetime.combine(cur + timedelta(days=1), time(0, 0))))
                cur += timedelta(days=1)

        out[eid] = off + lunch_windows + vac_windows
        out[eid].sort(key=lambda x: x[0])
        # merge overlaps to keep list smaller
        merged: list[tuple[datetime, datetime]] = []
        for s, e2 in out[eid]:
            if not merged:
                merged.append((s, e2))
            else:
                ps, pe = merged[-1]
                if s <= pe:
                    merged[-1] = (ps, max(pe, e2))
                else:
                    merged.append((s, e2))
        out[eid] = merged

    return out


def build_machine_unavailability(downtime_df: pd.DataFrame) -> dict[int, list[tuple[datetime, datetime]]]:
    out: dict[int, list[tuple[datetime, datetime]]] = {}
    for mid, grp in downtime_df.groupby("machine_id"):
        rows = []
        for _, r in grp.iterrows():
            rows.append((pd.Timestamp(r["from_dt"]).to_pydatetime(), pd.Timestamp(r["to_dt"]).to_pydatetime()))
        rows.sort(key=lambda x: x[0])
        out[int(mid)] = rows
    return out

