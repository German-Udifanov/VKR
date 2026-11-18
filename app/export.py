from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from app.calendar_mod import _is_work_day_for_pattern, resolve_shift_type
from app.utils import shift_type_by_time


RU_WEEKDAYS = {
    0: "Понедельник",
    1: "Вторник",
    2: "Среда",
    3: "Четверг",
    4: "Пятница",
    5: "Суббота",
    6: "Воскресенье",
}


HEADERS = [
    "День недели",
    "ID сотрудника",
    "ФИО сотрудника",
    "Тип смены",
    "ID заказа",
    "ID станка",
    "Начало этапа в операции",
    "Конец этапа операции",
    "Тип станка операции",
    "ID операции",
    "duration_steps",
    "Примечание",
]


def _vacation_on_date(vacations_df: pd.DataFrame | None, employee_id: int, d: date) -> bool:
    if vacations_df is None or vacations_df.empty:
        return False
    vac = vacations_df[vacations_df["employee_id"].astype(int) == int(employee_id)]
    for _, v in vac.iterrows():
        d0 = pd.to_datetime(v["date_from"]).date()
        d1 = pd.to_datetime(v["date_to"]).date()
        if d0 <= d <= d1:
            return True
    return False


def build_schedule_table_dataframe(
    schedule_df: pd.DataFrame,
    *,
    employees_df: pd.DataFrame | None = None,
    calendar_df: pd.DataFrame | None = None,
    vacations_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Плоская таблица с теми же строками, что и лист «Расписание» в export_daily_excel
    (колонка «Дата» — календарный день блока).
    """
    if schedule_df.empty:
        return pd.DataFrame(columns=["Дата", *HEADERS])

    df = schedule_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["end_dt"] = pd.to_datetime(df["end_dt"])
    df["work_date"] = df["start_dt"].dt.date

    emp_name: dict[int, str] = {}
    if employees_df is not None and not employees_df.empty:
        emp_name = (
            employees_df[["employee_id", "name"]]
            .dropna()
            .assign(employee_id=lambda d: d["employee_id"].astype(int))
            .set_index("employee_id")["name"]
            .astype(str)
            .to_dict()
        )

    base_date: date | None = None
    rf_work: dict[date, int] = {}
    if calendar_df is not None and not calendar_df.empty:
        c = calendar_df.copy()
        c["date"] = pd.to_datetime(c["date"]).dt.date
        base_date = c["date"].min()
        for _, r in c.iterrows():
            rf_work[r["date"]] = int(r["is_workday_rf"])

    d_min = df["work_date"].min()
    d_max = df["work_date"].max()
    assert isinstance(d_min, date) and isinstance(d_max, date)

    rows_out: list[dict[str, object]] = []
    d_cur = d_min
    while d_cur <= d_max:
        g = df[df["work_date"] == d_cur].sort_values("start_dt")
        assert isinstance(d_cur, date)

        if employees_df is not None and not employees_df.empty and base_date is not None:
            irf = int(rf_work.get(d_cur, 1))
            for _, er in employees_df.sort_values("employee_id").iterrows():
                eid = int(er["employee_id"])
                pat = str(er.get("shift_pattern", "2/2")).strip()
                anchor = int(er.get("cycle_anchor", 0))
                if not _is_work_day_for_pattern(
                    d_cur,
                    pattern=pat,
                    cycle_anchor=anchor,
                    is_workday_rf=irf,
                    base_date=base_date,
                ):
                    continue
                full_name = str(emp_name.get(eid, "") or "")
                stype = resolve_shift_type(
                    str(er.get("preferred_shift_type", "DAY")),
                    bool(er.get("can_work_night", False)),
                    pat,
                    eid,
                )
                if _vacation_on_date(vacations_df, eid, d_cur):
                    weekday = RU_WEEKDAYS.get(int(d_cur.weekday()), "")
                    values = [
                        weekday,
                        eid,
                        full_name,
                        stype,
                        None,
                        None,
                        None,
                        None,
                        "",
                        None,
                        None,
                        "Отпуск",
                    ]
                    rows_out.append({"Дата": d_cur, **dict(zip(HEADERS, values))})
                    continue

                sub = g[g["employee_id"].astype(int) == eid]
                if sub.empty:
                    weekday = RU_WEEKDAYS.get(int(d_cur.weekday()), "")
                    values = [
                        weekday,
                        eid,
                        full_name,
                        stype,
                        None,
                        None,
                        None,
                        None,
                        "",
                        None,
                        None,
                        "Простой: по графику рабочий день, операции не назначены",
                    ]
                    rows_out.append({"Дата": d_cur, **dict(zip(HEADERS, values))})
                else:
                    for _, r in sub.iterrows():
                        weekday = RU_WEEKDAYS.get(int(pd.Timestamp(r["start_dt"]).dayofweek), "")
                        shift_type = shift_type_by_time(pd.Timestamp(r["start_dt"]).to_pydatetime())
                        eeid = int(r["employee_id"]) if pd.notna(r["employee_id"]) else None
                        fn = emp_name.get(int(eeid), "") if eeid is not None else ""
                        values = [
                            weekday,
                            eeid,
                            fn,
                            shift_type,
                            int(r["order_id"]) if pd.notna(r["order_id"]) else None,
                            int(r["machine_id"]) if pd.notna(r["machine_id"]) else None,
                            pd.Timestamp(r["start_dt"]).to_pydatetime(),
                            pd.Timestamp(r["end_dt"]).to_pydatetime(),
                            str(r.get("machine_type", "")),
                            int(r["op_id"]) if pd.notna(r["op_id"]) else None,
                            int(r["duration_steps"]) if pd.notna(r["duration_steps"]) else None,
                            "",
                        ]
                        rows_out.append({"Дата": d_cur, **dict(zip(HEADERS, values))})
        else:
            for _, r in g.iterrows():
                weekday = RU_WEEKDAYS.get(int(pd.Timestamp(r["start_dt"]).dayofweek), "")
                shift_type = shift_type_by_time(pd.Timestamp(r["start_dt"]).to_pydatetime())
                eid = int(r["employee_id"]) if pd.notna(r["employee_id"]) else None
                full_name = emp_name.get(int(eid), "") if eid is not None else ""
                values = [
                    weekday,
                    eid,
                    full_name,
                    shift_type,
                    int(r["order_id"]) if pd.notna(r["order_id"]) else None,
                    int(r["machine_id"]) if pd.notna(r["machine_id"]) else None,
                    pd.Timestamp(r["start_dt"]).to_pydatetime(),
                    pd.Timestamp(r["end_dt"]).to_pydatetime(),
                    str(r.get("machine_type", "")),
                    int(r["op_id"]) if pd.notna(r["op_id"]) else None,
                    int(r["duration_steps"]) if pd.notna(r["duration_steps"]) else None,
                    "",
                ]
                rows_out.append({"Дата": d_cur, **dict(zip(HEADERS, values))})

        d_cur = d_cur + timedelta(days=1)

    return pd.DataFrame(rows_out)


def _write_daily_schedule_sheet(
    ws: Worksheet,
    schedule_df: pd.DataFrame,
    *,
    employees_df: pd.DataFrame | None = None,
    calendar_df: pd.DataFrame | None = None,
    vacations_df: pd.DataFrame | None = None,
) -> None:
    disp = build_schedule_table_dataframe(
        schedule_df,
        employees_df=employees_df,
        calendar_df=calendar_df,
        vacations_df=vacations_df,
    )
    if disp.empty:
        ws.cell(row=1, column=1, value="(schedule is empty)")
        return

    d_min = disp["Дата"].min()
    d_max = disp["Дата"].max()
    assert isinstance(d_min, date) and isinstance(d_max, date)

    row_cursor = 1
    d_cur = d_min
    while d_cur <= d_max:
        grp = disp[disp["Дата"] == d_cur]
        ws.cell(row=row_cursor, column=1, value=datetime(d_cur.year, d_cur.month, d_cur.day))
        row_cursor += 1
        for j, h in enumerate(HEADERS, start=1):
            ws.cell(row=row_cursor, column=j, value=h)
        row_cursor += 1
        for _, r in grp.iterrows():
            for j, h in enumerate(HEADERS, start=1):
                ws.cell(row=row_cursor, column=j, value=r[h])
            row_cursor += 1
        row_cursor += 3
        d_cur = d_cur + timedelta(days=1)


def export_daily_excel(
    out_path: str | Path,
    *,
    schedule_df: pd.DataFrame,
    employees_df: pd.DataFrame | None = None,
    calendar_df: pd.DataFrame | None = None,
    vacations_df: pd.DataFrame | None = None,
    kpi_sheets: dict[str, pd.DataFrame] | None = None,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Расписание"
    _write_daily_schedule_sheet(
        ws,
        schedule_df,
        employees_df=employees_df,
        calendar_df=calendar_df,
        vacations_df=vacations_df,
    )

    if kpi_sheets:
        for sheet_name, df in kpi_sheets.items():
            ws_k = wb.create_sheet(sheet_name[:31])
            if df is None or df.empty:
                ws_k.cell(row=1, column=1, value="(empty)")
                continue
            for j, col in enumerate(df.columns.tolist(), start=1):
                ws_k.cell(row=1, column=j, value=str(col))
            for i, row in enumerate(df.itertuples(index=False), start=2):
                for j, v in enumerate(row, start=1):
                    ws_k.cell(row=i, column=j, value=v)
    wb.save(out)


def export_comparison_excel(
    out_path: str | Path,
    *,
    schedule_base: pd.DataFrame,
    schedule_ml: pd.DataFrame,
    employees_df: pd.DataFrame | None = None,
    calendar_df: pd.DataFrame | None = None,
    vacations_df: pd.DataFrame | None = None,
    kpi_base: dict[str, object],
    kpi_ml: dict[str, object],
    time_base_sec: float,
    time_ml_sec: float,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Расписание_без_ML"
    _write_daily_schedule_sheet(
        ws1,
        schedule_base,
        employees_df=employees_df,
        calendar_df=calendar_df,
        vacations_df=vacations_df,
    )

    ws2 = wb.create_sheet("Расписание_с_ML")
    _write_daily_schedule_sheet(
        ws2,
        schedule_ml,
        employees_df=employees_df,
        calendar_df=calendar_df,
        vacations_df=vacations_df,
    )

    ws3 = wb.create_sheet("Сравнение_KPI")
    rows = [
        ("metric", "без ML", "с ML"),
        ("time_sec", round(time_base_sec, 2), round(time_ml_sec, 2)),
        ("balance_gini", kpi_base.get("balance_gini"), kpi_ml.get("balance_gini")),
        ("late_rate", kpi_base.get("late_rate"), kpi_ml.get("late_rate")),
        ("avg_delay_h", kpi_base.get("avg_delay_h"), kpi_ml.get("avg_delay_h")),
        ("pattern_idle", kpi_base.get("pattern_idle"), kpi_ml.get("pattern_idle")),
        ("late_orders", kpi_base.get("late_orders"), kpi_ml.get("late_orders")),
        ("orders_total", kpi_base.get("orders_total"), kpi_ml.get("orders_total")),
    ]
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row, start=1):
            ws3.cell(row=i, column=j, value=v)

    # Optional: add workload detail sheets for both scenarios (useful for diploma)
    def _write_df(sheet: Worksheet, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            sheet.cell(row=1, column=1, value="(empty)")
            return
        for j, col in enumerate(df.columns.tolist(), start=1):
            sheet.cell(row=1, column=j, value=str(col))
        for i, row in enumerate(df.itertuples(index=False), start=2):
            for j, v in enumerate(row, start=1):
                sheet.cell(row=i, column=j, value=v)

    if isinstance(kpi_base.get("workload_total"), pd.DataFrame):
        _write_df(wb.create_sheet("WL_total_base"), kpi_base["workload_total"])  # type: ignore[index]
    if isinstance(kpi_base.get("workload_day"), pd.DataFrame):
        _write_df(wb.create_sheet("WL_day_base"), kpi_base["workload_day"])  # type: ignore[index]
    if isinstance(kpi_base.get("workload_week"), pd.DataFrame):
        _write_df(wb.create_sheet("WL_week_base"), kpi_base["workload_week"])  # type: ignore[index]
    if isinstance(kpi_ml.get("workload_total"), pd.DataFrame):
        _write_df(wb.create_sheet("WL_total_ml"), kpi_ml["workload_total"])  # type: ignore[index]
    if isinstance(kpi_ml.get("workload_day"), pd.DataFrame):
        _write_df(wb.create_sheet("WL_day_ml"), kpi_ml["workload_day"])  # type: ignore[index]
    if isinstance(kpi_ml.get("workload_week"), pd.DataFrame):
        _write_df(wb.create_sheet("WL_week_ml"), kpi_ml["workload_week"])  # type: ignore[index]

    wb.save(out)

