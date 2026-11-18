from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.export import RU_WEEKDAYS, build_schedule_table_dataframe
from app.pipeline import run_schedule_without_ml
from app.utils import RuntimeConfig

st.set_page_config(page_title="Расписание производства", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    h1 { font-weight: 600; letter-spacing: -0.02em; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Расписание производства")
st.caption("Те же данные и экспорт, что и в Excel по дням — плюс сохранение файла в папку «Raspisanie».")

with st.sidebar:
    st.header("Входные данные")
    default_in = PROJECT_ROOT / "data_out" / "synthetic_medium_150_new.xlsx"
    input_xlsx = Path(
        st.text_input("Путь к входному Excel", value=str(default_in), help="Файл с листами employees, orders, …")
    )
    default_out = PROJECT_ROOT / "Raspisanie"
    output_dir = Path(st.text_input("Папка для выходного Excel", value=str(default_out)))

    st.header("Решатель")
    step_min = st.number_input("Шаг, мин", min_value=5, max_value=120, value=30, step=5)
    time_limit_sec = st.number_input("Лимит времени, сек", min_value=30, value=500, step=30)
    workers = st.number_input("Потоки OR-Tools", min_value=1, max_value=32, value=8)

    st.subheader("Веса цели")
    w_tardiness = st.number_input("w_tardiness", min_value=1, value=80, step=5)
    w_balance = st.number_input("w_balance", min_value=1, value=30, step=1)
    w_idle_day = st.number_input("w_idle_day", min_value=0, value=2000, step=100)
    w_low_work_day = st.number_input("w_low_work_day", min_value=0, value=15, step=1)
    w_emp_idle_day = st.number_input("w_emp_idle_day", min_value=0, value=120, step=10)
    min_work_hours = st.number_input("min_work_hours_per_day", min_value=0.0, value=6.0, step=0.5)
    two_pass = st.toggle("Двухфазная оптимизация (two_pass)", value=True)

    run_btn = st.button("Построить расписание и Excel", type="primary", use_container_width=True)

if run_btn:
    if not input_xlsx.is_file():
        st.error(f"Файл не найден: {input_xlsx}")
        st.stop()

    cfg = RuntimeConfig(
        input_xlsx=input_xlsx,
        output_dir=output_dir,
        step_min=int(step_min),
        time_limit_sec=int(time_limit_sec),
        workers=int(workers),
        w_tardiness=int(w_tardiness),
        w_balance=int(w_balance),
        w_idle_day=int(w_idle_day),
        w_low_work_day=int(w_low_work_day),
        min_work_hours_per_day=float(min_work_hours),
        w_emp_idle_day=int(w_emp_idle_day),
        two_pass=bool(two_pass),
    )

    with st.spinner("Загрузка данных и CP-SAT… это может занять несколько минут."):
        try:
            result = run_schedule_without_ml(cfg)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()

    st.session_state["last_result"] = result
    st.session_state["display_df"] = build_schedule_table_dataframe(
        result.schedule_df,
        employees_df=result.tables.employees,
        calendar_df=result.tables.calendar,
        vacations_df=result.tables.vacations,
    )

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    disp = st.session_state["display_df"]

    st.success(f"Excel сохранён: `{res.excel_path}`")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Время решения, с", f"{res.solve_seconds:.1f}")
    m2.metric("Gini баланса", f"{res.kpi.balance_score:.3f}")
    m3.metric("Доля опозданий", f"{res.kpi.late_rate:.1%}")
    m4.metric("Простой по паттерну", f"{res.kpi.pattern_idle_rate:.1%}")

    st.subheader("Проверки")
    for c in res.checks:
        st.write(("**OK**" if c.ok else "**FAIL**") + f" — {c.name} (count={c.count})")

    st.subheader("Расписание по дням")
    st.download_button(
        label="Скачать тот же Excel",
        data=res.excel_path.read_bytes(),
        file_name=res.excel_path.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    dates = sorted({pd.Timestamp(d).date() for d in disp["Дата"].tolist() if pd.notna(d)})
    if not dates:
        st.warning("Нет строк для отображения.")
    else:
        pick = st.selectbox(
            "День для просмотра",
            options=dates,
            format_func=lambda d: f"{d.strftime('%Y-%m-%d')} ({RU_WEEKDAYS.get(d.weekday(), '')})",
        )

        sub = disp.loc[
            pd.to_datetime(disp["Дата"]).dt.date == pick,
            [c for c in disp.columns if c != "Дата"],
        ].copy()
        for col in ("Начало этапа в операции", "Конец этапа операции"):
            if col in sub.columns:
                sub[col] = pd.to_datetime(sub[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(sub, use_container_width=True, hide_index=True, height=min(520, 40 + len(sub) * 36))

    if not disp.empty:
        with st.expander("Все дни (сводная таблица)"):
            full = disp.copy()
            for col in ("Начало этапа в операции", "Конец этапа операции"):
                if col in full.columns:
                    full[col] = pd.to_datetime(full[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            full["Дата"] = pd.to_datetime(full["Дата"]).dt.strftime("%Y-%m-%d")
            st.dataframe(full, use_container_width=True, hide_index=True, height=480)
else:
    st.info("Укажите пути в боковой панели и нажмите **Построить расписание и Excel**.")
