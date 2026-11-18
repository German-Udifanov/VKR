"""
Standalone Gantt chart for the production schedule.

Usage:
  python gantt_chart.py
  python gantt_chart.py --input data_out/synthetic_medium_150_new.xlsx --output Raspisanie/gantt.html
  python gantt_chart.py --resource machine --date 2026-01-14 --time-limit 120
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_io import load_all_tables
from app.pipeline import run_schedule_without_ml
from app.planning import expand_orders_to_operations
from app.utils import RuntimeConfig, ensure_dir

ResourceAxis = Literal["employee", "machine"]


def prepare_gantt_dataframe(
    schedule_df: pd.DataFrame,
    *,
    employees_df: pd.DataFrame | None = None,
    ops_df: pd.DataFrame | None = None,
    resource: ResourceAxis = "employee",
    day: date | None = None,
) -> pd.DataFrame:
    """Build a flat table for timeline/Gantt (one row per scheduled operation)."""
    if schedule_df.empty:
        return pd.DataFrame(
            columns=[
                "resource_id",
                "resource_label",
                "operation_label",
                "start_dt",
                "end_dt",
                "order_id",
                "op_id",
                "machine_id",
                "employee_id",
            ]
        )

    df = schedule_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["end_dt"] = pd.to_datetime(df["end_dt"])

    if day is not None:
        df = df[df["start_dt"].dt.date == day]
        if df.empty:
            return df

    emp_names: dict[int, str] = {}
    if employees_df is not None and not employees_df.empty:
        emp_names = (
            employees_df[["employee_id", "name"]]
            .dropna()
            .assign(employee_id=lambda d: d["employee_id"].astype(int))
            .set_index("employee_id")["name"]
            .astype(str)
            .to_dict()
        )

    if ops_df is not None and "op_id" in ops_df.columns and "operation_name" in ops_df.columns:
        op_names = ops_df[["op_id", "operation_name"]].drop_duplicates("op_id")
        df = df.merge(op_names, on="op_id", how="left")
    if "operation_name" not in df.columns:
        df["operation_name"] = df["machine_type"]

    df["operation_label"] = (
        df["operation_name"].astype(str)
        + " ("
        + df["machine_type"].astype(str)
        + ", Z"
        + df["order_id"].astype(str)
        + ")"
    )

    if resource == "machine":
        df["resource_id"] = df["machine_id"].astype(int)
        df["resource_label"] = "Machine_" + df["machine_id"].astype(str)
    else:
        df["resource_id"] = df["employee_id"].astype(int)
        df["resource_label"] = df["employee_id"].map(
            lambda eid: f"{eid} — {emp_names.get(int(eid), f'Worker_{eid}')}"
        )

    sort_cols = ["resource_id", "start_dt", "order_id", "op_seq"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def build_gantt_figure(
    gantt_df: pd.DataFrame,
    *,
    title: str = "Gantt — production schedule",
    height_per_row: int = 28,
    min_height: int = 400,
    max_height: int = 2400,
) -> go.Figure:
    """Horizontal timeline chart: resource on Y, time on X."""
    if gantt_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title + " (no operations in range)",
            xaxis_title="Time",
            yaxis_title="Resource",
        )
        return fig

    n_resources = gantt_df["resource_label"].nunique()
    height = min(max_height, max(min_height, 120 + n_resources * height_per_row))

    fig = px.timeline(
        gantt_df,
        x_start="start_dt",
        x_end="end_dt",
        y="resource_label",
        color="operation_name",
        hover_data={
            "resource_label": False,
            "operation_name": True,
            "order_id": True,
            "machine_id": True,
            "employee_id": True,
            "machine_type": True,
            "start_dt": True,
            "end_dt": True,
            "operation_label": True,
        },
        title=title,
    )
    fig.update_yaxes(autorange="reversed", title="Resource")
    fig.update_xaxes(title="Time")
    fig.update_layout(
        height=height,
        legend_title_text="Operation",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return fig


def save_gantt_html(fig: go.Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return path


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return pd.Timestamp(value).date()


def _build_title(resource: ResourceAxis, day: date | None) -> str:
    axis = "employees" if resource == "employee" else "machines"
    if day is None:
        return f"Gantt — by {axis} (full horizon)"
    return f"Gantt — by {axis} ({day.isoformat()})"


def main(argv: list[str] | None = None) -> int:
    default_in = PROJECT_ROOT / "data_out" / "synthetic_medium_150_new.xlsx"
    default_out = PROJECT_ROOT / "Raspisanie" / "gantt.html"

    parser = argparse.ArgumentParser(description="Build an interactive Gantt chart (HTML).")
    parser.add_argument("--input", type=Path, default=default_in, help="Input Excel with orders/employees/…")
    parser.add_argument("--output", type=Path, default=default_out, help="Output HTML path")
    parser.add_argument(
        "--resource",
        choices=("employee", "machine"),
        default="employee",
        help="Y-axis: employees or machines",
    )
    parser.add_argument("--date", type=str, default=None, help="Filter to one calendar day (YYYY-MM-DD)")
    parser.add_argument("--time-limit", type=int, default=500, help="CP-SAT time limit (seconds)")
    parser.add_argument("--step-min", type=int, default=30, help="Scheduling step (minutes)")
    parser.add_argument("--workers", type=int, default=8, help="OR-Tools worker threads")
    parser.add_argument("--no-solve", action="store_true", help="Skip solver (requires --schedule-parquet)")
    parser.add_argument(
        "--schedule-parquet",
        type=Path,
        default=None,
        help="Load schedule_df from parquet instead of running the solver",
    )
    parser.add_argument("--open", action="store_true", help="Open HTML in the default browser")
    args = parser.parse_args(argv)

    day = _parse_date(args.date)
    resource: ResourceAxis = args.resource

    if args.no_solve or args.schedule_parquet:
        if not args.schedule_parquet or not args.schedule_parquet.is_file():
            print("Error: --schedule-parquet path required with --no-solve", file=sys.stderr)
            return 1
        schedule_df = pd.read_parquet(args.schedule_parquet)
        tables = load_all_tables(args.input)
        ops_df = expand_orders_to_operations(
            tables.orders, tables.operations, step_min=args.step_min
        )
        employees_df = tables.employees
    else:
        if not args.input.is_file():
            print(f"Error: input file not found: {args.input}", file=sys.stderr)
            return 1
        cfg = RuntimeConfig(
            input_xlsx=args.input,
            output_dir=args.output.parent,
            step_min=args.step_min,
            time_limit_sec=args.time_limit,
            workers=args.workers,
        )
        print("Running scheduler (CP-SAT)…")
        result = run_schedule_without_ml(cfg)
        schedule_df = result.schedule_df
        employees_df = result.tables.employees
        ops_df = expand_orders_to_operations(
            result.tables.orders, result.tables.operations, step_min=cfg.step_min
        )
        print(f"Solve done in {result.solve_seconds:.1f}s, operations: {len(schedule_df)}")

    gantt_df = prepare_gantt_dataframe(
        schedule_df,
        employees_df=employees_df,
        ops_df=ops_df,
        resource=resource,
        day=day,
    )
    if gantt_df.empty:
        print("Warning: no operations for the selected filters.", file=sys.stderr)

    fig = build_gantt_figure(gantt_df, title=_build_title(resource, day))
    out_path = save_gantt_html(fig, args.output)
    print(f"Gantt saved: {out_path}")

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
