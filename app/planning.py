from __future__ import annotations

import math

import pandas as pd


def expand_orders_to_operations(
    orders_df: pd.DataFrame,
    operations_route_df: pd.DataFrame,
    *,
    step_min: int,
) -> pd.DataFrame:
  
    tmpl = operations_route_df.copy().sort_values(["product_type", "op_seq"])
    rows: list[dict[str, object]] = []
    op_id = 1
    for _, order in orders_df.sort_values("order_id").iterrows():
        product = str(order["product_type"])
        quantity = int(order["quantity"])
        route = tmpl[tmpl["product_type"] == product]
        if route.empty:
            raise ValueError(f"Нет маршрута для product_type='{product}'")
        for _, op in route.iterrows():
            dur_min = int(round(float(op["duration_min_per_unit"]) * quantity))
            dur_steps = max(1, int(math.ceil(dur_min / float(step_min))))
            rows.append(
                {
                    "op_id": op_id,
                    "order_id": int(order["order_id"]),
                    "op_seq": int(op["op_seq"]),
                    "operation_name": str(op["operation_name"]),
                    "machine_type": str(op["machine_type"]),
                    "required_skill": str(op["required_skill"]).upper(),
                    "min_skill_level": int(op["min_skill_level"]),
                    "duration_steps": dur_steps,
                    "created_at": pd.to_datetime(order["created_at"]),
                    "due_at": pd.to_datetime(order["due_at"]),
                    "late_penalty": float(order.get("late_penalty", 1.0)),
                }
            )
            op_id += 1
    return pd.DataFrame(rows).sort_values(["order_id", "op_seq"]).reset_index(drop=True)


def build_precedence_pairs(ops_df: pd.DataFrame) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for _, grp in ops_df.sort_values(["order_id", "op_seq"]).groupby("order_id"):
        ids = grp["op_id"].tolist()
        for i in range(len(ids) - 1):
            pairs.append((int(ids[i]), int(ids[i + 1])))
    return pairs

