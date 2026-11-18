from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


PRODUCT_TYPES = ["TABLE", "WARDROBE", "CABINET", "SHELF", "KITCHEN_MODULE"]


@dataclass(frozen=True)
class OrderClassSpec:
    name: str
    count: int
    due_days_lo: int
    due_days_hi: int
    qty_lo: int
    qty_hi: int


CLASS_SPECS = [
    OrderClassSpec("short", 35, 5, 7, 5, 10),
    OrderClassSpec("medium", 70, 8, 14, 10, 20),
    OrderClassSpec("large", 45, 14, 21, 20, 50),
]


def random_datetime_on_date(d, rng: random.Random) -> datetime:
    hh = rng.randrange(8, 18)
    mm = rng.choice([0, 15, 30, 45])
    return datetime(d.year, d.month, d.day, hh, mm, 0)


def build_created_dates(start_date, end_date) -> list:
    days = (end_date - start_date).days
    return [start_date + timedelta(days=i) for i in range(days + 1)]


def generate_orders(calendar_df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    cal = calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.date

    start_date = cal["date"].min()
    end_date = cal["date"].max()

    all_days = build_created_dates(start_date, end_date)

    # Make sure orders exist on most days: sample created_at days with replacement,
    # biased towards workdays (but still allow some on weekends).
    # Also add "seasonality" and a few peak days (useful for diploma demo).
    is_work = {row["date"]: bool(row["is_workday_rf"]) for _, row in cal.iterrows()}
    month_factor = {1: 0.7, 2: 0.8, 3: 1.0}
    peak_days = set(rng.sample(all_days, k=min(6, len(all_days))))
    weights = []
    for d in all_days:
        w = 2.0 if is_work.get(d, True) else 0.8
        w *= month_factor.get(d.month, 1.0)
        if d in peak_days:
            w *= 3.0
        weights.append(w)
    total_orders = sum(s.count for s in CLASS_SPECS)

    created_days = rng.choices(all_days, weights=weights, k=total_orders)
    rng.shuffle(created_days)

    classes = []
    for spec in CLASS_SPECS:
        classes += [spec.name] * spec.count
    rng.shuffle(classes)

    spec_by_name = {s.name: s for s in CLASS_SPECS}

    rows = []
    for i in range(total_orders):
        klass = classes[i]
        spec = spec_by_name[klass]
        created_at = random_datetime_on_date(created_days[i], rng)
        due_at = created_at + timedelta(days=rng.randint(spec.due_days_lo, spec.due_days_hi))

        product_type = rng.choice(PRODUCT_TYPES)
        quantity = rng.randint(spec.qty_lo, spec.qty_hi)
        # Priority distribution: short orders more often urgent, large more often moderate.
        if klass == "short":
            pr_w = [0.05, 0.12, 0.22, 0.32, 0.29]
        elif klass == "large":
            pr_w = [0.18, 0.25, 0.28, 0.19, 0.10]
        else:
            pr_w = [0.12, 0.2, 0.3, 0.25, 0.13]
        priority = rng.choices([1, 2, 3, 4, 5], weights=pr_w, k=1)[0]
        late_penalty = float(priority) * 10.0

        rows.append(
            {
                "order_id": i + 1,
                "created_at": created_at,
                "due_at": due_at,
                "product_type": product_type,
                "quantity": quantity,
                "priority": priority,
                "late_penalty": late_penalty,
                "order_class": klass,
            }
        )

    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    return df


def main() -> int:
    inp = Path("data_out/synthetic_medium.xlsx")
    out = Path("data_out/synthetic_medium_150.xlsx")
    if not inp.exists():
        raise FileNotFoundError(f"Missing input: {inp}")

    xls = pd.ExcelFile(inp)
    required = ["employees", "machines", "orders", "vacations", "operations", "calendar", "shifts", "downtime"]
    tables = {s: pd.read_excel(inp, sheet_name=s) for s in required}

    orders_150 = generate_orders(tables["calendar"], seed=42)
    tables["orders"] = orders_150

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for name in required:
                tables[name].to_excel(writer, sheet_name=name, index=False)
    except PermissionError:
        # On Windows Excel often locks the file. Save to a new name instead.
        out2 = out.with_name(out.stem + "_new.xlsx")
        with pd.ExcelWriter(out2, engine="openpyxl") as writer:
            for name in required:
                tables[name].to_excel(writer, sheet_name=name, index=False)
        out = out2

    print(f"Wrote: {out} (orders={len(orders_150)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

