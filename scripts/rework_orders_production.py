"""
Пересборка листа orders под ёмкость производства (тот же xlsx).

Идеи:
- даты создания — равномерно по рабочим дням календаря (без «пиковых» дней);
- сроки — чуть больше воздуха (lead time), верхняя граница не уезжает далеко за календарь;
- объёмы — умеренные верхние пределы для «large», меньше экстремальных 50;
- mix product_type — чаще лёгкие маршруты (SHELF, TABLE…), реже самые длинные (WARDROBE, KITCHEN_MODULE).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from app.data_io import load_all_tables, validate_tables

PRODUCT_TYPES = ["TABLE", "WARDROBE", "CABINET", "SHELF", "KITCHEN_MODULE"]

# суммарная длительность шаблона мин/ед (из operations) — для весов mix
ROUTE_WEIGHT_PER_UNIT = {
    "SHELF": 31,
    "CABINET": 74,
    "TABLE": 69,
    "WARDROBE": 120,
    "KITCHEN_MODULE": 102,
}


@dataclass(frozen=True)
class OrderClassSpec:
    name: str
    count: int
    due_days_lo: int
    due_days_hi: int
    qty_lo: int
    qty_hi: int


CLASS_SPECS = [
    OrderClassSpec("short", 35, 8, 13, 5, 12),
    OrderClassSpec("medium", 70, 12, 20, 8, 18),
    OrderClassSpec("large", 45, 18, 28, 12, 32),
]


def random_datetime_on_date(d, rng: random.Random) -> datetime:
    hh = rng.randrange(8, 18)
    mm = rng.choice([0, 15, 30, 45])
    return datetime(d.year, d.month, d.day, hh, mm, 0)


def generate_orders(calendar_df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    cal = calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    end_date = cal["date"].max()
    is_work = {row["date"]: bool(int(row["is_workday_rf"])) for _, row in cal.iterrows()}
    workdays = [d for d in sorted(is_work) if is_work.get(d, True)]
    if not workdays:
        workdays = sorted(is_work.keys())

    total_orders = sum(s.count for s in CLASS_SPECS)
    created_days = rng.choices(workdays, k=total_orders)
    rng.shuffle(created_days)

    classes: list[str] = []
    for spec in CLASS_SPECS:
        classes += [spec.name] * spec.count
    rng.shuffle(classes)

    spec_by_name = {s.name: s for s in CLASS_SPECS}

    min_each = 18
    quota: dict[str, int] = {p: min_each for p in PRODUCT_TYPES}
    remaining = total_orders - sum(quota.values())
    extra_assign: list[str] = []
    heavy = {p: 1.0 / max(1, ROUTE_WEIGHT_PER_UNIT.get(p, 80)) for p in PRODUCT_TYPES}
    sw = sum(heavy.values())
    pw = [heavy[p] / sw for p in PRODUCT_TYPES]
    for _ in range(remaining):
        extra_assign.append(rng.choices(PRODUCT_TYPES, weights=pw, k=1)[0])
    rng.shuffle(extra_assign)
    product_pool = [p for p in PRODUCT_TYPES for _ in range(quota[p])] + extra_assign
    rng.shuffle(product_pool)

    last_cal = end_date
    soft_due_cap_date = last_cal + timedelta(days=28)
    rows: list[dict] = []

    for i in range(total_orders):
        klass = classes[i]
        spec = spec_by_name[klass]
        created_at = random_datetime_on_date(created_days[i], rng)
        lead = rng.randint(spec.due_days_lo, spec.due_days_hi)
        due_at = created_at + timedelta(days=lead)
        if due_at.date() > soft_due_cap_date:
            due_at = datetime(soft_due_cap_date.year, soft_due_cap_date.month, soft_due_cap_date.day, 17, 0, 0)
        if due_at <= created_at:
            due_at = created_at + timedelta(days=max(spec.due_days_lo, 1))

        product_type = product_pool[i]

        quantity = rng.randint(spec.qty_lo, spec.qty_hi)
        if klass == "short":
            pr_w = [0.08, 0.18, 0.28, 0.32, 0.14]
        elif klass == "large":
            pr_w = [0.22, 0.28, 0.28, 0.17, 0.05]
        else:
            pr_w = [0.14, 0.24, 0.32, 0.22, 0.08]
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

    return pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", type=Path, nargs="?", default=Path("data_out/synthetic_medium_150_new.xlsx"))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    path = (_ROOT / args.xlsx).resolve() if not args.xlsx.is_absolute() else args.xlsx.resolve()
    if not path.is_file():
        raise SystemExit(f"Не найден: {path}")

    if not args.no_backup:
        bak = path.with_name(path.stem + "_orders_rework.bak" + path.suffix)
        shutil.copy2(path, bak)
        print(f"Резервная копия: {bak}")

    sheets: dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(path) as xls:
        for name in xls.sheet_names:
            sheets[name] = pd.read_excel(xls, sheet_name=name)

    new_orders = generate_orders(sheets["calendar"], seed=args.seed)
    sheets["orders"] = new_orders

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    t = load_all_tables(path)
    validate_tables(t)
    print(f"Обновлён лист orders: {path} ({len(new_orders)} строк)")
    print("created_at:", new_orders["created_at"].min(), "…", new_orders["created_at"].max())
    print("due_at:", new_orders["due_at"].min(), "…", new_orders["due_at"].max())
    print("product_type:\n", new_orders["product_type"].value_counts())


if __name__ == "__main__":
    main()
