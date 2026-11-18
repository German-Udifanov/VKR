from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
@dataclass(frozen=True)
class MLArtifacts:
    model: RandomForestRegressor
    feature_columns: list[str]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "month" not in x.columns and "created_at" in x.columns:
        x["month"] = pd.to_datetime(x["created_at"]).dt.month
    cat_cols = [c for c in ["product_type", "operation_name", "machine_type", "required_skill"] if c in x.columns]
    num_cols = [c for c in ["min_skill_level", "quantity", "month"] if c in x.columns]
    for c in cat_cols:
        x[c] = x[c].astype(str).str.upper()
    for c in num_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x_cat = pd.get_dummies(x[cat_cols], prefix=cat_cols, dtype=float) if cat_cols else pd.DataFrame(index=x.index)
    x_num = x[num_cols].astype(float) if num_cols else pd.DataFrame(index=x.index)
    return pd.concat([x_num, x_cat], axis=1)


def simulate_history(operations_route_df: pd.DataFrame, orders_df: pd.DataFrame, *, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ops = operations_route_df.copy()
    orders = orders_df.copy()
    orders["month"] = pd.to_datetime(orders["created_at"]).dt.month
    rows = []
    for _ in range(n):
        o = orders.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        prod = str(o["product_type"])
        qty = int(o["quantity"])
        month = int(o["month"])
        route = ops[ops["product_type"] == prod]
        t = route.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        base = 1.0 + 0.05 * (int(t["min_skill_level"]) - 1)
        season = 1.0 + (0.03 if month == 3 else 0.0)
        noise = float(rng.lognormal(mean=0.0, sigma=0.15))
        mult = base * season * noise
        rows.append(
            {
                "product_type": prod,
                "operation_name": str(t["operation_name"]),
                "machine_type": str(t["machine_type"]),
                "required_skill": str(t["required_skill"]).upper(),
                "min_skill_level": int(t["min_skill_level"]),
                "quantity": qty,
                "month": month,
                "duration_multiplier": mult,
            }
        )
    return pd.DataFrame(rows)


def train_duration_model(history_df: pd.DataFrame, *, seed: int = 42) -> MLArtifacts:
    y = history_df["duration_multiplier"].astype(float)
    x = _features(history_df)
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=seed,
        max_depth=16,
        min_samples_leaf=5,
        n_jobs=-1,
    )
    model.fit(x, y)
    return MLArtifacts(model=model, feature_columns=list(x.columns))


def predict_multiplier(artifacts: MLArtifacts, df: pd.DataFrame) -> pd.Series:
    x = _features(df)
    for c in artifacts.feature_columns:
        if c not in x.columns:
            x[c] = 0.0
    x = x[artifacts.feature_columns]
    return pd.Series(artifacts.model.predict(x), index=df.index, name="mult_pred").clip(lower=0.6, upper=1.8)


def apply_ml_to_route(
    operations_route_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    hist = simulate_history(operations_route_df, orders_df, n=5000, seed=seed)
    artifacts = train_duration_model(hist, seed=seed)

    orders = orders_df.copy()
    orders["month"] = pd.to_datetime(orders["created_at"]).dt.month
    pred_rows = []
    for _, op in operations_route_df.iterrows():
        prod = str(op["product_type"])
        subset = orders[orders["product_type"].astype(str) == prod]
        month = int(subset["month"].mode().iloc[0]) if not subset.empty else 1
        qty = int(subset["quantity"].median()) if not subset.empty else 10
        pred_rows.append(
            {
                "product_type": prod,
                "operation_name": str(op["operation_name"]),
                "machine_type": str(op["machine_type"]),
                "required_skill": str(op["required_skill"]).upper(),
                "min_skill_level": int(op["min_skill_level"]),
                "quantity": qty,
                "month": month,
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    mult = predict_multiplier(artifacts, pred_df)

    out = operations_route_df.copy()
    out["duration_min_per_unit"] = (out["duration_min_per_unit"].astype(float) * mult.values).round().astype(int).clip(lower=1)
    return out

