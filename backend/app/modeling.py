from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

TARGET = "freight_rate_usd_per_tonne"
GROUP_COLS = ["route_id", "origin_port_id", "destination_port_id", "vessel_type"]
HORIZONS = [1, 2, 3, 4]
LAGS = [1, 2, 3, 6, 12]
ROLLING = [3, 6, 12]
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parents[1] / "data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", Path(__file__).parents[1] / "artifacts"))


def load_data() -> dict[str, pd.DataFrame]:
    names = ["vessels", "ports", "freight_rates", "market_indicators", "route_distances", "cargo_orders"]
    frames = {}
    for name in names:
        path = DATA_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Required dataset is missing: {path}")
        frames[name] = pd.read_csv(path)
    return frames


def prepare_monthly(rates: pd.DataFrame, indicators: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], pd.Timestamp]:
    raw = rates.merge(indicators, on="date", how="inner")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    raw = raw.dropna(subset=["date", TARGET]).copy()
    # The source files include simulated observations later than the current month.
    # Exclude them so the model does not train on information from the future.
    current_month = pd.Timestamp.today().to_period("M").to_timestamp()
    raw = raw.loc[raw["date"] <= current_month].copy()
    raw = raw.drop(columns=["rate_id"], errors="ignore")
    for col in GROUP_COLS:
        raw[col] = raw[col].fillna("Unknown").astype(str)
    numeric_cols = [c for c in raw.select_dtypes(include=[np.number]).columns if c != TARGET]
    agg = {c: "mean" for c in numeric_cols}
    agg[TARGET] = "mean"
    monthly = (raw.groupby(["date"] + GROUP_COLS, as_index=False, dropna=False)
               .agg(agg).sort_values(GROUP_COLS + ["date"]).reset_index(drop=True))
    for lag in LAGS:
        monthly[f"rate_lag_{lag}"] = monthly.groupby(GROUP_COLS, sort=False)[TARGET].shift(lag)
    for window in ROLLING:
        monthly[f"rate_rolling_mean_{window}"] = monthly.groupby(GROUP_COLS, sort=False)[TARGET].transform(
            lambda values, w=window: values.shift(1).rolling(w, min_periods=1).mean()
        )
    for h in HORIZONS:
        grouped = monthly.groupby(GROUP_COLS, sort=False)
        monthly[f"target_h{h}"] = grouped[TARGET].shift(-h)
        monthly[f"target_date_h{h}"] = grouped["date"].shift(-h)
    base_date = monthly.date.min()
    categorical = GROUP_COLS.copy()
    features = categorical + numeric_cols + [f"rate_lag_{n}" for n in LAGS] + [f"rate_rolling_mean_{n}" for n in ROLLING]
    features = list(dict.fromkeys(c for c in features if c in monthly.columns and c not in GROUP_COLS[:0]))
    features += ["target_month_sin", "target_month_cos", "target_time_index"]
    return monthly, features, categorical, base_date


def calendarize(frame: pd.DataFrame, dates: pd.Series, base_date: pd.Timestamp) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(dates)
    out["target_month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    out["target_month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    out["target_time_index"] = (dates.dt.year - base_date.year) * 12 + dates.dt.month - base_date.month
    return out


def make_pipeline(features: list[str], categorical: list[str]) -> Pipeline:
    cats = [c for c in categorical if c in features]
    nums = [c for c in features if c not in cats]
    prep = ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nums),
    ], remainder="drop")
    return Pipeline([("preprocessor", prep), ("xgb", XGBRegressor(
        n_estimators=350, max_depth=5, learning_rate=0.04, subsample=0.85,
        colsample_bytree=0.85, objective="reg:squarederror", random_state=42,
        n_jobs=2, verbosity=0,
    ))])


def train_and_save(frames: dict[str, pd.DataFrame]) -> dict:
    monthly, features, categorical, base_date = prepare_monthly(frames["freight_rates"], frames["market_indicators"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Measure performance on a chronological holdout before fitting the
    # production models. The holdout targets are strictly later than training
    # targets, matching how the model is used for future months.
    all_dates = np.array(sorted(monthly["date"].dropna().unique()))
    if len(all_dates) < 18:
        raise ValueError("At least 18 distinct months are required for a temporal model evaluation.")
    split_position = min(max(int(len(all_dates) * 0.80), 1), len(all_dates) - 5)
    split_date = pd.Timestamp(all_dates[split_position])
    last_historical_date = pd.Timestamp(all_dates[-1])
    trained = {}
    evaluation = {}
    for h in HORIZONS:
        target = f"target_h{h}"
        target_date_col = f"target_date_h{h}"
        target_date = pd.to_datetime(monthly[target_date_col])
        frame = calendarize(monthly, target_date, base_date)
        train_mask = frame[target].notna() & target_date.notna() & (target_date < split_date)
        test_mask = (frame[target].notna() & target_date.notna() &
                     (frame["date"] >= split_date) & (target_date <= last_historical_date))
        if not train_mask.any() or not test_mask.any():
            raise ValueError(f"Insufficient chronological train/test rows for horizon {h}.")

        evaluation_model = make_pipeline(features, categorical)
        evaluation_model.fit(frame.loc[train_mask, features], frame.loc[train_mask, target])
        actual = frame.loc[test_mask, target].to_numpy(dtype=float)
        predicted = evaluation_model.predict(frame.loc[test_mask, features]).astype(float)
        absolute_error = np.abs(actual - predicted)
        nonzero = np.abs(actual) > 1e-9
        evaluation[f"month_{h}"] = {
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "mae_usd_per_tonne": float(np.mean(absolute_error)),
            "rmse_usd_per_tonne": float(np.sqrt(np.mean(np.square(actual - predicted)))),
            "r2": float(1.0 - np.sum(np.square(actual - predicted)) /
                        np.sum(np.square(actual - np.mean(actual)))) if len(actual) > 1 and np.var(actual) > 0 else None,
            "wape_pct": float(100.0 * np.sum(absolute_error) / max(np.sum(np.abs(actual)), 1e-9)),
            "within_10pct_pct": float(100.0 * np.mean(absolute_error[nonzero] <= 0.10 * np.abs(actual[nonzero]))) if nonzero.any() else None,
        }

        valid = frame[target].notna() & target_date.notna()
        model = make_pipeline(features, categorical)
        model.fit(frame.loc[valid, features], frame.loc[valid, target])
        joblib.dump(model, ARTIFACT_DIR / f"model_h{h}.joblib")
        trained[h] = model
    metadata = {"target": TARGET, "horizons": HORIZONS, "features": features,
                "categorical_features": categorical, "base_date": base_date.strftime("%Y-%m-%d"),
                "history_start": monthly.date.min().strftime("%Y-%m-%d"),
                "history_end": monthly.date.max().strftime("%Y-%m-%d"),
                "training_rows": int(len(monthly)),
                "evaluation_method": "chronological holdout; training targets precede the split and test origins begin at the split",
                "evaluation_split_date": split_date.strftime("%Y-%m-%d"),
                "holdout_metrics": evaluation}
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def load_models() -> tuple[dict[int, Pipeline], dict]:
    metadata_path = ARTIFACT_DIR / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("Model artifacts are missing. Run `python train.py` first.")
    metadata = json.loads(metadata_path.read_text())
    models = {h: joblib.load(ARTIFACT_DIR / f"model_h{h}.joblib") for h in HORIZONS}
    return models, metadata


def forecast_for_series(monthly: pd.DataFrame, features: list[str], base_date: pd.Timestamp,
                        models: dict[int, Pipeline], route_id: str, origin_id: str,
                        destination_id: str, vessel_type: str, first_forecast: pd.Timestamp,
                        route_context: Optional[dict] = None) -> list[dict]:
    mask = (monthly.route_id == route_id) & (monthly.origin_port_id == origin_id) & \
           (monthly.destination_port_id == destination_id) & (monthly.vessel_type == vessel_type)
    history = monthly.loc[mask].sort_values("date")
    forecast_basis = "historical_route"
    if history.empty:
        # The training data does not contain every selectable port pair. Use the
        # nearest historical route for the same vessel class as an input baseline,
        # then let the trained model predict with the requested route identifiers
        # and route-specific measurements. This remains a model prediction rather
        # than returning null or inventing a fixed freight rate.
        analogs = monthly.loc[monthly.vessel_type.astype(str) == str(vessel_type)].copy()
        if analogs.empty:
            return []

        latest_analogs = (analogs.sort_values("date")
                          .groupby(GROUP_COLS, as_index=False, dropna=False)
                          .tail(1).copy())
        same_origin = latest_analogs.origin_port_id.astype(str) == str(origin_id)
        same_destination = latest_analogs.destination_port_id.astype(str) == str(destination_id)
        if same_origin.any():
            latest_analogs = latest_analogs.loc[same_origin]
        elif same_destination.any():
            latest_analogs = latest_analogs.loc[same_destination]

        if route_context and "distance_nm" in route_context and "distance_nm" in latest_analogs:
            latest_analogs["_distance_gap"] = (
                pd.to_numeric(latest_analogs.distance_nm, errors="coerce") -
                float(route_context["distance_nm"])
            ).abs()
            latest_analogs = latest_analogs.sort_values("_distance_gap", na_position="last")

        analog = latest_analogs.iloc[0]
        analog_mask = pd.Series(True, index=monthly.index)
        for column in GROUP_COLS:
            analog_mask &= monthly[column].astype(str) == str(analog[column])
        history = monthly.loc[analog_mask].sort_values("date")
        forecast_basis = "similar_route_model"

    last = history.iloc[-1]
    actual = history[TARGET].astype(float).tolist()
    outputs = []
    for h in HORIZONS:
        month = first_forecast + pd.DateOffset(months=h - 1)
        row = last.to_dict()
        # Categorical identifiers and route-specific numeric features must describe
        # the requested route, even when lag history came from an analog route.
        row.update({
            "route_id": str(route_id),
            "origin_port_id": str(origin_id),
            "destination_port_id": str(destination_id),
        })
        if route_context:
            row.update(route_context)
        for lag in LAGS:
            row[f"rate_lag_{lag}"] = actual[-lag] if len(actual) >= lag else np.nan
        for window in ROLLING:
            row[f"rate_rolling_mean_{window}"] = float(np.mean(actual[-window:])) if actual else np.nan
        row["target_month_sin"] = np.sin(2 * np.pi * month.month / 12)
        row["target_month_cos"] = np.cos(2 * np.pi * month.month / 12)
        row["target_time_index"] = (month.year - base_date.year) * 12 + month.month - base_date.month
        x = pd.DataFrame([{feature: row.get(feature, np.nan) for feature in features}])
        rate = max(0.0, float(models[h].predict(x[features])[0]))
        outputs.append({"month": month.strftime("%Y-%m"), "horizon_months": h,
                        "freight_rate_usd_per_tonne": round(rate, 3),
                        "forecast_basis": forecast_basis})
    return outputs
