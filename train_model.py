"""Train the exhaust-heat forecaster: Q_in at t + 1 h and t + 4 h for the WHR dispatcher.

Pipeline
  1. load factory_sensor_data.csv (15-minute telemetry, ground-truth q_in_kw)
  2. causal feature engineering - row t only uses readings up to t:
       rolling mean / std over 15 min, 1 h and 4 h (exhaust temperature, load, Q_in),
       thermal gradients dT/dt, lags, cyclic hour-of-day / day-of-week for origin and target time
  3. chronological split: the last 7 days are a hold-out test set; training rows whose target
     falls inside the test period are purged
  4. one LightGBM model per horizon (early stopping on the tail of the training period),
     walk-forward CV on the training period, hold-out metrics against two naive baselines
  5. save everything the dashboard needs to heat_forecaster.pkl (joblib)

Usage
    python train_model.py
    python train_model.py --data factory_sensor_data.csv --out heat_forecaster.pkl --test-days 7
"""
from __future__ import annotations

import argparse
import inspect
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from dataset_builder import CP_EXHAUST_KJ_KGK, OUTPUT_CSV, STEP_MINUTES, STEPS_PER_DAY, day_type, waste_heat_kw

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - fallback keeps the pipeline runnable without LightGBM
    lgb = None
    from sklearn.ensemble import HistGradientBoostingRegressor

MODEL_PATH = "heat_forecaster.pkl"
TARGET = "q_in_kw"
HORIZONS = {"1h": 4, "4h": 16}            # forecast horizons in 15-minute steps
SENSOR_COLUMNS = ["engine_rpm", "engine_load_pct", "exhaust_temp_c", "exhaust_flow_kg_h", "ambient_temp_c"]
ROLLING_COLUMNS = ["exhaust_temp_c", "engine_load_pct", TARGET]
ROLLING_WINDOWS = {"15m": "15min", "1h": "1h", "4h": "4h"}
SEED = 42
STEP = pd.Timedelta(minutes=STEP_MINUTES)


# --------------------------------------------------------------------------------------
# Data + features (shared with optimizer.py and app.py)
# --------------------------------------------------------------------------------------
def load_sensor_data(path: str | Path = OUTPUT_CSV) -> pd.DataFrame:
    """Read the telemetry CSV on a regular 15-minute DatetimeIndex and make sure Q_in is present."""
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")].asfreq(f"{STEP_MINUTES}min")
    if TARGET not in df:
        df[TARGET] = waste_heat_kw(df["exhaust_flow_kg_h"], df["exhaust_temp_c"], df["ambient_temp_c"])
    return df


def _calendar(ts: pd.DatetimeIndex, prefix: str) -> dict[str, np.ndarray]:
    hour = np.asarray(ts.hour + ts.minute / 60.0, dtype=float)
    dow = np.asarray(ts.day_of_week, dtype=float)
    return {
        f"{prefix}_hour_sin": np.sin(2 * np.pi * hour / 24.0),
        f"{prefix}_hour_cos": np.cos(2 * np.pi * hour / 24.0),
        f"{prefix}_dow_sin": np.sin(2 * np.pi * dow / 7.0),
        f"{prefix}_dow_cos": np.cos(2 * np.pi * dow / 7.0),
        f"{prefix}_day_type": day_type(ts),  # weekday / Saturday / Sunday
    }


def build_features(df: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    """Feature matrix for forecasting Q_in(t + horizon) from what is known at time t."""
    horizon = horizon_steps * STEP

    def lagged(col: str, lag: pd.Timedelta) -> pd.Series:
        return df[col].shift(freq=lag).reindex(df.index)

    feats: dict[str, pd.Series | np.ndarray] = {col: df[col] for col in SENSOR_COLUMNS + [TARGET]}

    for col in ROLLING_COLUMNS:
        for label, window in ROLLING_WINDOWS.items():
            # closed="both" includes both ends of the window, so the 15-min window holds the two
            # readings that bound it (a right-closed window would hold one point and no std).
            rolling = df[col].rolling(window, closed="both")
            feats[f"{col}_mean_{label}"] = rolling.mean()
            feats[f"{col}_std_{label}"] = rolling.std()

    # Thermal gradients dT/dt (C per minute) and rate-of-change of load and heat output
    feats["exhaust_temp_grad_15m"] = (df["exhaust_temp_c"] - lagged("exhaust_temp_c", STEP)) / 15.0
    feats["exhaust_temp_grad_1h"] = (df["exhaust_temp_c"] - lagged("exhaust_temp_c", 4 * STEP)) / 60.0
    feats["load_grad_15m"] = (df["engine_load_pct"] - lagged("engine_load_pct", STEP)) / 15.0
    feats["q_in_grad_1h"] = (df[TARGET] - lagged(TARGET, 4 * STEP)) / 60.0

    # Lags: one hour back, and the same clock time yesterday aligned to the *target* time
    feats["q_in_lag_1h"] = lagged(TARGET, pd.Timedelta(hours=1))
    feats["q_in_lag_24h_at_target"] = lagged(TARGET, pd.Timedelta(days=1) - horizon)
    feats["load_lag_24h_at_target"] = lagged("engine_load_pct", pd.Timedelta(days=1) - horizon)

    # Cyclic calendar features: the production plan is known in advance for the target time
    feats.update(_calendar(df.index, "origin"))
    feats.update(_calendar(df.index + horizon, "target"))
    return pd.DataFrame(feats, index=df.index)


def make_target(df: pd.DataFrame, horizon_steps: int) -> pd.Series:
    """y(t) = Q_in(t + horizon)."""
    return df[TARGET].shift(freq=-horizon_steps * STEP).reindex(df.index)


# --------------------------------------------------------------------------------------
# Inference helpers (used by optimizer.py and app.py)
# --------------------------------------------------------------------------------------
def load_forecaster(path: str | Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_horizons(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Run every horizon model and align each forecast on its target time.

    Row tau of column ``q_pred_1h_kw`` is the forecast for tau that was issued at tau - 1 h,
    so it can be compared directly with ``df.q_in_kw`` at tau.
    """
    out = {}
    for name, steps in bundle["horizons_steps"].items():
        X = build_features(df, steps)[bundle["feature_columns"][name]]
        issued = pd.Series(bundle["models"][name].predict(X), index=df.index)
        out[f"q_pred_{name}_kw"] = issued.shift(freq=steps * STEP).reindex(df.index)
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_model(n_estimators: int):
    if lgb is not None:
        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=0.03, num_leaves=31, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED, n_jobs=2, verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=n_estimators, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, random_state=SEED,
    )


def tune_iterations(X: pd.DataFrame, y: pd.Series, horizon_steps: int) -> int:
    """Early stopping on the last 3 training days (purged by the horizon) to pick the tree count."""
    if lgb is None:
        return 400
    val_size = 3 * STEPS_PER_DAY
    fit_end = len(X) - val_size - horizon_steps
    X_val, y_val = X.iloc[-val_size:], y.iloc[-val_size:]
    # LightGBM >= 4.7 takes eval_X / eval_y; older releases take eval_set
    if "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters:
        eval_kwargs = {"eval_X": X_val, "eval_y": y_val}
    else:
        eval_kwargs = {"eval_set": [(X_val, y_val)]}
    model = make_model(3000)
    model.fit(
        X.iloc[:fit_end], y.iloc[:fit_end], eval_metric="rmse",
        callbacks=[lgb.early_stopping(150, verbose=False)], **eval_kwargs,
    )
    return max(50, int(model.best_iteration_ or model.n_estimators))


def walk_forward_cv(X: pd.DataFrame, y: pd.Series, horizon_steps: int, n_estimators: int) -> list[float]:
    """Expanding-window CV over the last 4 x 2 training days (every fold trains on >= 2 weeks).

    ``gap`` drops the rows whose targets would overlap the next validation fold.
    """
    scores = []
    splitter = TimeSeriesSplit(n_splits=4, test_size=2 * STEPS_PER_DAY, gap=horizon_steps)
    for train_idx, val_idx in splitter.split(X):
        model = make_model(n_estimators).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(float(root_mean_squared_error(y.iloc[val_idx], model.predict(X.iloc[val_idx]))))
    return scores


def train_horizon(df: pd.DataFrame, name: str, steps: int, test_start: pd.Timestamp) -> dict:
    X = build_features(df, steps)
    y = make_target(df, steps)
    target_time = pd.Series(df.index + steps * STEP, index=df.index)
    usable = y.notna() & X["q_in_lag_24h_at_target"].notna()
    X, y, target_time = X[usable], y[usable], target_time[usable]

    is_train = target_time < test_start          # purge: no training target inside the test period
    X_train, y_train = X[is_train], y[is_train]
    X_test, y_test = X[~is_train], y[~is_train]

    n_estimators = tune_iterations(X_train, y_train, steps)
    cv_rmse = walk_forward_cv(X_train, y_train, steps, n_estimators)
    model = make_model(int(n_estimators * 1.1)).fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "model": regression_metrics(y_test, pred),
        "persistence": regression_metrics(y_test, X_test[TARGET]),
        "seasonal_naive_24h": regression_metrics(y_test, X_test["q_in_lag_24h_at_target"]),
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_rmse_std": float(np.std(cv_rmse)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_estimators": int(model.n_estimators if lgb is not None else model.max_iter),
    }
    importance = {}
    if lgb is not None:
        gain = model.booster_.feature_importance(importance_type="gain")
        importance = dict(sorted(zip(X.columns, gain / gain.sum()), key=lambda kv: -kv[1]))
    return {"model": model, "features": list(X.columns), "metrics": metrics, "importance": importance}


def print_report(results: dict, test_start: pd.Timestamp, data_end: pd.Timestamp) -> None:
    library = f"LightGBM {lgb.__version__}" if lgb is not None else "scikit-learn HistGradientBoosting"
    print(f"\nModel: {library} | hold-out test targets {test_start} -> {data_end}\n")
    header = f"{'Horizon':<8}{'Forecaster':<22}{'RMSE kW':>10}{'MAE kW':>10}{'R2':>9}{'Skill vs persist.':>20}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        m = res["metrics"]
        persistence_rmse = m["persistence"]["rmse"]
        for label, key in (("LightGBM" if lgb else "HistGB", "model"),
                           ("Persistence", "persistence"),
                           ("Seasonal naive (24h)", "seasonal_naive_24h")):
            s = m[key]
            skill = 1.0 - s["rmse"] / persistence_rmse
            print(f"{name:<8}{label:<22}{s['rmse']:>10.2f}{s['mae']:>10.2f}{s['r2']:>9.3f}{skill:>19.1%}")
        print(f"{'':<8}walk-forward CV RMSE {m['cv_rmse_mean']:.2f} +/- {m['cv_rmse_std']:.2f} kW "
              f"({m['n_train']} train / {m['n_test']} test rows, {m['n_estimators']} trees)")
        if res["importance"]:
            top = ", ".join(f"{k} {v:.0%}" for k, v in list(res["importance"].items())[:5])
            print(f"{'':<8}top features: {top}")
        print("-" * len(header))


def train(data_path: str | Path = OUTPUT_CSV, out_path: str | Path = MODEL_PATH, test_days: int = 7) -> dict:
    """Train both horizon models, print the evaluation report and save the bundle."""
    df = load_sensor_data(data_path)
    data_end = df.index[-1]
    test_start = (data_end + STEP - pd.Timedelta(days=test_days)).normalize()
    print(f"Loaded {len(df):,} rows ({df.index[0]} -> {data_end}); {len(build_features(df, 4).columns)} features")

    results = {name: train_horizon(df, name, steps, test_start) for name, steps in HORIZONS.items()}
    print_report(results, test_start, data_end)

    bundle = {
        "models": {name: res["model"] for name, res in results.items()},
        "feature_columns": {name: res["features"] for name, res in results.items()},
        "metrics": {name: res["metrics"] for name, res in results.items()},
        "feature_importance": {name: res["importance"] for name, res in results.items()},
        "horizons_steps": dict(HORIZONS),
        "step_minutes": STEP_MINUTES,
        "target": TARGET,
        "cp_exhaust_kj_kgk": CP_EXHAUST_KJ_KGK,
        "data_start": str(df.index[0]),
        "test_start": str(test_start),
        "data_end": str(data_end),
        "library": f"lightgbm {lgb.__version__}" if lgb is not None else "sklearn HistGradientBoosting",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, out_path, compress=3)
    print(f"\nSaved forecaster bundle -> {Path(out_path).resolve()}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Q_in forecaster (1 h and 4 h ahead).")
    parser.add_argument("--data", default=OUTPUT_CSV)
    parser.add_argument("--out", default=MODEL_PATH)
    parser.add_argument("--test-days", type=int, default=7)
    args = parser.parse_args()
    train(args.data, args.out, args.test_days)


if __name__ == "__main__":
    main()
