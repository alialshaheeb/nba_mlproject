"""Train models with a proper three-way time-based split.

  Train:      pairs where Season <= 2021 (12 seasons)
  Validation: pairs where Season == 2022 (held out for hyperparameter tuning)
  Test:       pairs where Season == 2023 (held out, only touched once)

Each fitted model writes to ``outputs/models/<name>/``.
Predictions for both val and test are written to ``outputs/predictions/``.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.formula import stats_to_ovr  # noqa: E402
from src.models.models import (  # noqa: E402
    train_autoencoder_knn,
    train_ensemble,
    train_mlp,
    train_xgb,
)
from src.models.pairs import (  # noqa: E402
    TARGET_OPTION_B_COLS,
    TARGET_OVR_COL,
)
from src.models.preprocess import prepare_features  # noqa: E402

PAIRS = REPO_ROOT / "data" / "processed" / "training_pairs.csv"
MODELS_DIR = REPO_ROOT / "outputs" / "models"
PREDS_DIR = REPO_ROOT / "outputs" / "predictions"

TRAIN_MAX_SEASON = 2021
VAL_SEASON = 2022
TEST_SEASON = 2023


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    return {
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2":   float(1 - np.sum(err**2) / np.sum((actual - actual.mean())**2)),
    }


def _wipe_model_dir(name: str) -> Path:
    p = MODELS_DIR / name
    if p.exists():
        shutil.rmtree(p)
    return p


def _evaluate(name: str, optA_models: dict, optB_models: dict, X: pd.DataFrame, y_true: np.ndarray, next_ages: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Predict, score, and return a DataFrame of predictions for one held-out split."""
    out = pd.DataFrame({
        "Player":      df["Player"].values,
        "Season":      df["Season"].values,
        "next_Season": df["next_Season"].values,
        "Age":         df["Age"].values,
        "actual_ovr":  y_true,
    })

    print(f"\n--- {name} metrics (vs actual next-season OVR) ---")
    print(f"  {'model':<25} {'MAE':>8} {'RMSE':>8} {'R^2':>8}")

    for model_name, model in optA_models.items():
        pred = model.predict(X)
        out[model_name] = pred
        mt = _metrics(y_true, pred)
        print(f"  {model_name:<23} {mt['mae']:>8.3f} {mt['rmse']:>8.3f} {mt['r2']:>8.3f}")

    for model_name, model in optB_models.items():
        pred_stats = model.predict(X)
        pred_ovr = stats_to_ovr(pred_stats, next_ages)
        out[model_name] = pred_ovr
        mt = _metrics(y_true, pred_ovr)
        print(f"  {model_name:<23} {mt['mae']:>8.3f} {mt['rmse']:>8.3f} {mt['r2']:>8.3f}")

    return out


def main() -> None:
    pairs = pd.read_csv(PAIRS)
    pairs = pairs.dropna(subset=[TARGET_OVR_COL] + TARGET_OPTION_B_COLS).reset_index(drop=True)

    train_df = pairs[pairs["Season"] <= TRAIN_MAX_SEASON].reset_index(drop=True)
    val_df   = pairs[pairs["Season"] == VAL_SEASON].reset_index(drop=True)
    test_df  = pairs[pairs["Season"] == TEST_SEASON].reset_index(drop=True)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    X_train = prepare_features(train_df)
    X_val   = prepare_features(val_df,  feature_template=X_train)
    X_test  = prepare_features(test_df, feature_template=X_train)

    yA_train = train_df[TARGET_OVR_COL].values
    yA_val   = val_df[TARGET_OVR_COL].values
    yA_test  = test_df[TARGET_OVR_COL].values
    yB_train = train_df[TARGET_OPTION_B_COLS].values

    print("\n--- Training Option A (predict OVR directly) ---")
    xgb_A = train_xgb(X_train, yA_train); print("  xgboost done")
    mlp_A = train_mlp(X_train, yA_train); print("  mlp done")
    ae_A  = train_autoencoder_knn(X_train, yA_train, k=10, epochs=200); print("  autoencoder+knn done")
    ens_A = train_ensemble(xgb_A, mlp_A, names=["optA_xgboost", "optA_mlp"])
    print("  ensemble done")

    print("\n--- Training Option B (predict 8 stats, apply formula) ---")
    xgb_B = train_xgb(X_train, yB_train); print("  xgboost done")
    mlp_B = train_mlp(X_train, yB_train); print("  mlp done")
    ae_B  = train_autoencoder_knn(X_train, yB_train, k=10, epochs=200); print("  autoencoder+knn done")
    ens_B = train_ensemble(xgb_B, mlp_B, names=["optB_xgboost", "optB_mlp"])
    print("  ensemble done")

    optA_models = {"optA_xgboost": xgb_A, "optA_mlp": mlp_A, "optA_autoencoder": ae_A, "optA_ensemble": ens_A}
    optB_models = {"optB_xgboost": xgb_B, "optB_mlp": mlp_B, "optB_autoencoder": ae_B, "optB_ensemble": ens_B}

    print("\n--- Saving models (native readable formats) ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, m in {**optA_models, **optB_models}.items():
        path = _wipe_model_dir(name)
        m.save(path)
        print(f"  outputs/models/{name}/")
    feature_cols_path = MODELS_DIR / "feature_columns.json"
    feature_cols_path.write_text(json.dumps(X_train.columns.tolist(), indent=2))

    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    next_ages_val  = (val_df["Age"].values + 1).astype(float)
    val_preds = _evaluate("VALIDATION", optA_models, optB_models, X_val, yA_val, next_ages_val, val_df)
    val_path = PREDS_DIR / "val_predictions.csv"
    val_preds.to_csv(val_path, index=False)

    next_ages_test = (test_df["Age"].values + 1).astype(float)
    test_preds = _evaluate("TEST (held out)", optA_models, optB_models, X_test, yA_test, next_ages_test, test_df)
    test_path = PREDS_DIR / "test_predictions.csv"
    test_preds.to_csv(test_path, index=False)

    print(f"\nWrote {val_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {test_path.relative_to(REPO_ROOT)}")
    print(f"Saved feature columns to {feature_cols_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
