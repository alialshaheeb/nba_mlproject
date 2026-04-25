"""Train all 4 algorithms x 2 options (A, B) and save in human-readable formats.

Time-based split:
  Train: pairs where Season <= 2022 (predicting up to 2023-24 outcomes)
  Test:  pairs where Season == 2023 (predicting 2024-25 — actuals available)

Each fitted model writes to ``outputs/models/<name>/``.
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

TRAIN_MAX_SEASON = 2022
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


def main() -> None:
    pairs = pd.read_csv(PAIRS)
    pairs = pairs.dropna(subset=[TARGET_OVR_COL] + TARGET_OPTION_B_COLS).reset_index(drop=True)

    train_df = pairs[pairs["Season"] <= TRAIN_MAX_SEASON].reset_index(drop=True)
    test_df  = pairs[pairs["Season"] == TEST_SEASON].reset_index(drop=True)
    print(f"Train pairs: {len(train_df):,} | Test pairs: {len(test_df):,}")

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df, feature_template=X_train)

    yA_train = train_df[TARGET_OVR_COL].values
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

    next_ages = (test_df["Age"].values + 1).astype(float)

    out = pd.DataFrame({
        "Player":      test_df["Player"].values,
        "Season":      test_df["Season"].values,
        "next_Season": test_df["next_Season"].values,
        "Age":         test_df["Age"].values,
        "actual_ovr":  yA_test,
    })

    print("\n--- Test metrics (vs actual next-season OVR) ---")
    print(f"{'model':<25} {'MAE':>8} {'RMSE':>8} {'R^2':>8}")
    for name, m in optA_models.items():
        pred = m.predict(X_test)
        out[name] = pred
        mt = _metrics(yA_test, pred)
        print(f"  {name:<23} {mt['mae']:>8.3f} {mt['rmse']:>8.3f} {mt['r2']:>8.3f}")

    for name, m in optB_models.items():
        pred_stats = m.predict(X_test)
        pred_ovr = stats_to_ovr(pred_stats, next_ages)
        out[name] = pred_ovr
        mt = _metrics(yA_test, pred_ovr)
        print(f"  {name:<23} {mt['mae']:>8.3f} {mt['rmse']:>8.3f} {mt['r2']:>8.3f}")

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDS_DIR / "test_predictions.csv"

    rename_for_notebook = {f"opt{ab}_{algo}": f"opt{ab}_{algo}" for ab in ("A", "B") for algo in ("xgboost", "mlp", "autoencoder", "ensemble")}
    out = out.rename(columns=rename_for_notebook)
    out.to_csv(out_path, index=False)
    print(f"\nWrote test predictions to {out_path.relative_to(REPO_ROOT)}")
    print(f"Saved feature columns to {feature_cols_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
