"""Time-series cross-validation across multiple held-out seasons.

For each fold year F:
  Train: pairs where Season < F
  Test:  pairs where Season == F

Output:
  outputs/predictions/cv_results.csv  (one row per (fold, model))
"""
from __future__ import annotations

import sys
import warnings
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
from src.models.preprocess import compute_train_medians, prepare_features  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PAIRS = REPO_ROOT / "data" / "processed" / "training_pairs.csv"
OUT = REPO_ROOT / "outputs" / "predictions" / "cv_results.csv"

# Test seasons. Each fold trains on everything strictly earlier.
FOLD_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    return {
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2":   float(1 - np.sum(err ** 2) / np.sum((actual - actual.mean()) ** 2)),
    }


def _fold_predictions(X_train, yA_train, yB_train, X_test, next_ages) -> dict[str, np.ndarray]:
    """Train all 8 models for one fold, return predicted OVR per model on the test set."""
    xgb_A = train_xgb(X_train, yA_train)
    mlp_A = train_mlp(X_train, yA_train)
    ae_A  = train_autoencoder_knn(X_train, yA_train, k=10, epochs=200)
    ens_A = train_ensemble(xgb_A, mlp_A, names=["optA_xgboost", "optA_mlp"])

    xgb_B = train_xgb(X_train, yB_train)
    mlp_B = train_mlp(X_train, yB_train)
    ae_B  = train_autoencoder_knn(X_train, yB_train, k=10, epochs=200)
    ens_B = train_ensemble(xgb_B, mlp_B, names=["optB_xgboost", "optB_mlp"])

    preds: dict[str, np.ndarray] = {}
    for name, m in [
        ("optA_xgboost", xgb_A), ("optA_mlp", mlp_A),
        ("optA_autoencoder", ae_A), ("optA_ensemble", ens_A),
    ]:
        preds[name] = np.asarray(m.predict(X_test))

    for name, m in [
        ("optB_xgboost", xgb_B), ("optB_mlp", mlp_B),
        ("optB_autoencoder", ae_B), ("optB_ensemble", ens_B),
    ]:
        pred_stats = np.asarray(m.predict(X_test))
        preds[name] = stats_to_ovr(pred_stats, next_ages)

    return preds


def main() -> None:
    pairs = pd.read_csv(PAIRS)
    pairs = pairs.dropna(subset=[TARGET_OVR_COL] + TARGET_OPTION_B_COLS).reset_index(drop=True)

    rows: list[dict] = []
    for fold_year in FOLD_YEARS:
        train_df = pairs[pairs["Season"] < fold_year].reset_index(drop=True)
        test_df  = pairs[pairs["Season"] == fold_year].reset_index(drop=True)
        if test_df.empty or len(train_df) < 100:
            print(f"  skipping fold {fold_year}: insufficient data")
            continue

        target_label = f"{int(test_df['next_Season'].iloc[0]) - 1}-{str(int(test_df['next_Season'].iloc[0]))[2:]}"
        print(f"\nFold: test season={fold_year} (predicting {target_label}) | train={len(train_df):,} test={len(test_df):,}")

        medians = compute_train_medians(train_df)
        X_train = prepare_features(train_df, medians=medians)
        X_test = prepare_features(test_df, feature_template=X_train, medians=medians)
        yA_train = train_df[TARGET_OVR_COL].values
        yA_test  = test_df[TARGET_OVR_COL].values
        yB_train = train_df[TARGET_OPTION_B_COLS].values
        next_ages = (test_df["Age"].values + 1).astype(float)

        preds = _fold_predictions(X_train, yA_train, yB_train, X_test, next_ages)

        for model_name, pred in preds.items():
            m = _metrics(yA_test, pred)
            rows.append({
                "fold_year": fold_year,
                "model": model_name,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
            })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote per-fold results to {OUT.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 80)
    print(f"CV SUMMARY ({len(FOLD_YEARS)} folds, mean ± std)")
    print("=" * 80)

    agg = df.groupby("model").agg(
        mae_mean=("mae", "mean"),  mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),     r2_std=("r2", "std"),
    ).reset_index().sort_values("mae_mean")

    print(f"  {'model':<22} {'MAE':>15} {'RMSE':>15} {'R^2':>15}")
    print(f"  {'-' * 22} {'-' * 15} {'-' * 15} {'-' * 15}")
    for _, r in agg.iterrows():
        mae_s  = f"{r['mae_mean']:>5.2f} ± {r['mae_std']:<5.2f}"
        rmse_s = f"{r['rmse_mean']:>5.2f} ± {r['rmse_std']:<5.2f}"
        r2_s   = f"{r['r2_mean']:>+5.3f} ± {r['r2_std']:<5.3f}"
        print(f"  {r['model']:<22} {mae_s:>15} {rmse_s:>15} {r2_s:>15}")

    best = agg.iloc[0]["model"]
    print(f"\nPer-fold MAE for best model ({best}):")
    print(f"  {'fold (test season)':<22} {'MAE':>8} {'RMSE':>8} {'R^2':>8}")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 8}")
    for _, r in df[df["model"] == best].sort_values("fold_year").iterrows():
        print(f"  test={int(r['fold_year']):<17} {r['mae']:>8.2f} {r['rmse']:>8.2f} {r['r2']:>8.3f}")


if __name__ == "__main__":
    main()
