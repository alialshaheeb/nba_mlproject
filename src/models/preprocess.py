"""Feature preparation for OVR forecasting models."""
from __future__ import annotations

import pandas as pd

from .pairs import FEATURE_NUMERIC


def compute_train_medians(df: pd.DataFrame) -> pd.Series:
    """Compute imputation medians from training data only — pass these to prepare_features() for val/test."""
    return df[FEATURE_NUMERIC].median(numeric_only=True)


def prepare_features(
    df: pd.DataFrame,
    feature_template: pd.DataFrame | None = None,
    medians: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a numeric feature matrix from a pairs DataFrame.

    Numeric stats are median-imputed. ``Pos`` is one-hot encoded.
    Pass ``feature_template`` to lock the output column order — required so test/val matrices align with train.
    Pass ``medians`` (computed via ``compute_train_medians`` on training data) when preparing val/test
    to avoid leakage. If omitted, medians are computed from ``df`` itself (only safe for training)."""

    X = df[FEATURE_NUMERIC].copy()
    if medians is None:
        medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    pos = df["Pos"].fillna("UNK").astype(str)
    pos_dummies = pd.get_dummies(pos, prefix="Pos", dtype=float)
    X = pd.concat([X.reset_index(drop=True), pos_dummies.reset_index(drop=True)], axis=1)

    if feature_template is not None:
        for col in feature_template.columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_template.columns]

    return X
