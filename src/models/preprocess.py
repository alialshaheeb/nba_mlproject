"""Feature preparation for OVR forecasting models."""
from __future__ import annotations

import pandas as pd

from .pairs import FEATURE_NUMERIC


def prepare_features(df: pd.DataFrame, feature_template: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a numeric feature matrix from a pairs DataFrame.

    Numeric stats are median-imputed. ``Pos`` is one-hot encoded.
    Pass ``feature_template`` (a previously-prepared train X) to lock the
    output column order — required so test/val matrices align with train.
    """
    X = df[FEATURE_NUMERIC].copy()
    X = X.fillna(X.median(numeric_only=True))

    pos = df["Pos"].fillna("UNK").astype(str)
    pos_dummies = pd.get_dummies(pos, prefix="Pos", dtype=float)
    X = pd.concat([X.reset_index(drop=True), pos_dummies.reset_index(drop=True)], axis=1)

    if feature_template is not None:
        for col in feature_template.columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_template.columns]

    return X
