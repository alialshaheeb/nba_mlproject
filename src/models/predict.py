"""Prediction stubs for trade value model."""

from typing import Any

import pandas as pd


def load_model(model_path: str) -> Any:
    """Load trained model from disk.

    TODO:
    - Implement artifact deserialization.
    """
    raise NotImplementedError("Implement model loading.")


def predict_trade_value(model: Any, df: pd.DataFrame) -> pd.Series:
    """Generate trade value predictions.

    TODO:
    - Validate model input schema.
    - Return clipped 0-100 predictions.
    """
    raise NotImplementedError("Implement trade value prediction.")
