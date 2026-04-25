"""Training stubs for trade value model."""

from pathlib import Path
from typing import Any

import pandas as pd


def train_model(df: pd.DataFrame, target_col: str = "trade_value_score_100") -> Any:
    """Train a baseline regression model.

    TODO:
    - Split data with season-aware strategy.
    - Train baseline regressor.
    - Return fitted model object.
    """
    raise NotImplementedError("Implement model training.")


def save_model(model: Any, output_path: str | Path) -> None:
    """Persist model artifact.

    TODO:
    - Serialize model safely (joblib or pickle).
    - Ensure parent output directory exists.
    """
    raise NotImplementedError("Implement model persistence.")
