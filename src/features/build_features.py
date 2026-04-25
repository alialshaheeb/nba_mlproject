"""Feature engineering stubs for trade value model."""

import pandas as pd


def add_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create availability and durability proxy features.

    TODO:
    - Derive games missed proxy from games played.
    - Add availability rate and rolling durability trend.
    """
    raise NotImplementedError("Implement availability features.")


def add_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create salary-context features.

    TODO:
    - Add salary percentile by season.
    - Add production-per-dollar style metrics.
    """
    raise NotImplementedError("Implement contract features.")


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble final model feature table.

    TODO:
    - Select stable feature set.
    - Handle NaNs and final typing.
    """
    raise NotImplementedError("Implement model feature table.")
