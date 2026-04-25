"""Data loading stubs for trade value modeling project."""

from pathlib import Path

import pandas as pd


def load_advanced_stats(path: str | Path) -> pd.DataFrame:
    """Load advanced stats CSV.

    TODO:
    - Validate required columns.
    - Standardize column names if needed.
    """
    raise NotImplementedError("Implement advanced stats loading.")


def load_salary_data(path: str | Path) -> pd.DataFrame:
    """Load salary CSV.

    TODO:
    - Coerce salary to numeric.
    - Normalize player naming keys for merges.
    """
    raise NotImplementedError("Implement salary data loading.")
