"""Data cleaning stubs for trade value modeling project."""

import pandas as pd


def clean_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Clean advanced stats table.

    TODO:
    - Handle missing values.
    - Coerce numeric metrics.
    - Remove duplicates by player-season.
    """
    raise NotImplementedError("Implement advanced stats cleaning.")


def clean_salary_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean salary table.

    TODO:
    - Standardize season format.
    - Remove non-player rows and invalid salaries.
    """
    raise NotImplementedError("Implement salary data cleaning.")


def merge_stats_and_salary(
    stats_df: pd.DataFrame,
    salary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cleaned stats and salary on player-season keys.

    TODO:
    - Define join keys.
    - Resolve duplicate matches.
    - Track unmatched rows.
    """
    raise NotImplementedError("Implement stats-salary merge.")
