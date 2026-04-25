"""Trade value score stubs (0-100 scale)."""

import pandas as pd


def compute_performance_score(df: pd.DataFrame) -> pd.Series:
    """Compute performance component score in [0, 100].

    TODO:
    - Combine impact, production, and efficiency metrics.
    - Normalize by season.
    """
    raise NotImplementedError("Implement performance score.")


def compute_contract_score(df: pd.DataFrame) -> pd.Series:
    """Compute contract-value component score in [0, 100].

    TODO:
    - Penalize high salary with low production.
    - Reward surplus value profiles.
    """
    raise NotImplementedError("Implement contract score.")


def compute_trade_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with final trade value score and components.

    TODO:
    - Combine component scores into final 0-100 output.
    - Add business-rule penalties and clipping.
    """
    raise NotImplementedError("Implement final trade value score.")
