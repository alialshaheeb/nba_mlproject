"""Cleaning functions for the trade-value modeling project.

The canonical cleaned outputs live in ``data/processed/``. Notebooks and
scripts should import from this module rather than reimplementing cleaning
logic locally.
"""
from __future__ import annotations

import pandas as pd

NUMERIC_COLS = [
    "Age", "G", "GS", "MP",
    "PER", "TS%", "3PAr", "FTr",
    "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%",
    "OWS", "DWS", "WS", "WS/48",
    "OBPM", "DBPM", "BPM", "VORP",
    "Season",
]

TEXT_COLS = ["Player", "Team", "Pos", "Awards", "player_id", "Season_Label"]


def clean_advanced_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Filter junk rows, normalize text, coerce numerics on the advanced-stats table."""
    out = df.copy()

    out = out.dropna(subset=["Player"])
    out = out[~out["Player"].astype("string").str.strip().isin(["", "Player", "League Average"])]

    for col in TEXT_COLS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

    if "Awards" in out.columns:
        out["Awards"] = out["Awards"].fillna("")

    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.drop(columns=["Rk"], errors="ignore")
    out = out.reset_index(drop=True)
    return out


def clean_salary_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the BR salary table (placeholder — current salary file already cleaned)."""
    raise NotImplementedError("Salary cleaning will be rewritten when historical salaries are scraped.")


def merge_stats_and_salary(
    stats_df: pd.DataFrame,
    salary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cleaned stats and salary on player_id (placeholder)."""
    raise NotImplementedError("Stats-salary merge will be implemented in the modeling stage.")
