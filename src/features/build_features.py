"""Feature engineering for the trade-value model.

Stage 1 of the score: a deterministic 0-100 production OVR computed per
player-season. Every input stat is converted to a within-season percentile
rank before combining, so eras and stat scales no longer distort the result.
"""
from __future__ import annotations

import pandas as pd

PERCENTILE_COLS = ["VORP", "BPM", "WS", "PER", "TS%", "USG%", "MP"]

SUBSCORE_RECIPES: dict[str, list[str]] = {
    "production_score":  ["VORP_pct", "BPM_pct", "WS_pct"],
    "efficiency_score":  ["PER_pct", "TS%_pct"],
    "role_score":        ["USG%_pct", "MP_pct"],
}

OVR_WEIGHTS: dict[str, float] = {
    "production_score":   0.40,
    "efficiency_score":   0.20,
    "role_score":         0.15,
    "availability_score": 0.15,
    "age_score":          0.10,
}

DEFAULT_MIN_MP = 500
DEFAULT_MIN_G = 20


def filter_qualified(df: pd.DataFrame, min_mp: int = DEFAULT_MIN_MP, min_games: int = DEFAULT_MIN_G) -> pd.DataFrame:
    """Keep only player-seasons with enough playing time for stable advanced stats."""
    return df[(df["MP"] >= min_mp) & (df["G"] >= min_games)].copy()


def add_season_percentiles(df: pd.DataFrame, cols: list[str] = PERCENTILE_COLS) -> pd.DataFrame:
    """Add a 0-100 percentile rank for each stat, computed within each season."""
    out = df.copy()
    for col in cols:
        out[f"{col}_pct"] = out.groupby("Season")[col].rank(pct=True) * 100
    return out


def add_availability_score(df: pd.DataFrame) -> pd.DataFrame:
    """Games played as a percent of the 82-game NBA season."""
    out = df.copy()
    out["availability_score"] = (out["G"] / 82.0 * 100).clip(0, 100)
    return out


def _age_curve(age: float) -> float:
    if pd.isna(age):
        return 50.0
    if age <= 27:
        return max(50.0, 100.0 - 3.5 * (27 - age))
    return max(0.0, 100.0 - 5.0 * (age - 27))


def add_age_score(df: pd.DataFrame) -> pd.DataFrame:
    """Smooth age curve peaking at 27, with a steeper decline after peak."""
    out = df.copy()
    out["age_score"] = out["Age"].apply(_age_curve)
    return out


def add_sub_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Average each sub-score's percentile components into a single 0-100 column."""
    out = df.copy()
    for sub, cols in SUBSCORE_RECIPES.items():
        out[sub] = out[cols].mean(axis=1)
    return out


def add_ovr(df: pd.DataFrame) -> pd.DataFrame:
    """Final 0-100 OVR as a weighted combination of sub-scores."""
    out = df.copy()
    out["ovr"] = sum(out[col] * w for col, w in OVR_WEIGHTS.items())
    out["ovr"] = out["ovr"].round(1)
    return out


def build_player_scores(df: pd.DataFrame, min_mp: int = DEFAULT_MIN_MP, min_games: int = DEFAULT_MIN_G) -> pd.DataFrame:
    """Run the full Stage 1 feature pipeline and return a scored DataFrame."""
    out = filter_qualified(df, min_mp=min_mp, min_games=min_games)
    out = add_season_percentiles(out)
    out = add_availability_score(out)
    out = add_age_score(out)
    out = add_sub_scores(out)
    out = add_ovr(out)
    return out
