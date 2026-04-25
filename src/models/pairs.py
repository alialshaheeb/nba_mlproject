"""Build (season N -> season N+1) training pairs for OVR forecasting."""
from __future__ import annotations

import pandas as pd

FEATURE_NUMERIC = [
    "Age", "G", "GS", "MP",
    "PER", "TS%", "3PAr", "FTr",
    "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%",
    "OWS", "DWS", "WS", "WS/48",
    "OBPM", "DBPM", "BPM", "VORP",
]
FEATURE_CATEGORICAL = ["Pos"]

TARGET_STATS = ["VORP", "BPM", "WS", "PER", "TS%", "USG%", "MP", "G"]

TARGET_OVR_COL = "next_ovr"
TARGET_OPTION_B_COLS = [f"next_{s}" for s in TARGET_STATS]


def build_pairs(stats: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Pair each (player, season N) row with that player's (season N+1) row.

    Returns one row per consecutive-season pair, with feature columns from
    season N and next target columns from season N+1.
    """
    merged = stats.merge(scores[["Player", "Season", "ovr"]], on=["Player", "Season"], how="inner")
    merged = merged.sort_values("MP", ascending=False).drop_duplicates(["Player", "Season"], keep="first")
    merged = merged.sort_values(["Player", "Season"]).reset_index(drop=True)

    grp = merged.groupby("Player")
    merged["next_Season"] = grp["Season"].shift(-1)
    merged["next_ovr"] = grp["ovr"].shift(-1)
    for stat in TARGET_STATS:
        merged[f"next_{stat}"] = grp[stat].shift(-1)

    pairs = merged[merged["next_Season"] == merged["Season"] + 1].copy()
    return pairs.reset_index(drop=True)
