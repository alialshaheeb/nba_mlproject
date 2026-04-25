"""Apply the OVR formula to predicted stats (used by Option B models)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pairs import TARGET_STATS

OVR_WEIGHTS = {
    "production_score":   0.40,
    "efficiency_score":   0.20,
    "role_score":         0.15,
    "availability_score": 0.15,
    "age_score":          0.10,
}


def age_curve(age: float) -> float:
    if pd.isna(age):
        return 50.0
    if age <= 27:
        return max(50.0, 100.0 - 3.5 * (27 - age))
    return max(0.0, 100.0 - 5.0 * (age - 27))


def stats_to_ovr(pred_stats: np.ndarray, next_ages: np.ndarray) -> np.ndarray:
    """Convert (n, 8) predicted stats + next-year ages (n,) into predicted OVR (n,).

    Column order in ``pred_stats`` must match ``TARGET_STATS``:
    [VORP, BPM, WS, PER, TS%, USG%, MP, G].

    Percentiles are computed across the predicted batch — i.e. the predicted
    set acts as its own synthetic season for ranking.
    """
    df = pd.DataFrame(pred_stats, columns=TARGET_STATS)

    for col in TARGET_STATS:
        df[f"{col}_pct"] = df[col].rank(pct=True) * 100

    df["production_score"] = df[["VORP_pct", "BPM_pct", "WS_pct"]].mean(axis=1)
    df["efficiency_score"] = df[["PER_pct", "TS%_pct"]].mean(axis=1)
    df["role_score"] = df[["USG%_pct", "MP_pct"]].mean(axis=1)
    df["availability_score"] = (df["G"] / 82.0 * 100).clip(0, 100)
    df["age_score"] = pd.Series(next_ages).apply(age_curve).values

    ovr = sum(df[col] * w for col, w in OVR_WEIGHTS.items())
    return ovr.values
