"""Drive Stage 1 feature engineering: load clean stats -> compute OVR -> save scores."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.build_features import build_player_scores  # noqa: E402

CLEAN = REPO_ROOT / "data" / "processed" / "advanced_stats_clean.csv"
OUT = REPO_ROOT / "data" / "processed" / "player_scores.csv"

KEEP_COLS = [
    "Player", "Season", "Season_Label", "Team", "Pos", "Age",
    "G", "GS", "MP",
    "production_score", "efficiency_score", "role_score",
    "availability_score", "age_score",
    "ovr",
    "player_id",
]


def main() -> None:
    clean = pd.read_csv(CLEAN)
    scores = build_player_scores(clean)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in KEEP_COLS if c in scores.columns]
    scores[keep].to_csv(OUT, index=False)

    print(f"Read  {len(clean):,} clean rows from {CLEAN.relative_to(REPO_ROOT)}")
    print(f"Wrote {len(scores):,} scored rows to {OUT.relative_to(REPO_ROOT)}")

    s2025 = scores[scores["Season"] == 2025].copy()
    s2025 = s2025.sort_values("MP", ascending=False).drop_duplicates(subset=["Player"], keep="first")
    top = s2025.nlargest(10, "ovr")[
        ["Player", "Team", "Age", "ovr", "production_score", "efficiency_score",
         "role_score", "availability_score", "age_score"]
    ]
    print("\nTop 10 OVR — 2024-25:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
