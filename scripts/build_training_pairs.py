"""Build (season N -> season N+1) training pairs and save to data/processed/."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.pairs import build_pairs  # noqa: E402

STATS = REPO_ROOT / "data" / "processed" / "advanced_stats_clean.csv"
SCORES = REPO_ROOT / "data" / "processed" / "player_scores.csv"
OUT = REPO_ROOT / "data" / "processed" / "training_pairs.csv"


def main() -> None:
    stats = pd.read_csv(STATS)
    scores = pd.read_csv(SCORES)
    pairs = build_pairs(stats, scores)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT, index=False)

    print(f"Wrote {len(pairs):,} pairs to {OUT.relative_to(REPO_ROOT)}")
    print(f"Season span: {int(pairs['Season'].min())} -> {int(pairs['Season'].max())}")
    print(f"Unique players: {pairs['Player'].nunique():,}")
    print()
    print("Pairs per starting season:")
    print(pairs["Season"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
