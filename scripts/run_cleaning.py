"""Drive advanced-stats cleaning: load raw → clean → save processed."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.clean_data import clean_advanced_stats  # noqa: E402

RAW = REPO_ROOT / "data" / "raw" / "advanced_stats_2010_2025.csv"
PROCESSED = REPO_ROOT / "data" / "processed" / "advanced_stats_clean.csv"


def main() -> None:
    raw = pd.read_csv(RAW)
    clean = clean_advanced_stats(raw)

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED, index=False)

    print(f"Read {len(raw):,} raw rows from {RAW.relative_to(REPO_ROOT)}")
    print(f"Wrote {len(clean):,} clean rows to {PROCESSED.relative_to(REPO_ROOT)}")
    print(f"Columns ({len(clean.columns)}): {list(clean.columns)}")
    print(f"Seasons: {sorted(clean['Season'].dropna().astype(int).unique().tolist())}")
    print()
    print("Top 5 VORP in 2024-25 (sanity check):")
    last = clean[clean["Season"] == 2025].nlargest(5, "VORP")
    print(last[["Player", "Team", "Age", "VORP", "BPM", "WS"]].to_string(index=False))


if __name__ == "__main__":
    main()
