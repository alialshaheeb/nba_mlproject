"""Reconstruct a clean raw advanced-stats file from the corrupted concat output.

Background
----------
``scripts/data_combine.py`` read the local 2024-25 file with ``skiprows=1`` and
silently used Mikal Bridges' first data row as the column header. After the
``pd.concat`` with the cleanly-scraped 2010-2024 DataFrames, the merged file
ended up with two parallel schemas:

* 2010-2024 rows live in the proper columns (``Rk, Player, Age, ...``).
* 2024-25 rows live in the garbage columns (``1, Mikal Bridges, 28, NYK, ...``).
* Mikal Bridges' own 2024-25 row was lost — its values became the garbage header.

This script realigns every row onto a single canonical schema and synthesizes
the lost Mikal Bridges row from the garbage column names themselves.
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "raw" / "advanced_stats_all.csv"
DST = REPO_ROOT / "data" / "raw" / "advanced_stats_2010_2025.csv"

# Garbage column name -> proper column name (for Season=2025 rows).
# The keys are literally Mikal Bridges' 2024-25 stat values, captured as headers.
GARBAGE_TO_CLEAN = {
    "1": "Rk",
    "Mikal Bridges": "Player",
    "28": "Age",
    "NYK": "Team",
    "SF": "Pos",
    "82": "G",
    "82.1": "GS",        # pandas dup-renamed (original value was 82)
    "3036": "MP",
    "14.0": "PER",
    ".585": "TS%",
    ".391": "3PAr",
    ".100": "FTr",
    "2.7": "ORB%",
    "7.0": "DRB%",
    "4.9": "TRB%",
    "14.4": "AST%",
    "1.2": "STL%",
    "1.3": "BLK%",
    "9.7": "TOV%",
    "19.6": "USG%",
    "3.7": "OWS",
    "2.0": "DWS",
    "5.7": "WS",
    ".090": "WS/48",
    "0.4": "OBPM",
    "-0.9": "DBPM",
    "-0.5": "BPM",
    "1.2.1": "VORP",     # pandas dup-renamed (original value was 1.2)
    "Unnamed: 28": "Awards",
    "bridgmi01": "player_id",
}

# Bridges' lost row, recovered from the garbage column names.
MIKAL_BRIDGES_2025 = {
    "Rk": "1", "Player": "Mikal Bridges", "Age": "28", "Team": "NYK", "Pos": "SF",
    "G": "82", "GS": "82", "MP": "3036",
    "PER": "14.0", "TS%": "0.585", "3PAr": "0.391", "FTr": "0.100",
    "ORB%": "2.7", "DRB%": "7.0", "TRB%": "4.9", "AST%": "14.4",
    "STL%": "1.2", "BLK%": "1.3", "TOV%": "9.7", "USG%": "19.6",
    "OWS": "3.7", "DWS": "2.0", "WS": "5.7", "WS/48": "0.090",
    "OBPM": "0.4", "DBPM": "-0.9", "BPM": "-0.5", "VORP": "1.2",
    "Awards": "", "player_id": "bridgmi01",
    "Season": "2025", "Season_Label": "2024-25",
}

OUTPUT_COLS = [
    "Rk", "Player", "Age", "Team", "Pos", "G", "GS", "MP",
    "PER", "TS%", "3PAr", "FTr",
    "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%",
    "OWS", "DWS", "WS", "WS/48",
    "OBPM", "DBPM", "BPM", "VORP",
    "Awards", "player_id", "Season", "Season_Label",
]


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)

    with open(SRC, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        clean_idx = {col: i for i, col in enumerate(header) if col in OUTPUT_COLS}
        garbage_idx = {
            GARBAGE_TO_CLEAN[col]: i
            for i, col in enumerate(header)
            if col in GARBAGE_TO_CLEAN
        }

        season_pos = header.index("Season")
        season_label_pos = header.index("Season_Label")

        out_rows: list[dict[str, str]] = []
        for row in reader:
            season = row[season_pos]
            if season == "2025":
                rec = {col: row[garbage_idx[col]] if col in garbage_idx else "" for col in OUTPUT_COLS}
                rec["Season"] = row[season_pos]
                rec["Season_Label"] = row[season_label_pos]
            else:
                rec = {col: row[clean_idx[col]] if col in clean_idx else "" for col in OUTPUT_COLS}

            if rec["Player"].strip() in ("", "Player", "League Average"):
                continue
            out_rows.append(rec)

    out_rows.append(MIKAL_BRIDGES_2025)

    with open(DST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        writer.writerows(out_rows)

    by_season: dict[str, int] = {}
    for r in out_rows:
        by_season[r["Season"]] = by_season.get(r["Season"], 0) + 1
    print(f"Wrote {DST.relative_to(REPO_ROOT)}")
    print(f"Total rows: {len(out_rows)}")
    for s in sorted(by_season):
        print(f"  Season {s}: {by_season[s]} rows")

    # Sanity: top 5 by VORP for 2024-25 should be actual stars
    rows_2025 = [r for r in out_rows if r["Season"] == "2025"]
    def _vorp(r: dict[str, str]) -> float:
        try:
            return float(r["VORP"])
        except (ValueError, TypeError):
            return float("-inf")
    top5 = sorted(rows_2025, key=_vorp, reverse=True)[:5]
    print("\nSanity check — top 5 VORP in 2024-25:")
    for r in top5:
        print(f"  {r['Player']:<28} Team={r['Team']:<4} VORP={r['VORP']:<5} BPM={r['BPM']}")


if __name__ == "__main__":
    main()
