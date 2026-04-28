"""CLI: predict a player's OVR for a target season.

Usage:
  python3 scripts/predict_ovr.py "<player>" <year>

Where year is the year-end of the target NBA season:
  2025  -> 2024-25 season  (lookup, latest in data)
  2026  -> 2025-26 season  (predict 1 year ahead)
  2027  -> 2026-27 season  (predict 2 years ahead)
  2024  -> 2023-24 season  (lookup, historical)

  python3 scripts/predict_ovr.py "Stephen Curry" 2024
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.formula import OVR_WEIGHTS, age_curve  # noqa: E402
from src.models.models import load_model  # noqa: E402
from src.models.pairs import TARGET_STATS  # noqa: E402
from src.models.preprocess import prepare_features  # noqa: E402
from src.models.trade_value import format_dollars, ovr_to_dollars, ovr_to_tier  # noqa: E402

CLEAN = REPO_ROOT / "data" / "processed" / "advanced_stats_clean.csv"
SCORES = REPO_ROOT / "data" / "processed" / "player_scores.csv"
MODELS_DIR = REPO_ROOT / "outputs" / "models"
LATEST_KNOWN_SEASON = 2025  # Season=2025 means 2024-25
BEST_OPTION_A = "optA_ensemble"
BEST_OPTION_B = "optB_xgboost"

MODEL_NAMES = [
    "optA_xgboost", "optA_mlp", "optA_autoencoder", "optA_ensemble",
    "optB_xgboost", "optB_mlp", "optB_autoencoder", "optB_ensemble",
]


def _label(year_end: int) -> str:
    return f"{year_end - 1}-{str(year_end)[2:]}"


def _load_artifacts() -> tuple[dict, list[str]]:
    feature_cols_path = MODELS_DIR / "feature_columns.json"
    if not feature_cols_path.exists():
        raise FileNotFoundError("feature_columns.json missing — run scripts/train_models.py first.")
    feature_cols = json.loads(feature_cols_path.read_text())
    models = {name: load_model(name, MODELS_DIR) for name in MODEL_NAMES}
    return models, feature_cols


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _find_player(name: str, stats: pd.DataFrame) -> pd.DataFrame:
    """Case- and accent-insensitive search: 'jokic' -> 'Nikola Jokić'."""
    nm = _strip_accents(name.lower().strip())
    normalized = stats["Player"].astype(str).str.lower().apply(_strip_accents)
    exact = stats[normalized == nm]
    if not exact.empty:
        return exact
    return stats[normalized.str.contains(nm, na=False, regex=False)]


def _suggest_names(query: str, stats: pd.DataFrame, n: int = 5) -> list[str]:
    candidates = stats["Player"].dropna().astype(str).unique().tolist()
    norm_query = _strip_accents(query.lower())
    norm_map = {_strip_accents(c.lower()): c for c in candidates}
    matches = get_close_matches(norm_query, list(norm_map.keys()), n=n, cutoff=0.5)
    return [norm_map[m] for m in matches]


def _align_features(row: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = prepare_features(row)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    return X[feature_cols]


def _ovr_from_predicted_stats(pred_stats: np.ndarray, next_age: float, reference_dist: pd.DataFrame) -> float:
    """Apply OVR formula to a single predicted stat row, using ``reference_dist`` for percentile rank."""
    pcts = {}
    for stat, val in zip(TARGET_STATS, pred_stats):
        ref_vals = reference_dist[stat].dropna().values
        pcts[stat] = float((ref_vals < val).mean() * 100) if ref_vals.size else 50.0

    production  = float(np.mean([pcts["VORP"], pcts["BPM"], pcts["WS"]]))
    efficiency  = float(np.mean([pcts["PER"], pcts["TS%"]]))
    role        = float(np.mean([pcts["USG%"], pcts["MP"]]))
    g_pred      = float(pred_stats[TARGET_STATS.index("G")])
    availability = max(0.0, min(100.0, g_pred / 82.0 * 100.0))
    age_s = age_curve(next_age)

    ovr = (OVR_WEIGHTS["production_score"]   * production
         + OVR_WEIGHTS["efficiency_score"]   * efficiency
         + OVR_WEIGHTS["role_score"]         * role
         + OVR_WEIGHTS["availability_score"] * availability
         + OVR_WEIGHTS["age_score"]          * age_s)
    return ovr


def _print_lookup(canonical: str, row: pd.Series) -> None:
    print(f"\n{canonical}")
    print(f"  {row['Season_Label']} | Team: {row['Team']} | Age: {int(row['Age'])} | G: {int(row['G'])} | MP: {int(row['MP'])}")
    if pd.notna(row.get("ovr")):
        ovr = float(row["ovr"])
        print(f"  Actual OVR :    {ovr:.1f}")
        print(f"  Trade value : {format_dollars(ovr_to_dollars(ovr))}  ({ovr_to_tier(ovr)})")
    else:
        print("  No OVR for this season (player did not meet qualification: MP>=500, G>=20).")


def _print_one_year(canonical: str, latest: pd.DataFrame, models: dict, feature_cols: list[str], target_year: int, ref_dist: pd.DataFrame) -> None:
    base_age = float(latest["Age"].iloc[0])
    print(f"\n{canonical}")
    print(f"  Latest known:  {latest['Season_Label'].iloc[0]} | Team: {latest['Team'].iloc[0]} | Age: {int(base_age)}")
    print(f"  Predicting:    {_label(target_year)}        | Age: {int(base_age + 1)}  (1 year ahead)")
    print()

    X = _align_features(latest, feature_cols)
    next_age = base_age + 1

    rows: list[tuple[str, float]] = []
    for name in sorted(k for k in models if k.startswith("optA_")):
        rows.append((name, float(models[name].predict(X)[0])))
    for name in sorted(k for k in models if k.startswith("optB_")):
        pred_stats = np.asarray(models[name].predict(X)[0])
        rows.append((name, _ovr_from_predicted_stats(pred_stats, next_age, ref_dist)))

    print(f"  {'model':<24} {'predicted OVR':>14} {'trade value':>16} {'tier':<18}")
    print(f"  {'-' * 24} {'-' * 14} {'-' * 16} {'-' * 18}")
    for name, val in rows:
        dollars = format_dollars(ovr_to_dollars(val))
        tier = ovr_to_tier(val)
        print(f"  {name:<24} {val:>14.1f} {dollars:>16} {tier:<18}")
    print(f"\n  Note: typical model error is ~10 OVR points (MAE on held-out test set).")
    print(f"  Trade value is a tier-based mapping of OVR -> $ (no salary model trained yet).")


def _print_multi_year(canonical: str, latest: pd.DataFrame, models: dict, feature_cols: list[str], target_year: int, ref_dist: pd.DataFrame) -> None:
    base_age = float(latest["Age"].iloc[0])
    years_ahead = target_year - LATEST_KNOWN_SEASON

    print(f"\n{canonical}")
    print(f"  Latest known:  {latest['Season_Label'].iloc[0]} | Team: {latest['Team'].iloc[0]} | Age: {int(base_age)}")
    print(f"  Predicting:    {_label(target_year)}        | Age: {int(base_age + years_ahead)}  ({years_ahead} years ahead)")
    print(f"  Method: iterative roll-forward using {BEST_OPTION_B} (errors compound)")
    print()

    print(f"  {'season':<10} {'age':>4} {'predicted OVR':>14} {'trade value':>14} {'tier':<18}")
    print(f"  {'-' * 10} {'-' * 4} {'-' * 14} {'-' * 14} {'-' * 18}")

    model_b = models[BEST_OPTION_B]
    current_row = latest.copy()
    current_age = base_age

    for step in range(1, years_ahead + 1):
        X = _align_features(current_row, feature_cols)
        pred_stats = np.asarray(model_b.predict(X)[0])
        current_age += 1
        ovr = _ovr_from_predicted_stats(pred_stats, current_age, ref_dist)

        season_year = LATEST_KNOWN_SEASON + step
        dollars = format_dollars(ovr_to_dollars(ovr))
        tier = ovr_to_tier(ovr)
        print(f"  {_label(season_year):<10} {int(current_age):>4} {ovr:>14.1f} {dollars:>14} {tier:<18}")

        # Carry predicted stats forward into the next iteration's feature row
        new_row = current_row.copy()
        for stat, val in zip(TARGET_STATS, pred_stats):
            new_row[stat] = val
        new_row["Age"] = current_age
        current_row = new_row.reset_index(drop=True)

    print(f"\n  Note: each year past the first compounds prediction error. Treat as a rough trend, not precise values.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict a player's OVR for a target NBA season.")
    ap.add_argument("player", help='Player name (e.g., "LeBron James")')
    ap.add_argument("year", type=int, help="Year-end of target season (2026 = 2025-26)")
    args = ap.parse_args()

    stats = pd.read_csv(CLEAN)
    scores = pd.read_csv(SCORES)
    models, feature_cols = _load_artifacts()

    p_stats = _find_player(args.player, stats)
    if p_stats.empty:
        print(f"Player '{args.player}' not found.", file=sys.stderr)
        suggestions = _suggest_names(args.player, stats)
        if suggestions:
            print(f"Did you mean: {', '.join(suggestions)}?", file=sys.stderr)
        sys.exit(1)

    canonical = p_stats["Player"].mode().iloc[0]
    p_stats = p_stats[p_stats["Player"] == canonical]
    p_full = p_stats.merge(scores[["Player", "Season", "ovr"]], on=["Player", "Season"], how="left")
    p_full = p_full.sort_values("MP", ascending=False).drop_duplicates(["Player", "Season"], keep="first")
    p_full = p_full.sort_values("Season").reset_index(drop=True)

    # Past/current season — lookup
    if args.year <= LATEST_KNOWN_SEASON:
        match = p_full[p_full["Season"] == args.year]
        if match.empty:
            print(f"\nNo data for {canonical} in {_label(args.year)}.", file=sys.stderr)
            sys.exit(1)
        _print_lookup(canonical, match.iloc[0])
        return

    # Future season — predict
    if p_full.empty:
        print(f"No history for {canonical} to predict from.", file=sys.stderr)
        sys.exit(1)

    latest = p_full.iloc[[-1]].reset_index(drop=True)
    ref_dist = stats[stats["Season"] == LATEST_KNOWN_SEASON]

    if args.year - LATEST_KNOWN_SEASON == 1:
        _print_one_year(canonical, latest, models, feature_cols, args.year, ref_dist)
    else:
        _print_multi_year(canonical, latest, models, feature_cols, args.year, ref_dist)


if __name__ == "__main__":
    main()
