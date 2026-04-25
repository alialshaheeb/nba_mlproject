# Data Dictionary and Conventions

## Folders

- `raw/`: untouched source files
- `processed/`: cleaned and merged modeling tables

## Existing Project Data Files

- `advanced_stats_all.csv`: larger advanced stats table
- `advanced_stats_clean.csv`: cleaned subset used in analysis
- `player_salary.csv`: raw salary data
- `player_salary_clean.csv`: cleaned salary data

## Suggested Modeling Table

Create a processed table with one row per `player-season` and fields like:

- identity: `player`, `team`, `season`, `pos`, `age`
- performance: `PER`, `TS%`, `BPM`, `VORP`, `WS/48`, `USG%`
- availability proxies: `G`, `GS`, `MP`
- contract context: `salary`, `salary_rank`, `salary_percentile`
- outputs: `trade_value_score_100`, optional component scores

## Notes

- Do not overwrite raw files.
- Keep column naming consistent (snake_case preferred for new processed outputs).
