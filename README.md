# NBA Trade Value Model

Model NBA player trade value on a 0-100 scale using player performance and salary context.

## Project Goal

- Produce a `trade_value_score` where:
- Elite, high-impact players (e.g., Luka Doncic) score high.
- Overpaid, low-production players score lower.
- Output includes a final score and interpretable sub-scores.

## Current Scope

- Data cleaning notebooks and CSV assets are included.
- Repository structure and project scaffolding are set up.
- Main modeling pipeline implementation is intentionally not added yet.

## Repository Layout

- `data/raw/`: source datasets
- `data/processed/`: cleaned/merged datasets for modeling
- `notebooks/`: exploratory and reporting notebooks
- `src/`: reusable Python package code (modules scaffolded)
- `reports/`: figures and written findings
- `outputs/`: generated score tables and exports

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Review `data/README.md` for dataset definitions.
4. Run notebooks in `notebooks/` for exploration.

## Data Notes

- Existing files in `data/` include advanced stats and salary tables.
- Keep raw source files unchanged and write transformed tables to `data/processed/`.

## Next Implementation Step

Implement the first version of:

- feature engineering
- trade value scoring logic
- model training/evaluation pipeline