# NBA Trade Value Model

Predict an NBA player's overall rating (0–100) for any season — past, current, or future — and convert that rating into a fair-market dollar value. Built on 16 seasons of advanced stats from Basketball-Reference (2009-10 through 2024-25).

## What's in the box

- **A 0–100 OVR** computed per player-season, calibrated against league peers within each season (so a 5 VORP in 2014 isn't compared to a 5 VORP in 2024).
- **Eight forecasting models** trained to predict next-season OVR: XGBoost, MLP, Autoencoder + KNN, and an Ensemble — each in two flavors (predict OVR directly, or predict 8 underlying stats then apply the formula).
- **A CLI tool** that takes a player name and a year and prints the OVR + a tiered trade value in dollars.
- **An inspectable model store** — every trained model is saved in human-readable JSON / standard ML formats. No pickle.

## Setup

```bash
# Optional: create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Note: on macOS you may need `brew install libomp` for XGBoost.

## Pipeline — four commands, in order

Each step reads its predecessor's output. Run them top-to-bottom on a fresh clone:

```bash
python3 scripts/run_cleaning.py            # 1. raw stats  -> cleaned CSV
python3 scripts/run_features.py            # 2. cleaned    -> per-season OVR scores
python3 scripts/build_training_pairs.py    # 3. scores     -> (year N, year N+1) pairs
python3 scripts/train_models.py            # 4. pairs      -> 8 trained models + predictions
```

After step 4 you have everything needed to use the CLI tool below.

| Step | Reads | Writes |
|---|---|---|
| 1 — `run_cleaning.py` | `data/raw/advanced_stats_2010_2025.csv` | `data/processed/advanced_stats_clean.csv` |
| 2 — `run_features.py` | `data/processed/advanced_stats_clean.csv` | `data/processed/player_scores.csv` |
| 3 — `build_training_pairs.py` | `advanced_stats_clean.csv` + `player_scores.csv` | `data/processed/training_pairs.csv` |
| 4 — `train_models.py` | `training_pairs.csv` | `outputs/models/<name>/` (8 dirs) + `outputs/predictions/test_predictions.csv` |

## CLI tool — predict an OVR and trade value

```bash
python3 scripts/predict_ovr.py "<player>" <year>
```

`year` is the year-end of the NBA season (so `2026` = 2025-26).

### Three modes (auto-selected by year)

**Past or current season** (`year ≤ 2025`) — looks up the actual OVR from the data:

```bash
$ python3 scripts/predict_ovr.py "Stephen Curry" 2024

Stephen Curry
  2023-24 | Team: GSW | Age: 35 | G: 74 | MP: 2421
  Actual OVR (from data):    89.3
  Trade value (Stage 2 / A): $40.6M  (All-Star)
```

**One year ahead** (`year = 2026`) — runs all 8 models, marks the best with `<- best`:

```bash
$ python3 scripts/predict_ovr.py "jokic" 2026

Nikola Jokić
  Latest known:  2024-25 | Team: DEN | Age: 29
  Predicting:    2025-26 | Age: 30  (1 year ahead)

  model                     predicted OVR      trade value tier
  optA_ensemble                      97.7           $52.7M Superstar          <- best
  optA_xgboost                       90.0           $41.9M All-Star
  optA_mlp                          105.5           $55.0M Superstar
  optA_autoencoder                   91.9           $45.0M All-Star
  optB_xgboost                       93.3           $47.3M All-Star
  optB_mlp                           42.1            $2.5M Marginal
  optB_autoencoder                   92.8           $46.4M All-Star
  optB_ensemble                      77.4           $20.9M Quality Starter
```

**Multi-year** (`year ≥ 2027`) — iterative roll-forward using the best Option B model:

```bash
$ python3 scripts/predict_ovr.py "LeBron James" 2028

LeBron James
  Latest known:  2024-25 | Team: LAL | Age: 40
  Predicting:    2027-28 | Age: 43  (3 years ahead)
  Method: iterative roll-forward using optB_xgboost (errors compound)

  season      age  predicted OVR    trade value tier
  2025-26      41           80.7         $25.1M Quality Starter
  2026-27      42           77.5         $21.0M Quality Starter
  2027-28      43           74.2         $17.2M Rotation
```

### Quality of life

- **Accent / case insensitive name matching** — `"jokic"` finds `"Nikola Jokić"`, `"LUKA DONCIC"` finds `"Luka Dončić"`.
- **Typo suggestions** — wrong name prints "Did you mean: ..." with up to 5 close matches.

### Tier mapping (OVR → $)

Linear interpolation calibrated against the 2025-26 NBA cap structure. Defined in [src/models/trade_value.py](src/models/trade_value.py) — easy to retune.

| OVR | Tier | Approx $ |
|---|---|---|
| 95+ | Superstar | $50–55M |
| 85–94 | All-Star | $32–42M |
| 75–84 | Quality Starter | $18–24M |
| 65–74 | Rotation | $9–13M |
| 50–64 | Bench | $3–5M |
| <50 | Marginal | <$2.4M |

## Inspecting trained models

```bash
python3 scripts/inspect_model.py                 # list all 8 models
python3 scripts/inspect_model.py optA_xgboost    # detail one model
```

Or open the model files directly — they're all in human-readable formats:

```bash
code outputs/models/optA_xgboost/model.json      # XGBoost decision trees as JSON
code outputs/models/optA_mlp/weights.json        # MLP layer weights
code outputs/models/optA_ensemble/config.json    # ensemble component list
```

## Comparison notebook

```bash
jupyter notebook notebooks/model_comparison.ipynb
```

Pre-rendered with metrics tables, predicted-vs-actual scatters, residual plots, and a notable-players bar chart. Run on the held-out 2024-25 test season.

## Project structure

```
nba_mlproject/
├── data/
│   ├── raw/
│   │   ├── advanced_stats_2010_2025.csv     # 16 seasons of player stats
│   │   └── player_salary.csv                # current contracts (2025-26 onward)
│   └── processed/                            # all generated by the pipeline
├── src/
│   ├── data/clean_data.py                   # cleaning logic
│   ├── features/build_features.py           # OVR computation
│   └── models/
│       ├── pairs.py                         # (N, N+1) training-pair builder
│       ├── preprocess.py                    # feature matrix prep
│       ├── formula.py                       # OVR formula (used by Option B)
│       ├── trade_value.py                   # tier-based OVR -> $ mapping
│       └── models.py                        # XGBoost, MLP, Autoencoder, Ensemble + save/load
├── scripts/
│   ├── run_cleaning.py
│   ├── run_features.py
│   ├── build_training_pairs.py
│   ├── train_models.py
│   ├── predict_ovr.py                       # the user-facing CLI
│   └── inspect_model.py                     # peek inside a saved model
├── notebooks/
│   └── model_comparison.ipynb               # plots + metrics
├── outputs/
│   ├── models/                              # 8 trained models, native formats
│   └── predictions/                         # test_predictions.csv
└── requirements.txt
```

## Test-set performance

Trained on (year N → year N+1) pairs from 2009-10 through 2021-22, tested on 2023-24 → 2024-25 outcomes (~290 player-seasons).

| Model | MAE | RMSE | R² | Top-10 overlap |
|---|---:|---:|---:|---:|
| **optA_ensemble** | **10.17** | 12.87 | **0.528** | 5/10 |
| optA_mlp | 10.41 | 13.19 | 0.504 | 4/10 |
| optA_xgboost | 10.58 | 13.21 | 0.502 | **7/10** |
| optA_autoencoder | 11.23 | 13.93 | 0.447 | **7/10** |
| optB_xgboost | 11.45 | 14.56 | 0.396 | **7/10** |
| optB_autoencoder | 11.72 | 15.47 | 0.318 | **7/10** |
| optB_ensemble | 13.77 | 16.66 | 0.209 | 6/10 |
| optB_mlp | 19.79 | 23.82 | -0.618 | 3/10 |

**MAE ~10** means a typical prediction is within 10 OVR points of the actual. The ensemble is best for raw accuracy; XGBoost/autoencoder do best at correctly identifying the top 10.

## What's not built (yet)

- **Real surplus value** — current trade value is a deterministic OVR → $ tier mapping. A learned salary regression (predicting fair price from stats, then comparing to actual contract) would let you say "Chet Holmgren is +$26M of surplus" instead of just "tier All-Star." Needs historical salary data — not yet scraped.
- **Multi-year confidence bands** — multi-year OVR forecasts roll forward year by year and errors compound. There's no uncertainty estimate on the output.
- **Contract-aware features** — years of guaranteed control, no-trade clauses, supermax flags, rookie-scale flags. None of these feed into the current model.

## Notes

- Raw data was scraped from Basketball-Reference. The 2024-25 portion needed manual reconstruction due to a header-corruption bug in the original scraper; that's been fixed and the raw file is correct.
- All trained models are saved in inspectable native formats (`.json`, `.pt`, `.npy`) 
