# NBA Trade Value Model

We built this to answer a question we kept running into watching NBA trade rumors: *what is a player actually worth?* Not in terms of vibes or Twitter consensus, but in dollars, tied to their on-court production.

The model pulls 16 seasons of advanced stats from Basketball-Reference (2009-10 through 2024-25), converts each player-season into a 0–100 OVR score calibrated within that year's league, and maps that OVR to a fair-market salary using the 2025-26 cap structure. You can look up historical ratings, predict next season, or roll forward multiple years for players like LeBron or Curry who you want to project into the future.

---

## What it does

- **OVR scoring (0–100)** — computed per player-season, normalized within each year so a 5 VORP in 2014 doesn't compare unfairly to a 5 VORP in 2024.
- **Eight forecasting models** — XGBoost, MLP, Autoencoder + KNN, and an Ensemble, each in two flavors: Option A predicts OVR directly, Option B predicts 8 underlying stats then runs them through the OVR formula.
- **CLI predictor** — give it a player name and a year, get back an OVR and a tiered trade value in dollars.
- **Readable model files** — every trained model is saved in JSON or standard ML formats. No pickle files, so you can actually open and inspect them.

---

## Setup

```bash
# Optional but recommended: create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

> On macOS you may need `brew install libomp` for XGBoost.

---

## Running the pipeline

Four scripts, run in order. Each one reads what the previous step wrote:

```bash
python3 scripts/run_cleaning.py            # raw stats -> cleaned CSV
python3 scripts/run_features.py            # cleaned   -> per-season OVR scores
python3 scripts/build_training_pairs.py    # scores    -> (year N, year N+1) pairs
python3 scripts/train_models.py            # pairs     -> 8 trained models + predictions
```

Once step 4 finishes you're ready to use the CLI.

| Step | Reads | Writes |
|---|---|---|
| 1 — `run_cleaning.py` | `data/raw/advanced_stats_2010_2025.csv` | `data/processed/advanced_stats_clean.csv` |
| 2 — `run_features.py` | `data/processed/advanced_stats_clean.csv` | `data/processed/player_scores.csv` |
| 3 — `build_training_pairs.py` | `advanced_stats_clean.csv` + `player_scores.csv` | `data/processed/training_pairs.csv` |
| 4 — `train_models.py` | `training_pairs.csv` | `outputs/models/<name>/` (8 dirs) + `outputs/predictions/test_predictions.csv` |

---

## CLI — predict OVR and trade value

```bash
python3 scripts/predict_ovr.py "<player>" <year>
```

`year` is the year-end of the NBA season — so `2026` means the 2025-26 season.

The tool auto-selects one of three modes based on the year you pass:

**Past or current season** (`year ≤ 2025`) — looks up the actual OVR from the data:

```bash
$ python3 scripts/predict_ovr.py "Stephen Curry" 2024

Stephen Curry
  2023-24 | Team: GSW | Age: 35 | G: 74 | MP: 2421
  Actual OVR (from data):    89.3
  Trade value (Stage 2 / A): $40.6M  (All-Star)
```

**One year ahead** (`year = 2026`) — runs all 8 models and marks the best one:

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

**Multi-year** (`year ≥ 2027`) — rolls forward year-by-year using the best Option B model. Errors compound the further out you go, so take these with some skepticism:

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

A few quality-of-life things I added:
- **Name matching ignores accents and case** — `"jokic"` finds `"Nikola Jokić"`, `"LUKA DONCIC"` finds `"Luka Dončić"`.
- **Typo suggestions** — if the name doesn't match, it prints "Did you mean: ..." with up to 5 close options.

### How OVR maps to dollars

Calibrated against the 2025-26 NBA cap. The full interpolation logic is in [src/models/trade_value.py](src/models/trade_value.py) if you want to retune it.

| OVR | Tier | Approx $ |
|---|---|---|
| 95+ | Superstar | $50–55M |
| 85–94 | All-Star | $32–42M |
| 75–84 | Quality Starter | $18–24M |
| 65–74 | Rotation | $9–13M |
| 50–64 | Bench | $3–5M |
| <50 | Marginal | <$2.4M |

---

## Inspecting trained models

```bash
python3 scripts/inspect_model.py                 # list all 8 models
python3 scripts/inspect_model.py optA_xgboost    # detail one model
```

Or open the files directly — they're in readable formats:

```bash
code outputs/models/optA_xgboost/model.json      # XGBoost decision trees
code outputs/models/optA_mlp/weights.json        # MLP layer weights
code outputs/models/optA_ensemble/config.json    # ensemble component list
```

---

## Comparison notebook

```bash
jupyter notebook notebooks/model_comparison.ipynb
```

Pre-rendered with metrics tables, predicted-vs-actual scatter plots, residual plots, and a notable-players bar chart. Everything is run on the held-out 2024-25 test season.

---

## Project structure

```
nba_mlproject/
├── data/
│   ├── raw/
│   │   ├── advanced_stats_2010_2025.csv     # 16 seasons of player stats
│   │   └── player_salary.csv                # current contracts (2025-26 onward)
│   └── processed/                            # generated by the pipeline
├── src/
│   ├── data/clean_data.py                   # cleaning logic
│   ├── features/build_features.py           # OVR computation
│   └── models/
│       ├── pairs.py                         # (N, N+1) training-pair builder
│       ├── preprocess.py                    # feature matrix prep
│       ├── formula.py                       # OVR formula (used by Option B)
│       ├── trade_value.py                   # OVR -> $ mapping
│       └── models.py                        # XGBoost, MLP, Autoencoder, Ensemble
├── scripts/
│   ├── run_cleaning.py
│   ├── run_features.py
│   ├── build_training_pairs.py
│   ├── train_models.py
│   ├── predict_ovr.py                       # main CLI
│   └── inspect_model.py                     # peek inside a saved model
├── notebooks/
│   └── model_comparison.ipynb
├── outputs/
│   ├── models/                              # 8 trained models
│   └── predictions/                         # test_predictions.csv
└── requirements.txt
```

---

## Model performance

Trained on year-N → year-N+1 pairs from 2009-10 through 2021-22, tested on 2023-24 → 2024-25 outcomes (~290 player-seasons).

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

MAE of ~10 means the typical prediction lands within 10 OVR points of reality. The ensemble wins on raw accuracy; XGBoost and Autoencoder are better at identifying which players actually end up in the top 10.

---

## What's missing / what we would add next

- **Surplus value** — right now the trade value is a deterministic OVR → dollar mapping. What we actually want is a learned salary regression that predicts what a player *should* be paid based on their stats, then computes the gap from their actual contract. That would let you say "Chet Holmgren is worth $26M more than he costs" instead of just "tier: All-Star." We need historical salary data for that, which we haven't scraped yet.
- **Confidence bands on multi-year forecasts** — the roll-forward predictions just give you a single number, but uncertainty grows every year you project out. We would like to attach some kind of confidence interval to those.
- **Contract-aware features** — years of guaranteed control, no-trade clauses, supermax eligibility, rookie scale status. None of that feeds into the model right now, and it matters a lot for actual trade value.

---

## Notes

The raw data was scraped from Basketball-Reference. The 2024-25 portion needed manual reconstruction because of a header-corruption bug in the original scraper — that's been fixed and the raw CSV is correct. All trained models are saved in inspectable native formats (`.json`, `.pt`, `.npy`) — no pickle files anywhere.
