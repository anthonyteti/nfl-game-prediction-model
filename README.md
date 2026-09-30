# NFL Game Prediction Pipeline

End-to-end machine learning project that ingests nflverse / `nflfastR` data via `nfl_data_py`, engineers team-week and game-level features, trains baseline models, and surfaces predictions through a Streamlit dashboard.

## Highlights

- **Data pipeline**: python modules convert raw play-by-play and schedule tables into season-to-date team week features and game-level modeling sets.
- **Modeling**: Gradient Boosting models predict (1) home win probability and (2) expected point differential with leakage controls and season-based train/val/test splits.
- **Interactive UI**: Streamlit dashboard allows exploration of predictions, week summaries, and season hit rates (moneyline & spread) for 2019+ out-of-sample seasons.
- **Extensible**: Feature engineering is centralized, so adding more advanced metrics (CP/POE, QB stats, etc.) flows through the entire pipeline.

## Repository Structure

```
nfl-game-model/
├── data/
│   ├── raw/                # cached nfl_data_py parquet files
│   └── processed/          # team-week and game-level features
├── models/                 # serialized sklearn models + imputers
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── config.py
│   ├── data_ingestion.py        # nfl_data_py loaders
│   ├── team_week_stats.py       # offense/defense aggregates
│   ├── team_week_features.py    # season-to-date + rolling stats, rest/env
│   ├── game_features.py         # joins to game labels
│   ├── train_win_model.py
│   ├── train_margin_model.py
│   ├── evaluate_win_model.py
│   └── predict.py
├── app.py                  # Streamlit dashboard
└── main.py                 # convenience pipeline runner
```

## Getting Started

1. **Install Python**: project tested on Python 3.12 (`py -3.12` on Windows).  
2. **Install dependencies**:
   ```bash
   py -3.12 -m pip install --upgrade pip setuptools wheel
   py -3.12 -m pip install pandas numpy pyarrow nfl_data_py scikit-learn joblib streamlit
   ```
3. **Fetch data + build features** (run from repo root):
   ```bash
   py -3.12 -m src.data_ingestion
   py -3.12 -m src.team_week_stats
   py -3.12 -m src.team_week_features
   py -3.12 -m src.game_features
   ```
4. **Train models**:
   ```bash
   py -3.12 -m src.train_win_model
   py -3.12 -m src.train_margin_model
   py -3.12 -m src.evaluate_win_model   # evaluates 2021-2023 seasons
   ```

## Streamlit Dashboard

Launch the UI to inspect predictions and season summaries (2019+):

```bash
py -3.12 -m streamlit run app.py
```

Features:
- Sidebar selectors for season/week/matchup (limited to seasons not used in training).
- Week-level hit rate metrics for moneyline and spread signals.
- Season summary tab with hit rates for each season (2019+).
- Game explorer tab showing predicted probabilities, key features, final score, and week overview.

## Modeling Details

- **Targets**:
  - `home_win`: binary home team win label.
  - `point_diff`: home score minus away score.
- **Splits**:
  - Train: 2010–2018, Validation: 2019–2020, Test: 2021–2023.
- **Features**:
  - Season-to-date and last-four averages for offensive/defensive EPA, success rate, pass/rush splits, CP/CPOE, QB EPA, air/yac EPA, xpass, pass_oe.
  - Rest days, roof/surface, Vegas spread/total, division flag, simple feature diffs (rest, EPA).
- **Models**: `sklearn.ensemble.GradientBoostingClassifier/Regressor` with median imputation. Artifacts stored under `models/`.
- **Leakage controls**: feature builders drop per-week raw stats after computing time-aware aggregates. All modeling scripts share a leaky-column exclusion list to prevent final scores/labels from entering the feature matrix.

## Validation and reproducibility

### Reproduced held-out results

Run `python -m src.reproduce` from the repository root to download/cache the
required play-by-play columns, rebuild features, train both original models,
and evaluate fixed win-probability baselines. Install the exact run dependencies
with `python -m pip install -r requirements-reproduction.txt` in a Python 3.11
virtual environment. The download requires internet access.

The fixed protocol uses 2010–2018 for training, 2019–2020 for validation, and
2021–2023 for testing. No hyperparameters were tuned in this reproduction.

| Model | Test accuracy | Log loss ↓ | Brier score ↓ | ROC AUC ↑ |
|---|---:|---:|---:|---:|
| Gradient boosting with engineered features | 62.33% | 0.6654 | 0.2342 | 0.6596 |
| Training-set home-win prior | 54.23% | 0.6912 | 0.2490 | 0.5000 |
| Spread-only logistic regression | 65.52% | 0.6194 | 0.2157 | 0.7108 |

Test set: **815 completed regular-season games**, including three ties labeled
as home non-wins. Always predicting a home win also achieves 54.23% accuracy.
The engineered model beats the constant baseline but underperforms the
spread-only model on all four reported metrics. This run does not demonstrate
incremental predictive value over the spread baseline.

For home point differential, the gradient-boosting regressor has **10.21-point
MAE** and **13.25-point RMSE**, versus **10.93 / 14.11** for the training-mean
baseline and **9.83 / 12.77** for spread-only linear regression. All models are
fit on the training seasons only.

See [metrics](results/reproduction/metrics.json),
[per-game predictions](results/reproduction/predictions.csv), and
[data hashes, dependency versions, and feature list](results/reproduction/manifest.json).
The data is a current nflverse download, not an archived historical release.
Recorded weather and spreads lack decision-time snapshots here, so these are
retrospective results, not evidence of deployable betting performance.
Two consecutive full pipeline runs in the pinned Python 3.11.1 environment
produced [identical metrics](results/reproduction/repeatability.json). A preliminary
run in the pre-existing Python 3.12 environment gave slightly different results;
the table above uses the fresh, pinned environment consistently.

Run the synthetic-data training smoke test from the repository root:

```bash
python -m unittest discover -s tests -v
```

This exercises the margin trainer, metric calculation, and artifact-save calls
without downloading NFL data. It does not establish model quality or validate
the complete ingestion pipeline. Dependencies are currently unpinned.

Dashboard seasons 2019–2020 are validation seasons; only 2021–2023 are held-out
test seasons. Keep those groups separate when reporting performance. The current
binary label treats a tie as `home_win = 0`; it is not a three-way outcome model.

## Future Directions

- Add categorical encodings (team, home/away splits) and calibrate probabilities.
  - Evaluate “model vs market” differentials to focus on high-confidence bets.
- Incorporate injury/travel/weather adjustments or player-level advanced metrics (QB EPA, pressure rates).
- Experiment with alternative algorithms (LightGBM, XGBoost) or neural nets.
