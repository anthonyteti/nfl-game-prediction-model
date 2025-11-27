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

## Future Directions

- Add categorical encodings (team, home/away splits) and calibrate probabilities.
  - Evaluate “model vs market” differentials to focus on high-confidence bets.
- Incorporate injury/travel/weather adjustments or player-level advanced metrics (QB EPA, pressure rates).
- Experiment with alternative algorithms (LightGBM, XGBoost) or neural nets.
- Package the pipeline for automated retraining (e.g., Prefect, Airflow) and add CI for unit tests/linting.

## License

This repository is for educational and portfolio purposes. NFL data belongs to its respective providers (nflverse / nflfastR). Use responsibly according to their terms of service.
