"""Rebuild the 2010–2023 experiment and save fixed-baseline comparisons.

Run from the repository root: python -m src.reproduce
"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nfl_data_py import import_pbp_data
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

from .config import RAW_DIR, PROCESSED_DIR, MODELS_DIR, SEASONS, ensure_directories
from .data_ingestion import load_games_raw
from .game_features import build_game_features
from .team_week_features import build_team_week_features
from .team_week_stats import build_team_week_stats
from .train_win_model import train_win_model
from .train_margin_model import train_margin_model

PBP_COLUMNS = ["season_type", "week", "posteam", "defteam", "play", "pass", "rush",
               "epa", "success", "cp", "cpoe", "qb_epa", "air_epa", "yac_epa", "xpass", "pass_oe"]


def score(y, probabilities):
    return {
        "accuracy": float(accuracy_score(y, probabilities >= 0.5)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def main():
    ensure_directories()
    # Cache individual seasons so interrupted downloads can resume. Request only
    # columns used by the original feature pipeline, without participation joins.
    frames = []
    for season in SEASONS:
        cache = RAW_DIR / f"pbp_{season}_features.parquet"
        if not cache.exists():
            frame = import_pbp_data([season], columns=PBP_COLUMNS, include_participation=False)
            if frame.empty or set(frame.season.unique()) != {season}:
                raise ValueError(f"Missing play-by-play data for {season}")
            frame.loc[frame.season_type == "REG"].to_parquet(cache, index=False)
        frames.append(pd.read_parquet(cache))
    pbp_path = RAW_DIR / "pbp_2010_2023.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(pbp_path, index=False)
    games_path = RAW_DIR / "games_2010_2023.parquet"
    if not games_path.exists():
        load_games_raw(SEASONS).to_parquet(games_path, index=False)
    games = pd.read_parquet(games_path)
    if games[["home_score", "away_score"]].isna().any().any():
        raise ValueError("Schedule contains unplayed games; resolve them before modeling.")
    if games.game_id.duplicated().any():
        raise ValueError("Duplicate games in schedule")
    build_team_week_stats()
    build_team_week_features()
    data = build_game_features()
    if len(data) != len(games) or data.game_id.duplicated().any():
        raise ValueError("Feature joins changed game cardinality")
    train_win_model()
    train_margin_model()

    train = data[data.season.between(2010, 2018)]
    prior = float(train.home_win.mean())
    spread_model = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(random_state=42))
    spread_model.fit(train[["spread_line"]], train.home_win)
    spread_margin = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
    spread_margin.fit(train[["spread_line"]], train.point_diff)
    model = joblib.load(MODELS_DIR / "win_model.pkl")
    imputer = joblib.load(MODELS_DIR / "win_preprocessor.pkl")
    feature_names = list(imputer.feature_names_in_)
    margin_model = joblib.load(MODELS_DIR / "margin_model.pkl")
    margin_imputer = joblib.load(MODELS_DIR / "margin_preprocessor.pkl")
    predictions = []
    metrics = {}
    for name, years in {"validation": [2019, 2020], "test": [2021, 2022, 2023]}.items():
        split = data[data.season.isin(years)].copy()
        probability_sets = {
            "gradient_boosting": model.predict_proba(imputer.transform(split[feature_names]))[:, 1],
            "training_home_win_prior": np.full(len(split), prior),
            "spread_only_logistic": spread_model.predict_proba(split[["spread_line"]])[:, 1],
        }
        metrics[name] = {"n_games": len(split), "ties_labeled_zero": int((split.point_diff == 0).sum()),
                         "always_home_accuracy": float(split.home_win.mean()),
                         "models": {k: score(split.home_win, p) for k, p in probability_sets.items()}}
        rows = split[["game_id", "season", "home_win", "point_diff", "spread_line"]].copy()
        rows["split"] = name
        for model_name, probabilities in probability_sets.items():
            rows[model_name] = probabilities
        margins = {
            "gradient_boosting": margin_model.predict(margin_imputer.transform(split[margin_imputer.feature_names_in_])),
            "training_mean": np.full(len(split), train.point_diff.mean()),
            "spread_only_linear": spread_margin.predict(split[["spread_line"]]),
        }
        metrics[name]["margin_models"] = {
            k: {"mae": float(mean_absolute_error(split.point_diff, p)),
                "rmse": float(np.sqrt(mean_squared_error(split.point_diff, p)))}
            for k, p in margins.items()
        }
        for model_name, points in margins.items():
            rows["margin_" + model_name] = points
        predictions.append(rows)

    output = Path("results/reproduction")
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions).to_csv(output / "predictions.csv", index=False)
    manifest = {
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "scikit-learn", "nfl-data-py", "pyarrow", "joblib"]},
        "source": "nflverse via nfl_data_py; current downloaded snapshot, not an archived historical release",
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [pbp_path, games_path, PROCESSED_DIR / "games_with_features.parquet"]},
        "n_train": len(train), "training_home_win_prior": prior,
        "feature_names": feature_names,
        "protocol": "2010–2018 training; 2019–2020 validation; 2021–2023 test. No tuning in this reproduction.",
        "limitations": ["Ties are home non-wins, consistent with original code.",
                        "Recorded weather is not a timestamped pregame forecast.",
                        "Historical spreads have no decision-time snapshot here.",
                        "Retrospective evaluation does not demonstrate live betting profitability."],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
