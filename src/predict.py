"""Prediction utilities for single games."""

from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, PROCESSED_DIR
from .feature_lists import LEAKY_NUMERIC_COLS


DATA_PATH = PROCESSED_DIR / "games_with_features.parquet"
WIN_MODEL_PATH = MODELS_DIR / "win_model.pkl"
WIN_PREPROCESSOR_PATH = MODELS_DIR / "win_preprocessor.pkl"
MARGIN_MODEL_PATH = MODELS_DIR / "margin_model.pkl"
MARGIN_PREPROCESSOR_PATH = MODELS_DIR / "margin_preprocessor.pkl"


def _load_artifacts():
    win_model = joblib.load(WIN_MODEL_PATH)
    win_preprocessor = joblib.load(WIN_PREPROCESSOR_PATH)
    margin_model = joblib.load(MARGIN_MODEL_PATH)
    margin_preprocessor = joblib.load(MARGIN_PREPROCESSOR_PATH)
    return win_model, win_preprocessor, margin_model, margin_preprocessor


def _get_game_row(
    df: pd.DataFrame,
    game_id: Optional[str],
    season: Optional[int],
    week: Optional[int],
    home_team: Optional[str],
    away_team: Optional[str],
) -> pd.Series:
    if game_id is not None:
        row = df[df["game_id"] == game_id]
    else:
        if None in {season, week, home_team, away_team}:
            raise ValueError("Must supply game_id or (season, week, home_team, away_team).")
        row = df[
            (df["season"] == season)
            & (df["week"] == week)
            & (df["home_team"] == home_team)
            & (df["away_team"] == away_team)
        ]
    if row.empty:
        raise ValueError("Game not found in features table.")
    return row.iloc[0]


def _get_feature_cols(df: pd.DataFrame, drop_cols: list[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = set(drop_cols) | LEAKY_NUMERIC_COLS
    return [col for col in numeric_cols if col not in drop]


def predict_game(
    game_id: Optional[str] = None,
    *,
    season: Optional[int] = None,
    week: Optional[int] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
) -> dict[str, float]:
    """Return win probability and expected point differential for a single game."""
    df = pd.read_parquet(DATA_PATH)
    row = _get_game_row(df, game_id, season, week, home_team, away_team)
    row_df = row.to_frame().T

    win_feature_cols = _get_feature_cols(df, ["home_win"])
    margin_feature_cols = _get_feature_cols(df, ["point_diff"])

    win_model, win_preprocessor, margin_model, margin_preprocessor = _load_artifacts()

    win_features = row_df[win_feature_cols]
    win_transformed = win_preprocessor.transform(win_features)
    win_proba = win_model.predict_proba(win_transformed)[0, 1]

    margin_features = row_df[margin_feature_cols]
    margin_transformed = margin_preprocessor.transform(margin_features)
    point_diff_pred = margin_model.predict(margin_transformed)[0]

    return {
        "game_id": row["game_id"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "pred_home_win_proba": float(win_proba),
        "pred_point_diff": float(point_diff_pred),
    }


if __name__ == "__main__":
    example = predict_game(game_id=None, season=2023, week=1, home_team="KC", away_team="DET")
    print(example)
