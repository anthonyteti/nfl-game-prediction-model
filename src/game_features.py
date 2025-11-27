"""Build game-level feature table joining home/away team-week stats."""

from __future__ import annotations

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR


GAMES_PATH = RAW_DIR / "games_2010_2023.parquet"
TEAM_WEEK_FEATURES_PATH = PROCESSED_DIR / "team_week_features.parquet"
OUTPUT_PATH = PROCESSED_DIR / "games_with_features.parquet"


def _prepare_team_features(prefix: str, df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in {"season", "week", "team"}]
    renamed = df.rename(columns={col: f"{prefix}_{col}" for col in feature_cols})
    renamed = renamed.rename(columns={"team": f"{prefix}_team"})
    return renamed


def build_game_features() -> pd.DataFrame:
    """Assemble game-level modeling table with home/away features and labels."""
    games = pd.read_parquet(GAMES_PATH)
    if "season_type" in games.columns:
        games = games[games["season_type"] == "REG"].copy()
    elif "game_type" in games.columns:
        games = games[games["game_type"] == "REG"].copy()
    games = games[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
            "total_line",
            "div_game",
            "roof",
            "surface",
        ]
    ]

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["point_diff"] = games["home_score"] - games["away_score"]

    team_features = pd.read_parquet(TEAM_WEEK_FEATURES_PATH)

    home_features = _prepare_team_features("home", team_features)
    games = games.merge(
        home_features,
        on=["season", "week", "home_team"],
        how="left",
    )

    away_features = _prepare_team_features("away", team_features)
    games = games.merge(
        away_features,
        on=["season", "week", "away_team"],
        how="left",
    )

    games["rest_diff"] = games["home_rest_days"] - games["away_rest_days"]
    games["off_epa_per_play_to_date_diff"] = (
        games["home_off_epa_per_play_to_date"] - games["away_off_epa_per_play_to_date"]
    )
    games["def_epa_per_play_to_date_diff"] = (
        games["home_def_epa_per_play_to_date"] - games["away_def_epa_per_play_to_date"]
    )

    games.to_parquet(OUTPUT_PATH, index=False)
    return games


if __name__ == "__main__":
    build_game_features()
