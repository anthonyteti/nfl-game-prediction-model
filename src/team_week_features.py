"""Create time-aware team-week feature tables."""

from __future__ import annotations

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR


TEAM_WEEK_RAW_PATH = PROCESSED_DIR / "team_week_raw.parquet"
TEAM_WEEK_FEATURES_PATH = PROCESSED_DIR / "team_week_features.parquet"
GAMES_PATH = RAW_DIR / "games_2010_2023.parquet"

FEATURE_COLS = [
    "off_epa_per_play",
    "off_success_rate",
    "off_pass_rate",
    "off_rush_rate",
    "off_pass_epa_per_play",
    "off_rush_epa_per_play",
    "off_cp",
    "off_cpoe",
    "off_qb_epa_per_play",
    "off_air_epa_per_play",
    "off_yac_epa_per_play",
    "off_xpass",
    "off_pass_oe",
    "def_epa_per_play",
    "def_success_rate",
    "def_pass_rate",
    "def_rush_rate",
    "def_pass_epa_per_play",
    "def_rush_epa_per_play",
    "def_cp_allowed",
    "def_cpoe_allowed",
    "def_qb_epa_per_play",
    "def_air_epa_per_play",
    "def_yac_epa_per_play",
    "def_xpass",
    "def_pass_oe",
]


def add_season_to_date_and_rolling(team_week: pd.DataFrame) -> pd.DataFrame:
    """Add season-to-date and rolling-4 stats using only prior weeks."""

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("week").copy()
        for col in FEATURE_COLS:
            expanding = group[col].expanding(min_periods=1).mean()
            rolling = group[col].rolling(window=4, min_periods=1).mean()
            group[f"{col}_to_date"] = expanding.shift(1)
            group[f"{col}_last4"] = rolling.shift(1)
        return group

    enriched = (
        team_week.groupby(["team", "season"], group_keys=False).apply(_apply).reset_index(drop=True)
    )
    drop_cols = FEATURE_COLS + ["plays_count", "plays_count_def"]
    enriched.drop(columns=drop_cols, inplace=True, errors="ignore")
    return enriched


def add_rest_and_env(team_week: pd.DataFrame) -> pd.DataFrame:
    """Add rest days and stadium environment info from the schedules table."""
    games = pd.read_parquet(GAMES_PATH)
    date_col = "game_date" if "game_date" in games.columns else "gameday"
    games["game_date"] = pd.to_datetime(games[date_col])

    keep_cols = [
        "game_id",
        "season",
        "week",
        "game_date",
        "roof",
        "surface",
        "temp",
        "wind",
    ]

    home = games[keep_cols + ["home_team"]].copy()
    home["team"] = home.pop("home_team")
    home["is_home"] = 1

    away = games[keep_cols + ["away_team"]].copy()
    away["team"] = away.pop("away_team")
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True)

    merged = pd.merge(
        team_week,
        team_games,
        on=["team", "season", "week"],
        how="left",
        suffixes=("", "_game"),
    )

    merged.sort_values(["team", "season", "week"], inplace=True)
    merged["prev_game_date"] = merged.groupby("team")["game_date"].shift(1)
    merged["rest_days"] = (merged["game_date"] - merged["prev_game_date"]).dt.days
    merged.drop(columns=["prev_game_date"], inplace=True)

    return merged


def build_team_week_features() -> pd.DataFrame:
    """Load team-week stats and build enriched features."""
    team_week = pd.read_parquet(TEAM_WEEK_RAW_PATH)
    team_week = add_season_to_date_and_rolling(team_week)
    team_week = add_rest_and_env(team_week)
    team_week.to_parquet(TEAM_WEEK_FEATURES_PATH, index=False)
    return team_week


if __name__ == "__main__":
    build_team_week_features()
