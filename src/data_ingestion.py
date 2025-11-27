"""Data ingestion utilities for nflfastR play-by-play and schedule data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from nfl_data_py import import_pbp_data, import_schedules

from .config import RAW_DIR, SEASONS, ensure_directories


def load_pbp_raw(seasons: Iterable[int]) -> pd.DataFrame:
    """Load play-by-play data for given seasons and filter to regular season plays."""
    df = import_pbp_data(list(seasons))
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    return df.reset_index(drop=True)


def load_games_raw(seasons: Iterable[int]) -> pd.DataFrame:
    """Load schedule / game-level data for given seasons and filter to regular season."""
    games = import_schedules(list(seasons))
    if "season_type" in games.columns:
        games = games[games["season_type"] == "REG"]
    elif "game_type" in games.columns:
        games = games[games["game_type"] == "REG"]
    return games.reset_index(drop=True)


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_raw_parquet() -> None:
    """Load and persist raw pbp and schedule data to the raw directory."""
    ensure_directories()

    pbp = load_pbp_raw(SEASONS)
    games = load_games_raw(SEASONS)

    pbp_path = RAW_DIR / "pbp_2010_2023.parquet"
    games_path = RAW_DIR / "games_2010_2023.parquet"

    _save_parquet(pbp, pbp_path)
    _save_parquet(games, games_path)


if __name__ == "__main__":
    save_raw_parquet()
