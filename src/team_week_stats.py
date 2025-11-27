"""Aggregate play-by-play data into team-week level stats."""

from __future__ import annotations

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR


PBP_PATH = RAW_DIR / "pbp_2010_2023.parquet"
OUTPUT_PATH = PROCESSED_DIR / "team_week_raw.parquet"
ADVANCED_COLS = ["cp", "cpoe", "qb_epa", "air_epa", "yac_epa", "xpass", "pass_oe"]


def _calc_offense_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    grouped = pbp.groupby(["season", "week", "posteam"], dropna=False)

    def _offense_row(g: pd.DataFrame) -> pd.Series:
        pass_mask = g["pass"] == 1
        rush_mask = g["rush"] == 1
        return pd.Series(
            {
                "plays_count": len(g),
                "off_epa_per_play": g["epa"].mean(),
                "off_success_rate": g["success"].mean(),
                "off_pass_rate": g["pass"].mean(),
                "off_rush_rate": g["rush"].mean(),
                "off_pass_epa_per_play": g.loc[pass_mask, "epa"].mean(),
                "off_rush_epa_per_play": g.loc[rush_mask, "epa"].mean(),
                "off_cp": g.loc[pass_mask, "cp"].mean(),
                "off_cpoe": g.loc[pass_mask, "cpoe"].mean(),
                "off_qb_epa_per_play": g["qb_epa"].mean(),
                "off_air_epa_per_play": g.loc[pass_mask, "air_epa"].mean(),
                "off_yac_epa_per_play": g.loc[pass_mask, "yac_epa"].mean(),
                "off_xpass": g["xpass"].mean(),
                "off_pass_oe": g["pass_oe"].mean(),
            }
        )

    offense = grouped.apply(_offense_row).reset_index()
    offense = offense.rename(columns={"posteam": "team"})
    return offense


def _calc_defense_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    grouped = pbp.groupby(["season", "week", "defteam"], dropna=False)

    def _defense_row(g: pd.DataFrame) -> pd.Series:
        pass_mask = g["pass"] == 1
        rush_mask = g["rush"] == 1
        return pd.Series(
            {
                "plays_count_def": len(g),
                "def_epa_per_play": g["epa"].mean(),
                "def_success_rate": g["success"].mean(),
                "def_pass_rate": g["pass"].mean(),
                "def_rush_rate": g["rush"].mean(),
                "def_pass_epa_per_play": g.loc[pass_mask, "epa"].mean(),
                "def_rush_epa_per_play": g.loc[rush_mask, "epa"].mean(),
                "def_cp_allowed": g.loc[pass_mask, "cp"].mean(),
                "def_cpoe_allowed": g.loc[pass_mask, "cpoe"].mean(),
                "def_qb_epa_per_play": g["qb_epa"].mean(),
                "def_air_epa_per_play": g.loc[pass_mask, "air_epa"].mean(),
                "def_yac_epa_per_play": g.loc[pass_mask, "yac_epa"].mean(),
                "def_xpass": g["xpass"].mean(),
                "def_pass_oe": g["pass_oe"].mean(),
            }
        )

    defense = grouped.apply(_defense_row).reset_index()
    defense = defense.rename(columns={"defteam": "team"})
    return defense


def build_team_week_stats() -> pd.DataFrame:
    """Load raw pbp data and compute offense/defense team-week aggregates."""
    pbp = pd.read_parquet(PBP_PATH)
    pbp = pbp[pbp["play"] == 1].copy()
    for col in ADVANCED_COLS:
        if col not in pbp.columns:
            pbp[col] = pd.NA

    offense = _calc_offense_stats(pbp)
    defense = _calc_defense_stats(pbp)

    team_week = pd.merge(
        offense,
        defense,
        on=["season", "week", "team"],
        how="outer",
        sort=True,
    )

    team_week.sort_values(["team", "season", "week"], inplace=True)
    team_week.reset_index(drop=True, inplace=True)
    team_week.to_parquet(OUTPUT_PATH, index=False)

    return team_week


if __name__ == "__main__":
    build_team_week_stats()
