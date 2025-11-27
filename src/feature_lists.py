"""Shared feature list utilities for modeling."""

LEAKY_NUMERIC_COLS = {
    "home_score",
    "away_score",
    "result",
    "total",
    "point_diff",
    "home_win",
}


__all__ = ["LEAKY_NUMERIC_COLS"]
