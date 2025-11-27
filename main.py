"""Convenience runner for the core data pipeline."""

from src import data_ingestion, team_week_stats, team_week_features, game_features


def run_pipeline() -> None:
    """Execute ingestion through game feature generation."""
    data_ingestion.save_raw_parquet()
    team_week_stats.build_team_week_stats()
    team_week_features.build_team_week_features()
    game_features.build_game_features()


if __name__ == "__main__":
    run_pipeline()
