"""Streamlit dashboard for exploring NFL game predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.config import PROCESSED_DIR, MODELS_DIR
from src.feature_lists import LEAKY_NUMERIC_COLS


GAMES_FEATURES_PATH = PROCESSED_DIR / "games_with_features.parquet"
WIN_MODEL_PATH = MODELS_DIR / "win_model.pkl"
WIN_PREPROCESSOR_PATH = MODELS_DIR / "win_preprocessor.pkl"
MARGIN_MODEL_PATH = MODELS_DIR / "margin_model.pkl"
MARGIN_PREPROCESSOR_PATH = MODELS_DIR / "margin_preprocessor.pkl"


@st.cache_resource
def load_models():
    win_model = joblib.load(WIN_MODEL_PATH)
    win_preprocessor = joblib.load(WIN_PREPROCESSOR_PATH)
    margin_model = joblib.load(MARGIN_MODEL_PATH)
    margin_preprocessor = joblib.load(MARGIN_PREPROCESSOR_PATH)
    return win_model, win_preprocessor, margin_model, margin_preprocessor


@st.cache_data
def load_games() -> pd.DataFrame:
    df = pd.read_parquet(GAMES_FEATURES_PATH)
    date_col = "game_date" if "game_date" in df.columns else "gameday"
    if date_col in df.columns:
        df["game_date"] = pd.to_datetime(df[date_col])
    return df


def _get_feature_cols(df: pd.DataFrame, drop_cols: set[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = LEAKY_NUMERIC_COLS | drop_cols
    return [col for col in numeric_cols if col not in drop]


@st.cache_data
def load_games_with_predictions() -> pd.DataFrame:
    base_df = load_games()
    win_model, win_preprocessor, margin_model, margin_preprocessor = load_models()

    win_cols = _get_feature_cols(base_df, {"home_win"})
    win_data = win_preprocessor.transform(base_df[win_cols])
    pred_home_win_proba = win_model.predict_proba(win_data)[:, 1]
    pred_home_win = (pred_home_win_proba >= 0.5).astype(int)

    margin_cols = _get_feature_cols(base_df, {"point_diff"})
    margin_data = margin_preprocessor.transform(base_df[margin_cols])
    pred_point_diff = margin_model.predict(margin_data)

    df = base_df.copy()
    df["pred_home_win_proba"] = pred_home_win_proba
    df["pred_home_win"] = pred_home_win
    df["pred_point_diff"] = pred_point_diff
    df["ml_hit"] = (df["pred_home_win"] == df["home_win"]).astype(int)

    spread_pred_margin = df["pred_point_diff"] - df["spread_line"]
    spread_actual_margin = df["point_diff"] - df["spread_line"]
    spread_hit_bool = np.sign(spread_pred_margin) == np.sign(spread_actual_margin)
    spread_hit = spread_hit_bool.astype(float)
    spread_hit[spread_actual_margin == 0] = np.nan  # pushes ignored
    df["spread_hit"] = spread_hit

    return df


def compute_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("season")
        .agg(
            games=("game_id", "count"),
            ml_hit_rate=("ml_hit", "mean"),
            spread_hit_rate=("spread_hit", "mean"),
        )
        .reset_index()
        .sort_values("season", ascending=False)
    )
    return summary


def format_matchup(row: pd.Series) -> str:
    return f"Week {int(row['week'])}: {row['away_team']} @ {row['home_team']}"


def main() -> None:
    st.set_page_config(page_title="NFL Game Prediction Explorer", layout="wide")
    st.title("NFL Game Prediction Explorer")

    games_df = load_games_with_predictions()
    games_eval = games_df[games_df["season"] >= 2019].copy()
    if games_eval.empty:
        st.error("No evaluation seasons available (2019+).")
        return

    season_summary = compute_season_summary(games_eval)
    seasons = sorted(games_eval["season"].unique(), reverse=True)

    with st.sidebar:
        st.header("Game Selector")
        season = st.selectbox("Season", seasons, index=0)
        season_weeks = sorted(games_eval[games_eval["season"] == season]["week"].unique())
        week = st.selectbox("Week", season_weeks, index=0)

        week_games = games_eval[(games_eval["season"] == season) & (games_eval["week"] == week)].copy()
        if week_games.empty:
            st.warning("No games found for that season/week.")
            return

        week_ml_rate = week_games["ml_hit"].mean()
        week_spread_rate = week_games["spread_hit"].mean()
        if not np.isnan(week_ml_rate):
            st.metric("Week ML Hit Rate", f"{week_ml_rate * 100:.1f}%")
        if not np.isnan(week_spread_rate):
            st.metric("Week Spread Hit Rate", f"{week_spread_rate * 100:.1f}%")

        options = week_games.apply(format_matchup, axis=1).tolist()
        selection = st.selectbox("Matchup", options)
        selected_row = week_games.iloc[options.index(selection)]

    summary_tab, game_tab = st.tabs([f"Season Summary ({seasons[-1]}+)", "Game Explorer"])

    with summary_tab:
        st.subheader("Season Hit Rates")
        summary_display = season_summary.copy()
        summary_display["ml_hit_rate"] = (summary_display["ml_hit_rate"] * 100).round(1)
        summary_display["spread_hit_rate"] = (summary_display["spread_hit_rate"] * 100).round(1)
        summary_display.rename(
            columns={
                "ml_hit_rate": "ML Hit %",
                "spread_hit_rate": "Spread Hit %",
            },
            inplace=True,
        )
        st.dataframe(summary_display, use_container_width=True)

    with game_tab:
        st.subheader(selection)

        col1, col2 = st.columns(2)
        col1.metric("Predicted Home Win Prob.", f"{selected_row['pred_home_win_proba'] * 100:.1f}%")
        col2.metric("Predicted Point Diff (home-away)", f"{selected_row['pred_point_diff']:.1f}")

        if not pd.isna(selected_row.get("home_score")):
            st.write(
                f"**Final Score:** {selected_row['away_team']} {int(selected_row['away_score'])} - "
                f"{selected_row['home_team']} {int(selected_row['home_score'])}"
            )

        feature_cols = [
            "spread_line",
            "total_line",
            "div_game",
            "rest_diff",
            "home_off_epa_per_play_to_date",
            "away_off_epa_per_play_to_date",
            "home_def_epa_per_play_to_date",
            "away_def_epa_per_play_to_date",
            "pred_home_win_proba",
            "pred_point_diff",
        ]
        feature_cols = [col for col in feature_cols if col in selected_row.index]

        st.markdown("### Key Features")
        st.table(selected_row[feature_cols].to_frame(name="Value"))

        st.markdown("### Week Overview")
        display_cols = [
            "away_team",
            "home_team",
            "spread_line",
            "total_line",
            "pred_home_win_proba",
            "pred_point_diff",
            "home_win",
            "home_score",
            "away_score",
        ]
        display_cols = [col for col in display_cols if col in week_games.columns]
        st.dataframe(week_games[display_cols].reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
