"""Evaluate the saved win probability model on the held-out test set."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .config import MODELS_DIR, PROCESSED_DIR
from .feature_lists import LEAKY_NUMERIC_COLS


DATA_PATH = PROCESSED_DIR / "games_with_features.parquet"
MODEL_PATH = MODELS_DIR / "win_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "win_preprocessor.pkl"


def evaluate_win_model() -> None:
    data = pd.read_parquet(DATA_PATH)
    test_df = data[data["season"].between(2021, 2023)]

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"home_win"} | LEAKY_NUMERIC_COLS
    feature_cols = [col for col in numeric_cols if col not in drop_cols]

    X_test = test_df[feature_cols]
    y_test = test_df["home_win"]

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X_test_t = preprocessor.transform(X_test)
    probs = model.predict_proba(X_test_t)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    roc = roc_auc_score(y_test, probs)

    print("Test Metrics:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Log Loss : {ll:.3f}")
    print(f"  ROC AUC  : {roc:.3f}")


if __name__ == "__main__":
    evaluate_win_model()
