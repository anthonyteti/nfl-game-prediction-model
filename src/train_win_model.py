"""Train a baseline classifier predicting home win probability."""

from __future__ import annotations

from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

from .config import MODELS_DIR, PROCESSED_DIR, ensure_directories
from .feature_lists import LEAKY_NUMERIC_COLS


DATA_PATH = PROCESSED_DIR / "games_with_features.parquet"
MODEL_PATH = MODELS_DIR / "win_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "win_preprocessor.pkl"


def _train_val_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].between(2010, 2018)]
    val = df[df["season"].between(2019, 2020)]
    test = df[df["season"].between(2021, 2023)]
    return train, val, test


def _get_feature_columns(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = set(target_cols) | LEAKY_NUMERIC_COLS
    return [col for col in numeric_cols if col not in drop_cols]


def train_win_model() -> None:
    ensure_directories()
    data = pd.read_parquet(DATA_PATH)

    target = "home_win"
    feature_cols = _get_feature_columns(data, [target])

    train_df, val_df, _ = _train_val_test_split(data)
    X_train, y_train = train_df[feature_cols], train_df[target]
    X_val, y_val = val_df[feature_cols], val_df[target]

    preprocessor = SimpleImputer(strategy="median")
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_t, y_train)

    val_probs = model.predict_proba(X_val_t)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    acc = accuracy_score(y_val, val_preds)
    ll = log_loss(y_val, val_probs)
    roc = roc_auc_score(y_val, val_probs)

    print("Validation Metrics:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Log Loss : {ll:.3f}")
    print(f"  ROC AUC  : {roc:.3f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)


if __name__ == "__main__":
    train_win_model()
