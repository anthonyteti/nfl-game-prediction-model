"""Train a baseline regressor predicting home point differential."""

from __future__ import annotations

from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MODELS_DIR, PROCESSED_DIR, ensure_directories
from .feature_lists import LEAKY_NUMERIC_COLS


DATA_PATH = PROCESSED_DIR / "games_with_features.parquet"
MODEL_PATH = MODELS_DIR / "margin_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "margin_preprocessor.pkl"


def _train_val_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].between(2010, 2018)]
    val = df[df["season"].between(2019, 2020)]
    test = df[df["season"].between(2021, 2023)]
    return train, val, test


def _get_feature_columns(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = set(target_cols) | LEAKY_NUMERIC_COLS
    return [col for col in numeric_cols if col not in drop_cols]


def train_margin_model() -> None:
    ensure_directories()
    data = pd.read_parquet(DATA_PATH)

    target = "point_diff"
    feature_cols = _get_feature_columns(data, [target])

    train_df, val_df, _ = _train_val_test_split(data)
    X_train, y_train = train_df[feature_cols], train_df[target]
    X_val, y_val = val_df[feature_cols], val_df[target]

    preprocessor = SimpleImputer(strategy="median")
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_t, y_train)

    val_pred = model.predict(X_val_t)

    mae = mean_absolute_error(y_val, val_pred)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    r2 = r2_score(y_val, val_pred)

    print("Validation Metrics:")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R^2  : {r2:.3f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)


if __name__ == "__main__":
    train_margin_model()
