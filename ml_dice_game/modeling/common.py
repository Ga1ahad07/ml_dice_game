from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_dice_game.config import RANDOM_STATE, TARGET_COLUMN


SCORING = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}


def load_train_test(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN]


def make_cv(cv_n_splits: int, random_state: int = RANDOM_STATE) -> KFold:
    return KFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)


def evaluate_model(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "R2": float(r2_score(y_test, predictions)),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2,
                    ensure_ascii=False), encoding="utf-8")


def save_model(model, feature_columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    save_json(feature_columns, path.with_suffix(".features.json"))


def load_model(path: Path):
    return joblib.load(path)
