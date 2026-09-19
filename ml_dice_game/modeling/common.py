from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)

from ml_dice_game.config import RANDOM_STATE, TARGET_COLUMN


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def load_train(train_path: Path) -> pd.DataFrame:
    """Carga únicamente datos de entrenamiento."""
    return pd.read_csv(train_path)


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN]


def make_cv(cv_n_splits: int, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=cv_n_splits,
        shuffle=True,
        random_state=random_state,
    )


def evaluate_predictions(
    y_true: pd.Series,
    predictions,
    probabilities,
) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
    }


def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return evaluate_predictions(y_test, predictions, probabilities)


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
