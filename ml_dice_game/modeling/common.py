from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

METRIC_NAMES = ("Accuracy", "Precision", "Recall", "F1", "ROC_AUC")
SCORING = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc",
}


def calculate_metrics(y_true, predictions, probabilities) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
    }


def save_confusion_matrix(y_true, predictions, path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, predictions), display_labels=[0, 1]).plot(
        ax=axis, colorbar=False, cmap="Blues"
    )
    axis.set_title(title)
    axis.grid(False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_roc_curve(y_true, probabilities, path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(false_positive_rate, true_positive_rate,
              label=f"ROC_AUC = {auc:.3f}")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


@dataclass
class TrainingResult:
    name: str
    estimator: object
    metrics: dict[str, float]
    oof_predictions: object
    oof_probabilities: object


class CrossValidatedTrainer:
    def __init__(self, name: str, estimator, cv):
        self.name = name
        self.estimator = estimator
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        from sklearn.model_selection import cross_val_predict, cross_validate

        scores = cross_validate(
            self.estimator,
            X,
            y,
            cv=self.cv,
            scoring={key: value for key, value in SCORING.items()},
            n_jobs=-1,
        )
        metrics = {
            metric: float(scores[f"test_{metric}"].mean()) for metric in METRIC_NAMES
        }
        oof_predictions = cross_val_predict(
            self.estimator, X, y, cv=self.cv, method="predict", n_jobs=-1
        )
        oof_probabilities = cross_val_predict(
            self.estimator, X, y, cv=self.cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        fitted_estimator = clone(self.estimator).fit(X, y)
        return TrainingResult(
            self.name, fitted_estimator, metrics, oof_predictions, oof_probabilities
        )
