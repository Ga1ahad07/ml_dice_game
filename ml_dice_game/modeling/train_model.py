from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_predict, cross_validate

from ml_dice_game.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    load_params,
)
from ml_dice_game.modeling.common import (
    SCORING,
    evaluate_predictions,
    make_cv,
    save_json,
    save_model,
    split_xy,
)
from ml_dice_game.plots import EvaluationReporter
from ml_dice_game.modeling.tracking import ExperimentTracker


class TrainModel(ABC):
    """Pipeline comun para entrenar un modelo de regresion."""

    def __init__(
        self,
        train_path: Path = PROCESSED_DATA_DIR / "train.csv",
        model_path: Path = MODELS_DIR / "base" / "model.pkl",
        metrics_path: Path = REPORTS_DIR / "metrics" / "model.json",
        params: dict[str, Any] | None = None,
    ) -> None:
        self.train_path = train_path
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.params = params if params is not None else load_params()

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre estable usado en los JSON y reportes."""

    @property
    @abstractmethod
    def train_params_key(self) -> str:
        """Clave del estimador dentro de params['train']."""

    @abstractmethod
    def build_estimator(self) -> ClassifierMixin:
        """Construye el estimador especifico de la clase hija."""

    def model_params(self) -> dict[str, Any]:
        """Combina parametros YAML y parametros comunes de sklearn."""
        return {
            **self.params["train"][self.train_params_key],
            "random_state": self.params["random_state"],
            "n_jobs": -1,
        }

    def run(self) -> dict[str, Any]:
        """Entrena y evalúa el modelo mediante validación cruzada OOF."""
        train_frame = pd.read_csv(self.train_path)
        X_train, y_train = split_xy(train_frame)

        estimator = self.build_estimator()
        cv = make_cv(
            self.params["split"]["cv_n_splits"],
            self.params["random_state"],
        )

        cv_result = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
        )

        oof_predictions = cross_val_predict(
            estimator,
            X_train,
            y_train,
            cv=cv,
            method="predict",
            n_jobs=-1,
        )

        oof_probabilities = cross_val_predict(
            estimator,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        oof_metrics = evaluate_predictions(
            y_train,
            oof_predictions,
            oof_probabilities,
        )

        reporter = EvaluationReporter(save=True)
        reporter.plot_confusion_matrix_from_predictions(
            y_train,
            oof_predictions,
            self.model_name,
        )
        reporter.plot_roc_curve_from_predictions(
            y_train,
            oof_probabilities,
            self.model_name,
        )

        estimator.fit(X_train, y_train)

        payload = self.build_metrics_payload(
            estimator,
            cv_result,
            oof_metrics,
        )

        mlflow_params = self.params.get("mlflow", {})
        experiment_name = mlflow_params.get(
            "experiment_name",
            "ml_dice_game",
        )
        registered_model_name = mlflow_params.get(
            "registered_model_name",
        )

        tracker = ExperimentTracker(experiment_name)
        with tracker.run(self.model_name):
            tracker.log_params(self.model_params())
            tracker.log_cv_results(cv_result)
            tracker.log_oof_metrics(oof_metrics)
            tracker.log_model(
                estimator,
                self.model_name,
                registered_model_name=registered_model_name,
            )

        save_model(
            estimator,
            X_train.columns.tolist(),
            self.model_path,
        )
        save_json(payload, self.metrics_path)

        return payload

    def build_metrics_payload(
        self,
        estimator: ClassifierMixin,
        cv_result: dict[str, Any],
        oof_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Construye métricas basadas exclusivamente en entrenamiento OOF."""
        return {
            "target": TARGET_COLUMN,
            "model": self.model_name,
            **{
                f"OOF_{key}": value
                for key, value in oof_metrics.items()
            },
            "cv_Accuracy_mean": float(
                cv_result["test_accuracy"].mean()
            ),
            "cv_Precision_mean": float(
                cv_result["test_precision"].mean()
            ),
            "cv_Recall_mean": float(
                cv_result["test_recall"].mean()
            ),
            "cv_F1_mean": float(
                cv_result["test_f1"].mean()
            ),
            "cv_ROC_AUC_mean": float(
                cv_result["test_roc_auc"].mean()
            ),
            "params": estimator.get_params(),
        }
