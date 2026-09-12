from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.model_selection import cross_validate

from ml_dice_game.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_params,
)
from ml_dice_game.modeling.common import (
    SCORING,
    evaluate_model,
    load_train_test,
    make_cv,
    save_json,
    save_model,
    split_xy,
)


class TrainModel(ABC):
    """Pipeline comun para entrenar un modelo de regresion."""

    def __init__(
        self,
        train_path: Path = PROCESSED_DATA_DIR / "train.csv",
        test_path: Path = PROCESSED_DATA_DIR / "test.csv",
        model_path: Path = MODELS_DIR / "base" / "model.pkl",
        metrics_path: Path = REPORTS_DIR / "metrics" / "model.json",
        params: dict[str, Any] | None = None,
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
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
    def build_estimator(self) -> RegressorMixin:
        """Construye el estimador especifico de la clase hija."""

    def model_params(self) -> dict[str, Any]:
        """Combina parametros YAML y parametros comunes de sklearn."""
        return {
            **self.params["train"][self.train_params_key],
            "random_state": self.params["random_state"],
            "n_jobs": -1,
        }

    def run(self) -> dict[str, Any]:
        """Ejecuta el flujo completo y guarda los artefactos del stage."""
        train_frame, test_frame = load_train_test(
            self.train_path,
            self.test_path,
        )
        X_train, y_train = split_xy(train_frame)
        X_test, y_test = split_xy(test_frame)

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

        estimator.fit(X_train, y_train)
        test_metrics = evaluate_model(estimator, X_test, y_test)
        payload = self.build_metrics_payload(
            estimator,
            cv_result,
            test_metrics,
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
        estimator: RegressorMixin,
        cv_result: dict[str, Any],
        test_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Construye el formato comun consumido por select_best_model."""
        return {
            "model": self.model_name,
            **test_metrics,
            "cv_R2_mean": float(cv_result["test_r2"].mean()),
            "cv_MAE_mean": float(-cv_result["test_mae"].mean()),
            "cv_RMSE_mean": float(-cv_result["test_rmse"].mean()),
            "params": estimator.get_params(),
        }
