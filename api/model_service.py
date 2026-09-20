import os
from pathlib import Path

import pandas as pd

from ml_dice_game.config import FINAL_FEATURES_PATH, FINAL_MODEL_PATH
from ml_dice_game.modeling.predict import ModelPredictor


class ModelService:
    def __init__(self, model_path: Path | None = None, features_path: Path | None = None):
        self.model_path = Path(
            os.getenv("ML_DICE_MODEL_PATH", model_path or FINAL_MODEL_PATH))
        self.features_path = Path(
            os.getenv("ML_DICE_FEATURES_PATH",
                      features_path or FINAL_FEATURES_PATH)
        )
        self._predictor = None

    @property
    def is_ready(self) -> bool:
        return self.model_path.exists() and self.features_path.exists()

    @property
    def source(self) -> str:
        return str(self.model_path)

    def load(self) -> None:
        if not self.is_ready:
            raise FileNotFoundError(
                f"No se encontraron los artifacts del modelo: {self.model_path} y {self.features_path}"
            )
        self._predictor = ModelPredictor(self.model_path, self.features_path)

    def predict(self, values: dict) -> dict:
        if self._predictor is None:
            self.load()
        return self._predictor.predict(values)

    def predict_batch(self, rows: list[dict]) -> dict:
        if self._predictor is None:
            self.load()
        if not rows:
            return {"predictions": [], "probabilities": []}
        results = [self._predictor.predict(row) for row in rows]
        return {
            "predictions": [r["prediction"] for r in results],
            "probabilities": [r["probability"] for r in results],
        }
