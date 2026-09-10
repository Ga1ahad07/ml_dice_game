from pathlib import Path
import joblib
import pandas as pd
from loguru import logger

from ml_dice_game.config import MODELS_DIR


class ModelPredictor:
    def __init__(self, model_path: Path = MODELS_DIR / "model.pkl"):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Cargando modelo desde {self.model_path}")
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index, name="PUNTAJE_pred")
