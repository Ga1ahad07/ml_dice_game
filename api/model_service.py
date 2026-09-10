import mlflow
import pandas as pd
from loguru import logger

from ml_dice_game.config import MODELS_DIR, load_params
from ml_dice_game.modeling.predict import ModelPredictor


class ModelService:
    """Resuelve el modelo desde MLflow Model Registry (stage Production);
    si falla, cae al .pkl local versionado por DVC."""

    def __init__(self):
        params = load_params()
        self.registered_model_name = params["mlflow"]["registered_model_name"]
        self._model = None
        self._source = None

    def load(self):
        try:
            uri = f"models:/{self.registered_model_name}/Production"
            self._model = mlflow.pyfunc.load_model(uri)
            self._source = "mlflow_registry"
            logger.success(f"Modelo cargado desde MLflow: {uri}")
        except Exception as e:
            logger.warning(
                f"No se pudo cargar desde MLflow ({e}). Usando fallback local.")
            self._model = ModelPredictor(
                model_path=MODELS_DIR / "model.pkl").model
            self._source = "local_pkl"
        return self

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def source(self) -> str:
        return self._source or "not_loaded"

    def predict(self, df: pd.DataFrame) -> list[float]:
        preds = self.model.predict(df)
        return [float(p) for p in preds]


model_service = ModelService()  # singleton a nivel de módulo
