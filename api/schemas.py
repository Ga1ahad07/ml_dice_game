# api/schemas.py
import json

from pydantic import BaseModel, create_model
from ml_dice_game.config import MODELS_DIR

_feature_cols = json.load(open(MODELS_DIR / "model.features.json"))

# Genera dinamicamente un modelo Pydantic con un campo float por cada feature.
PredictRequest = create_model(
    "PredictRequest", **{col: (float, ...) for col in _feature_cols}
)


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_source: str


class BatchPredictRequest(BaseModel):
    # La clase individual se crea dinamicamente y no puede usarse como tipo
    # estatico en Pylance. Pydantic valida los nombres y valores en runtime.
    rows: list[dict[str, float]]


class BatchPredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    model_source: str
