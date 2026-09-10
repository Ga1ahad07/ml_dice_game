# api/schemas.py
import json
from pydantic import create_model, BaseModel
from ml_dice_game.config import MODELS_DIR

_feature_cols = json.load(open(MODELS_DIR / "model.features.json"))

# Genera dinámicamente un modelo Pydantic con un campo float por cada feature
PredictRequest = create_model(
    "PredictRequest", **{col: (float, ...) for col in _feature_cols}
)


class PredictResponse(BaseModel):
    prediction: float
    model_source: str


class BatchPredictRequest(BaseModel):
    rows: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: list[float]
    model_source: str
