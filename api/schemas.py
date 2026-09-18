# api/schemas.py
import json

from pydantic import BaseModel, ConfigDict, create_model
from ml_dice_game.config import MODELS_DIR

_feature_cols = json.load(open(MODELS_DIR / "model.features.json"))
_example_row = {column: 1.0 for column in _feature_cols}

# Genera dinamicamente un modelo Pydantic con un campo float por cada feature.
PredictRequest = create_model(
    "PredictRequest",
    __config__=ConfigDict(json_schema_extra={"examples": [_example_row]}),
    **{col: (float, ...) for col in _feature_cols},
)


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_source: str


class BatchPredictRequest(BaseModel):
    rows: list[PredictRequest]

    model_config = ConfigDict(
        json_schema_extra={"examples": [
            {"rows": [_example_row, _example_row]}]}
    )


class BatchPredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    model_source: str
