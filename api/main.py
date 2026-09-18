from fastapi import FastAPI, HTTPException
import pandas as pd
from loguru import logger

from api.model_service import model_service
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(title="ml_dice_game API", version="0.1.0")


@app.on_event("startup")
def startup():
    model_service.load()


@app.get("/health")
def health():
    return {"status": "ok", "model_source": model_service.source}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        df = pd.DataFrame([request.model_dump()])
        result = model_service.predict(df)
        return PredictResponse(
            prediction=int(result["predictions"][0]),
            probability=float(result["probabilities"][0]),
            model_source=model_service.source,
        )
    except Exception as e:
        logger.exception("Error en /predict")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    validated_rows = [PredictRequest.model_validate(
        row) for row in request.rows]
    df = pd.DataFrame([row.model_dump() for row in validated_rows])
    result = model_service.predict(df)
    return BatchPredictResponse(
        predictions=result["predictions"],
        probabilities=result["probabilities"],
        model_source=model_service.source,
    )
