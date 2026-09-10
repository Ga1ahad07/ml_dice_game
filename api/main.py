from fastapi import FastAPI, HTTPException
import pandas as pd
from loguru import logger

from api.model_service import model_service
from api.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse

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
        df = pd.DataFrame([request.dict()])
        pred = model_service.predict(df)[0]
        return PredictResponse(prediction=pred, model_source=model_service.source)
    except Exception as e:
        logger.exception("Error en /predict")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    df = pd.DataFrame([row.dict() for row in request.rows])
    preds = model_service.predict(df)
    return BatchPredictResponse(predictions=preds, model_source=model_service.source)
