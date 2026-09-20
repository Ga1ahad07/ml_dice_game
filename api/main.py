from fastapi import FastAPI, HTTPException
import pandas as pd

from api.model_service import ModelService
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse, BatchPredictRequest, BatchPredictResponse

app = FastAPI(title="ML Dice Game API", version="1.0.0")
service = ModelService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if service.is_ready else "model_not_ready",
        model_loaded=service.is_ready,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = service.predict(request.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    rows = [row.model_dump() for row in request.rows]
    try:
        result = service.predict_batch(rows)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return BatchPredictResponse(
        predictions=result["predictions"],
        probabilities=result["probabilities"],
        model_source=service.source,
    )
