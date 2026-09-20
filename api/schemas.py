from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    C1A: int = Field(...)
    C1B: int = Field(...)
    C1C: int = Field(...)
    C2A: int = Field(...)
    C2B: int = Field(...)
    C2C: int = Field(...)
    C3A: int = Field(...)
    C3B: int = Field(...)
    C3C: int = Field(...)
    T1A: int = Field(...)
    T1B: int = Field(...)
    T1C: int = Field(...)
    T2A: int = Field(...)
    T2B: int = Field(...)
    T2C: int = Field(...)
    T3A: int = Field(...)
    T3B: int = Field(...)
    T3C: int = Field(...)
    RONDA: int = Field(...)
    TURNO: int = Field(...)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BatchPredictRequest(BaseModel):
    rows: list[PredictionRequest]

    model_config = ConfigDict(extra="forbid")


class BatchPredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    model_source: str
