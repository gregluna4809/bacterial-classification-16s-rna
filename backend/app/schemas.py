from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sequence: str = Field(..., description="16S rRNA sequence using A, C, G, T, N.")


class PredictionItem(BaseModel):
    genus: str
    probability: float


class PredictResponse(BaseModel):
    predicted_genus: str
    confidence: float
    top_5: list[PredictionItem]
    sequence_length: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_loaded: bool
    metrics: dict | None
