from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_loader import ModelArtifacts, load_model_artifacts
from app.schemas import HealthResponse, MetadataResponse, PredictRequest, PredictResponse
from app.sequence_utils import validate_sequence


model_artifacts: ModelArtifacts | None = None
startup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_artifacts, startup_error
    try:
        model_artifacts = load_model_artifacts()
        startup_error = None
    except Exception as exc:
        model_artifacts = None
        startup_error = str(exc)
    yield


app = FastAPI(
    title="16S rRNA Bacterial Genus Classifier",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_artifacts is not None else f"degraded: {startup_error}",
        model_loaded=model_artifacts is not None,
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return MetadataResponse(
        model_loaded=model_artifacts is not None,
        metrics=model_artifacts.metrics if model_artifacts is not None else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if model_artifacts is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts are not loaded: {startup_error}",
        )

    try:
        sequence = validate_sequence(request.sequence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(**model_artifacts.predict(sequence))
