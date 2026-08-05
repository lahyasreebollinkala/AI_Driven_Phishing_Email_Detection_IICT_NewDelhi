# =============================================================================
# AI-Driven Phishing Email Detection Using NLP
# FastAPI REST API
# =============================================================================
"""
REST API exposing the phishing detection engine to external consumers.

Interactive Swagger/OpenAPI documentation is served automatically at
``/docs`` (and raw OpenAPI JSON at ``/openapi.json``).

Endpoints:
    GET  /                 Service information
    GET  /health           Health check (incl. model status)
    GET  /models           List of available models
    POST /predict          Classify a single email (JSON body)
    POST /predict/file     Classify an uploaded .txt / .eml file
    POST /predict/batch    Classify many emails (JSON list or CSV upload)

Run (from the project root):
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import APP_ICON, APP_SUBTITLE, APP_TITLE, APP_VERSION
from src.predict import PhishingDetector
from src.utils import setup_logger

logger = setup_logger("api")

# ---------------------------------------------------------------------------
# Shared detector instance (lazily loaded once)
# ---------------------------------------------------------------------------

_detector: PhishingDetector | None = None


def get_detector() -> PhishingDetector:
    """Return the shared PhishingDetector, initialising it on first use."""
    global _detector
    if _detector is None:
        _detector = PhishingDetector()
    return _detector


def reset_detector() -> None:
    """Drop the cached detector (used after retraining)."""
    global _detector
    _detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the detector so the first request is fast
    try:
        get_detector()
        logger.info("Detector pre-warmed on startup.")
    except Exception as e:  # noqa: BLE001
        logger.warning("Detector not available at startup: %s", e)
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    description=(
        f"{APP_ICON} {APP_SUBTITLE}\n\n"
        "Classifies emails as **Phishing** or **Legitimate** using an NLP "
        "pipeline (TF-IDF + metadata features) with a trained ensemble of "
        "machine-learning classifiers. Produces a confidence score, a 0-100 "
        "risk score, detected suspicious keywords, URLs, and an explanation."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Allow cross-origin requests (e.g., from a browser frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single-email prediction request body."""
    email_text: str = Field(..., description="Raw email text/body content.", min_length=1)
    subject: Optional[str] = Field(None, description="Optional email subject line.")
    model_name: Optional[str] = Field(None, description="Specific saved model to use (default: best model).")


class BatchPredictRequest(BaseModel):
    """Batch prediction request body."""
    emails: list[str] = Field(..., description="List of raw email texts.")
    subjects: Optional[list[str]] = Field(None, description="Optional parallel list of subjects.")


class ModelInfo(BaseModel):
    model_name: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"], summary="Service information")
def root() -> dict:
    """Return service metadata and a pointer to the API docs."""
    return {
        "service": APP_TITLE,
        "description": APP_SUBTITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", tags=["meta"], summary="Health check")
def health() -> dict:
    """Return service and model availability status."""
    detector = get_detector()
    model_name = (detector.model_info or {}).get("model_name") if detector.model_info else None
    return {
        "status": "healthy",
        "model_loaded": detector.is_loaded,
        "model_name": model_name or None,
    }


@app.get("/models", tags=["meta"], summary="List available models")
def list_models() -> dict:
    """List every classifier saved on disk that can be selected via model_name."""
    return {
        "models": PhishingDetector.available_models(),
        "default": "best_model",
    }


@app.post("/predict", tags=["predict"], summary="Classify a single email")
def predict(request: PredictRequest) -> dict:
    """Classify one email and return a detailed result."""
    detector = get_detector()
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts not available. Train the pipeline first.")

    try:
        result = detector.predict(request.email_text, subject=request.subject)
    except Exception as e:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@app.post("/predict/file", tags=["predict"], summary="Classify an uploaded email file")
async def predict_file(file: UploadFile = File(..., description=".txt or .eml email file")) -> dict:
    """Upload a .txt or .eml file and classify its contents."""
    detector = get_detector()
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts not available. Train the pipeline first.")

    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    result = detector.predict_from_text_bytes(content, filename=file.filename or "uploaded")
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/predict/batch", tags=["predict"], summary="Classify multiple emails")
def predict_batch(payload: BatchPredictRequest) -> dict:
    """Classify a list of emails in one request."""
    detector = get_detector()
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts not available. Train the pipeline first.")

    if not payload.emails:
        raise HTTPException(status_code=422, detail="'emails' list must not be empty.")

    results = detector.predict_batch(payload.emails, payload.subjects)
    return {"count": len(results), "results": results}


@app.post("/predict/csv", tags=["predict"], summary="Classify emails from a CSV upload")
async def predict_csv(
    file: UploadFile = File(..., description="CSV file with a text column."),
    text_column: str = Query("text_combined", description="Column containing email text."),
    subject_column: Optional[str] = Query(None, description="Optional column containing subjects."),
) -> dict:
    """Upload a CSV of emails and return verdicts appended per row."""
    detector = get_detector()
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts not available. Train the pipeline first.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    if text_column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}",
        )

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV contains no rows.")

    out = detector.predict_dataframe(
        df,
        text_col=text_column,
        subject_col=subject_column if subject_column and subject_column in df.columns else None,
    )

    # Return both full output (serializable) and summary
    return {
        "count": len(out),
        "results": out.fillna("").to_dict(orient="records"),
    }
