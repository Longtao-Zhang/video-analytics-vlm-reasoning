"""FastAPI inference service for the warehouse safety pipeline.

Exposes the end-to-end pipeline (video -> safety report) over HTTP.

Endpoints:
    GET  /health   -- liveness check (worked example)
    POST /analyze  -- upload a video, run the pipeline, return the safety report

Run locally (from the repo root):
    uvicorn app.main:app --reload --port 8000
    # then open http://localhost:8000/docs for the interactive Swagger UI

------------------------------------------------------------------------------
LEARNING EXERCISE
------------------------------------------------------------------------------
/health is fully implemented as a worked example. Implement /analyze yourself by
filling in the `# TODO(you)` block: this is the "deployment glue" that turns an
uploaded file into a pipeline run and a JSON response. The response model
(AnalyzeResponse) is given so you know the exact shape to return.

Note on `def` vs `async def`: /analyze is a *sync* `def` on purpose. run_pipeline
does heavy, blocking CPU work (YOLO); FastAPI runs sync endpoints in a worker
thread, so it won't block the event loop. An `async def` here would freeze the
whole server during inference.
------------------------------------------------------------------------------
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from src.utils import load_config
from src.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Load config once at import time so every request reuses it.
CONFIG_PATH = Path("configs/default.yaml")
config = load_config(CONFIG_PATH)

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(
    title="Warehouse Safety Video Intelligence API",
    description="Upload an operational video; get a structured safety report back.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Response models (the JSON shapes the API returns)
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    service: str


class AnalyzeResponse(BaseModel):
    video_name: Optional[str]
    overall_risk_level: str
    num_candidate_events: int
    num_confirmed_events: int
    num_dismissed_events: int
    report: dict[str, Any]  # the full safety_report.json payload


# ---------------------------------------------------------------------------
# GET /health  (WORKED EXAMPLE — study this, then write /analyze)
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe used by Docker / orchestrators to check the service is up."""
    return HealthResponse(status="ok", service="safety-video-intelligence")


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    video: UploadFile = File(..., description="Video file to analyze."),
    vlm_mode: str = Query("mock", pattern="^(mock|api)$", description="VLM mode."),
) -> AnalyzeResponse:
    """Run the full pipeline on an uploaded video and return the safety report.
    """
    logger.info(f"Received analyze request for video: {video.filename}, mode: {vlm_mode}")

    # 1. Validate the upload:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}.")
    
    # 2. Make a temp working dir
    work_dir = Path(tempfile.mkdtemp())

    try:
        # 3 Persist the upload to disk
        video_path = work_dir / (video.filename or "upload.mp4")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        # 4. Run the pipeline
        report = run_pipeline(video_path=video_path, config=config,
                              output_dir=work_dir, vlm_mode=vlm_mode)
        
        # 5. Return reports
        logger.info(f"Video processed successfully: {video.filename}")
        return AnalyzeResponse(
            video_name=video.filename,
            overall_risk_level=report.get("overall_risk_level", "UNKNOWN"),
            num_candidate_events=report.get("num_candidate_events", 0),
            num_confirmed_events=report.get("num_confirmed_events", 0),
            num_dismissed_events=report.get("num_dismissed_events", 0),
            report=report,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info(f"Work directory cleaned up.")
    
