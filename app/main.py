"""FastAPI inference service for the warehouse safety pipeline.

Exposes the end-to-end pipeline (video -> safety report) over HTTP.

Endpoints:
    GET  /health   -- liveness check
    POST /analyze  -- upload a video, run the pipeline, return the safety report

Run locally (from the repo root):
    uvicorn app.main:app --reload --port 8000
    # then open http://localhost:8000/docs for the interactive Swagger UI

Upload handling:
    - The client-supplied filename is never used as a path: the upload is saved
      under a random name inside a fresh temp dir (no path traversal).
    - Uploads larger than ``service.max_upload_mb`` are rejected with 413.
    - ``vlm_mode=api`` spends paid API quota, so it is only enabled when
      SERVICE_API_TOKEN is set, and requires ``Authorization: Bearer <token>``.
      ``mock`` mode stays open for demos.

/analyze is a sync ``def`` on purpose: run_pipeline does blocking CPU work
(YOLO), and FastAPI runs sync endpoints in a worker thread so the event loop
stays responsive.
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Optional

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
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
MAX_UPLOAD_BYTES = int(float(config.get("service", {}).get("max_upload_mb", 200)) * 1024 * 1024)
API_TOKEN_ENV = "SERVICE_API_TOKEN"
_COPY_CHUNK = 1024 * 1024

app = FastAPI(
    title="Warehouse Safety Video Intelligence API",
    description="Upload an operational video; get a structured safety report back.",
    version="0.2.0",
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
    num_unverified_events: int
    report: dict[str, Any]  # the full safety_report.json payload


class UploadTooLarge(Exception):
    """Raised when an upload exceeds MAX_UPLOAD_BYTES."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def display_name(filename: str | None, max_len: int = 100) -> str:
    """A log/response-safe version of the client filename (never used as a path)."""
    name = Path((filename or "").replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:max_len]
    return name or "upload"


def copy_limited(src: BinaryIO, dst: BinaryIO, max_bytes: int) -> int:
    """Stream ``src`` into ``dst``; raise UploadTooLarge past ``max_bytes``."""
    total = 0
    while chunk := src.read(_COPY_CHUNK):
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLarge
        dst.write(chunk)
    return total


def check_api_mode_allowed(authorization: str | None) -> None:
    """Gate the paid VLM API mode behind a bearer token."""
    expected = os.getenv(API_TOKEN_ENV)
    if not expected:
        raise HTTPException(
            status_code=403,
            detail=f"vlm_mode=api is disabled: set {API_TOKEN_ENV} on the server to enable it.",
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip(), expected):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")


# ---------------------------------------------------------------------------
# GET /health
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
    authorization: Optional[str] = Header(None, description="Bearer token (api mode only)."),
) -> AnalyzeResponse:
    """Run the full pipeline on an uploaded video and return the safety report."""
    name = display_name(video.filename)
    logger.info("Received analyze request: video=%r mode=%s", name, vlm_mode)

    if vlm_mode == "api":
        check_api_mode_allowed(authorization)

    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or '(none)'}.")

    work_dir = Path(tempfile.mkdtemp(prefix="safety_"))
    try:
        # Random server-side name: the client filename never touches the filesystem.
        video_path = work_dir / f"{uuid.uuid4().hex}{suffix}"
        try:
            with open(video_path, "wb") as f:
                copy_limited(video.file, f, MAX_UPLOAD_BYTES)
        except UploadTooLarge:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )

        report = run_pipeline(video_path=video_path, config=config,
                              output_dir=work_dir, vlm_mode=vlm_mode)

        # Do not leak server-side temp paths; show the client's (sanitized) name.
        report["video_name"] = name
        report.pop("video_path", None)

        logger.info("Video processed: %r overall_risk=%s", name, report.get("overall_risk_level"))
        return AnalyzeResponse(
            video_name=name,
            overall_risk_level=report.get("overall_risk_level", "UNKNOWN"),
            num_candidate_events=report.get("num_candidate_events", 0),
            num_confirmed_events=report.get("num_confirmed_events", 0),
            num_dismissed_events=report.get("num_dismissed_events", 0),
            num_unverified_events=report.get("num_unverified_events", 0),
            report=report,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
