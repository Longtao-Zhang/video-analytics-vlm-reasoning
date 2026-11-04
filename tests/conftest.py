"""Shared fixtures. Tests run without torch/YOLO or network access."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config() -> dict:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


def make_frame(idx: int, t: float, near: bool, image_path: str = "") -> dict:
    """One sampled frame with a person and a car, either close or far apart."""
    car = [20, 20, 40, 40] if near else [3000, 2000, 3100, 2100]
    return {
        "sampled_frame_id": idx,
        "original_frame_id": idx * 6,
        "timestamp": t,
        "image_path": image_path or f"frame_{idx:04d}.jpg",
        "detections": [
            {"label": "person", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
            {"label": "car", "confidence": 0.9, "bbox_xyxy": car},
        ],
    }
