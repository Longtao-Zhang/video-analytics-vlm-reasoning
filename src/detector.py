from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.utils import load_config, save_json