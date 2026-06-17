"""Ego-relative geometry & kinematics  (HANDS-ON MODULE 2 -- you implement the cores).

This module turns *global-frame* nuScenes annotations into *ego-relative*
quantities a driving agent actually reasons about:

    global position  --(ego pose)-->  ego-frame position (x fwd, y left)
    consecutive positions  -->  velocity, heading
    relative position + velocity  -->  time-to-collision (TTC)
    ego-frame position  -->  zone (ahead/left/right/behind), in-ego-path?

Frame convention (nuScenes): the EGO frame has **x forward, y left, z up**.
Do not take this on faith -- the Week-1 checkpoint is to *verify* it by
reprojecting GT boxes: a car directly in front must come out with large +x and
small |y| after ``global_to_ego``.

What is provided: function signatures, docstrings with the exact formulas, I/O,
one fully-worked helper (``normalize_quaternion``), and a CLI that becomes a
transform sanity-check once you fill in the TODOs:

    python -m src.geometry --scene-frames examples/outputs/nuscenes/scene_0/scene_frames.json \\
        --config configs/default.yaml --output examples/outputs/nuscenes/scene_0

What you implement: every function marked TODO(you).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Worked helper (provided) -- shows the expected style.
# --------------------------------------------------------------------------- #
def normalize_quaternion(q: list[float] | np.ndarray) -> np.ndarray:
    """Return the unit quaternion [w, x, y, z]. (Provided as a worked example.)"""
    arr = np.asarray(q, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Cannot normalize a zero quaternion.")
    return arr / norm


# --------------------------------------------------------------------------- #
# TODO(you) -- rotations & frame transforms (the heart of 座標変換).
# --------------------------------------------------------------------------- #
def quaternion_to_rotation_matrix(q: list[float] | np.ndarray) -> np.ndarray:
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix R (body->global).

    TODO(you). For a unit quaternion (w, x, y, z):

        R = [[1-2(y^2+z^2),  2(xy-wz),    2(xz+wy)],
             [2(xy+wz),      1-2(x^2+z^2),2(yz-wx)],
             [2(xz-wy),      2(yz+wx),    1-2(x^2+y^2)]]

    Normalize first with ``normalize_quaternion``. Cross-check against
    ``pyquaternion.Quaternion(w, x, y, z).rotation_matrix`` (ships with the
    devkit) -- they must match.
    """
    raise NotImplementedError


def global_to_ego(points_global: np.ndarray, ego_pose: dict[str, Any]) -> np.ndarray:
    """Transform global-frame points (N,3) into the ego frame at this pose.

    TODO(you):
        R = quaternion_to_rotation_matrix(ego_pose["rotation"])   # ego -> global
        t = np.asarray(ego_pose["translation"])                   # ego origin in global
        # invert the rigid transform: p_ego = R^T @ (p_global - t)
        return (points_global - t) @ R          # note: (p - t) @ R == (R^T @ (p - t)^T)^T

    Accept points shaped (3,) or (N,3); return the same shape.
    """
    raise NotImplementedError


def ego_to_global(points_ego: np.ndarray, ego_pose: dict[str, Any]) -> np.ndarray:
    """Inverse of ``global_to_ego``: p_global = R @ p_ego + t.

    TODO(you). Useful for sanity checks (round-trip should be identity).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# TODO(you) -- kinematics from a trajectory.
# --------------------------------------------------------------------------- #
def estimate_velocity(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Finite-difference velocity for a trajectory.

    positions: (T, D) positions (D=2 or 3); timestamps: (T,) seconds.

    TODO(you): v[i] = (positions[i] - positions[i-1]) / (timestamps[i] - timestamps[i-1]).
    Decide how to handle i=0 (e.g. copy v[1], or zero). Return (T, D).
    Guard against dt == 0.
    """
    raise NotImplementedError


def heading_from_velocity(velocity_xy: np.ndarray) -> float:
    """Heading angle (radians) of a 2D velocity in the current frame.

    TODO(you): ``math.atan2(vy, vx)``. Define behaviour when speed ~ 0
    (heading is undefined; pick a convention, e.g. return 0.0).
    """
    raise NotImplementedError


def time_to_collision(
    rel_position_xy: np.ndarray,
    rel_velocity_xy: np.ndarray,
    ttc_max_seconds: float,
) -> float:
    """Line-of-sight TTC between ego and an agent, both in the ego frame.

    rel_position_xy = agent_xy - ego_xy   (ego is origin, so == agent_xy in ego frame)
    rel_velocity_xy = agent_v_xy - ego_v_xy

    TODO(you), closing-speed form:
        r = rel_position_xy ;  v = rel_velocity_xy
        dist = ||r||
        closing_speed = -(r . v) / dist     # positive when they approach
        if closing_speed <= 0:  return ttc_max_seconds   # not converging
        return min(dist / closing_speed, ttc_max_seconds)

    Guard dist == 0. Return a capped float (never inf) so it serializes cleanly.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# TODO(you) -- ego-relative spatial semantics (feeds the VLM scene state).
# --------------------------------------------------------------------------- #
def relative_zone(point_ego_xy: np.ndarray, front_fov_deg: float) -> str:
    """Classify an ego-frame point as 'ahead' | 'left' | 'right' | 'behind'.

    Ego frame: x forward, y left. TODO(you):
        angle = atan2(y, x)                       # 0 = straight ahead
        if |angle| <= front_fov_deg/2 -> 'ahead'
        elif |angle| >= 180 - front_fov_deg/2 -> 'behind'
        elif angle > 0 -> 'left' else 'right'
    (Work in degrees or radians consistently.)
    """
    raise NotImplementedError


def in_ego_path(point_ego_xy: np.ndarray, half_width_m: float) -> bool:
    """True if the agent sits inside the forward ego corridor.

    TODO(you): x_ego > 0 (ahead) AND |y_ego| <= half_width_m.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# CLI -- transform sanity check (works once the TODOs above are implemented).
# --------------------------------------------------------------------------- #
def _agent_ego_record(agent: dict[str, Any], ego_pose: dict[str, Any],
                      geom_cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute ego-frame geometry for one GT agent at one frame (per-frame only).

    Velocity/TTC need a trajectory across frames, so this single-frame view
    fills those once you wire up multi-frame tracks in scene_state.py (Week 2).
    """
    p_ego = global_to_ego(np.asarray(agent["translation"], dtype=float), ego_pose)
    xy = p_ego[:2]
    return {
        "instance_token": agent["instance_token"],
        "category": agent["category"],
        "ego_xy": [float(xy[0]), float(xy[1])],
        "range_m": float(np.linalg.norm(xy)),
        "zone": relative_zone(xy, float(geom_cfg["zone_front_fov_deg"])),
        "in_ego_path": in_ego_path(xy, float(geom_cfg["ego_path_half_width_m"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ego-frame geometry sanity check.")
    parser.add_argument("--scene-frames", type=Path, required=True,
                        help="scene_frames.json from src.data.nuscenes_loader")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    geom_cfg = config["geometry"]

    with args.scene_frames.open("r", encoding="utf-8") as f:
        scene = json.load(f)

    frames_out: list[dict[str, Any]] = []
    for frame in scene["frames"]:
        ego_pose = frame["ego_pose"]
        agents = [_agent_ego_record(a, ego_pose, geom_cfg) for a in frame["gt_agents"]]
        frames_out.append({
            "sample_token": frame["sample_token"],
            "timestamp_s": frame["timestamp_s"],
            "agents_ego": agents,
        })

    out = {"scene_name": scene.get("scene_name"), "num_frames": len(frames_out),
           "frames": frames_out}
    out_path = args.output / "ego_geometry.json"
    save_json(out, out_path)
    logger.info("Saved ego-frame geometry to %s", out_path)


if __name__ == "__main__":
    main()
