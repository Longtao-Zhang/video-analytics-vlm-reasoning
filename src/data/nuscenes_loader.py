"""Load a nuScenes scene into a flat, framework-agnostic structure.

This is the data foundation for the driving pivot. It wraps the official
``nuscenes-devkit`` and turns one scene into an ordered list of *frames*, where
each frame bundles everything the downstream stages need:

    - the CAM_FRONT image path           (perception + VLM input)
    - the ego pose in the global frame   (coordinate transforms -> ego frame)
    - the camera intrinsic               (projecting 3D boxes onto the image)
    - the ground-truth agents            (3D boxes + velocity + a stable
                                          ``instance_token`` = GT track id)

The ``instance_token`` is the key asset: the same physical object keeps the
same token across samples, so it gives us *free* ground-truth tracks and
*free* future trajectories -- which is what makes real prediction evaluation
possible later (ADE/FDE vs. GT future).

Run (dumps one scene to JSON for inspection):
    python -m src.data.nuscenes_loader --scene 0 --config configs/default.yaml \\
        --output examples/outputs/nuscenes/scene_0

Most of this file is plumbing and is provided complete. Two functions are left
as TODO(you) on purpose -- they are where you learn the nuScenes token graph:
    1. iter_scene_samples  -> walk the linked list of samples
    2. extract_gt_agents   -> pull per-object global state + instance_token
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.utils import load_config, save_json

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data structures (deliberately plain so they serialize straight to JSON).
# --------------------------------------------------------------------------- #
@dataclass
class EgoPose:
    """Ego vehicle pose in the global frame at one timestamp."""
    translation: list[float]          # [x, y, z] in metres, global frame
    rotation: list[float]             # quaternion [w, x, y, z], global frame
    timestamp_us: int                 # microseconds (nuScenes native unit)


@dataclass
class GTAgent:
    """A ground-truth annotated object at one sample (keyframe)."""
    instance_token: str               # STABLE id of this object across samples
    category: str                     # coarse class (person/car/...) via category_map
    raw_category: str                 # original nuScenes category_name
    translation: list[float]          # [x, y, z] global
    size: list[float]                 # [w, l, h]
    rotation: list[float]             # quaternion [w, x, y, z] global
    velocity: list[float]             # [vx, vy, vz] global (nan -> 0.0)
    num_lidar_pts: int                # 0 => effectively invisible to lidar


@dataclass
class SceneFrame:
    """One keyframe: camera image + ego pose + GT agents."""
    sample_token: str
    timestamp_s: float
    image_path: str
    width: int
    height: int
    ego_pose: EgoPose
    cam_intrinsic: list[list[float]]
    gt_agents: list[GTAgent] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Devkit access (provided complete).
# --------------------------------------------------------------------------- #
def load_nuscenes(config: dict[str, Any]):
    """Instantiate the NuScenes devkit from the ``dataset`` config block.

    Lazy import so the rest of the repo does not hard-depend on the devkit.
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "nuscenes-devkit is not installed. Run `pip install nuscenes-devkit` "
            "and download the v1.0-mini split (see docs/nuscenes_setup.md)."
        ) from exc

    ds = config["dataset"]
    dataroot = Path(ds["dataroot"])
    if not dataroot.exists():
        raise FileNotFoundError(
            f"nuScenes dataroot not found: {dataroot} (see docs/nuscenes_setup.md)"
        )

    logger.info("Loading nuScenes %s from %s", ds["version"], dataroot)
    return NuScenes(version=ds["version"], dataroot=str(dataroot), verbose=False)


def resolve_scene_token(nusc, scene_id: str) -> str:
    """Accept a scene index ("0"), a scene name ("scene-0061"), or a raw token."""
    # Integer index into nusc.scene
    if scene_id.isdigit():
        idx = int(scene_id)
        if not (0 <= idx < len(nusc.scene)):
            raise IndexError(f"Scene index {idx} out of range (have {len(nusc.scene)}).")
        return nusc.scene[idx]["token"]

    # Scene name, e.g. "scene-0061"
    for scene in nusc.scene:
        if scene["name"] == scene_id:
            return scene["token"]

    # Assume it is already a token
    return scene_id


def get_camera_frame(nusc, sample: dict[str, Any], camera: str) -> tuple[SceneFrame, str]:
    """Build a SceneFrame (without GT agents yet) from one sample.

    Returns ``(scene_frame, cam_sd_token)``; the sample_data token is handy if
    you later want the GT boxes already transformed into the *camera* frame via
    ``nusc.get_sample_data(cam_sd_token)``.
    """
    cam_sd_token = sample["data"][camera]
    sd = nusc.get("sample_data", cam_sd_token)
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    frame = SceneFrame(
        sample_token=sample["token"],
        timestamp_s=sample["timestamp"] / 1e6,
        image_path=str(Path(nusc.dataroot) / sd["filename"]),
        width=int(sd["width"]),
        height=int(sd["height"]),
        ego_pose=EgoPose(
            translation=[float(x) for x in ego["translation"]],
            rotation=[float(x) for x in ego["rotation"]],
            timestamp_us=int(ego["timestamp"]),
        ),
        cam_intrinsic=[[float(v) for v in row] for row in cs["camera_intrinsic"]],
    )
    return frame, cam_sd_token


# --------------------------------------------------------------------------- #
# TODO(you) #1 -- walk the sample linked list.
# --------------------------------------------------------------------------- #
def iter_scene_samples(nusc, scene_token: str) -> Iterator[dict[str, Any]]:
    """Yield a scene's samples (keyframes) in time order.

    nuScenes stores samples as a doubly linked list. The scene record gives you
    ``first_sample_token``; each sample record has a ``next`` token that is ""
    at the end of the scene.

    TODO(you):
        scene = nusc.get("scene", scene_token)
        token = scene["first_sample_token"]
        while token:                       # "" is falsy -> loop ends
            sample = nusc.get("sample", token)
            yield sample
            token = sample["next"]

    Hint: ``nusc.get("sample", token)`` returns a dict with keys
    {token, timestamp, prev, next, scene_token, data, anns}. ``data`` maps each
    sensor channel -> a sample_data token; ``anns`` is the list of annotation
    tokens for this keyframe.
    """
    raise NotImplementedError("Implement iter_scene_samples (see docstring).")


# --------------------------------------------------------------------------- #
# TODO(you) #2 -- extract ground-truth agents for one sample.
# --------------------------------------------------------------------------- #
def extract_gt_agents(
    nusc,
    sample: dict[str, Any],
    category_map: dict[str, str],
) -> list[GTAgent]:
    """Turn a sample's annotation tokens into GTAgent records.

    For each ``ann_token`` in ``sample["anns"]``:
        ann = nusc.get("sample_annotation", ann_token)
            -> {instance_token, category_name, translation, size, rotation,
                num_lidar_pts, ...}  (translation/rotation are GLOBAL frame)
        vel = nusc.box_velocity(ann_token)   # np.array([vx, vy, vz]) global;
                                             # can be NaN for first/last frame.

    TODO(you):
        1. Map ann["category_name"] -> coarse class with ``category_map``.
           Skip (or label "other") categories that are not in the map.
        2. Replace NaN velocity components with 0.0
           (e.g. ``float(v) if not math.isnan(v) else 0.0``).
        3. Build and collect a GTAgent. ``instance_token`` is what links the
           same object across samples -- do not drop it; it is your GT track id.

    Returns the list of GTAgent for this keyframe.
    """
    raise NotImplementedError("Implement extract_gt_agents (see docstring).")


# --------------------------------------------------------------------------- #
# Orchestration (provided complete -- relies on the two functions above).
# --------------------------------------------------------------------------- #
def load_scene_frames(
    nusc,
    scene_token: str,
    camera: str,
    category_map: dict[str, str],
) -> list[SceneFrame]:
    """Load every keyframe of a scene as a list of SceneFrame (with GT agents)."""
    frames: list[SceneFrame] = []
    for sample in iter_scene_samples(nusc, scene_token):
        frame, _ = get_camera_frame(nusc, sample, camera)
        frame.gt_agents = extract_gt_agents(nusc, sample, category_map)
        frames.append(frame)
    logger.info("Loaded %d frames for scene %s", len(frames), scene_token)
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a nuScenes scene to JSON.")
    parser.add_argument("--scene", type=str, default="0",
                        help="Scene index, name (scene-0061), or token.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for scene_frames.json.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    ds = config["dataset"]

    nusc = load_nuscenes(config)
    scene_token = resolve_scene_token(nusc, args.scene)
    scene = nusc.get("scene", scene_token)

    frames = load_scene_frames(
        nusc,
        scene_token=scene_token,
        camera=ds["camera"],
        category_map=ds["category_map"],
    )

    result = {
        "dataset": ds["name"],
        "version": ds["version"],
        "scene_token": scene_token,
        "scene_name": scene["name"],
        "camera": ds["camera"],
        "num_frames": len(frames),
        "frames": [asdict(f) for f in frames],
    }
    out_path = args.output / "scene_frames.json"
    save_json(result, out_path)
    logger.info("Saved %d frames to %s", len(frames), out_path)


if __name__ == "__main__":
    main()
