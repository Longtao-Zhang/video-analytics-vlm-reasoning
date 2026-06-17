# nuScenes mini — setup & Week-1 tasks

The driving pivot is built on the **nuScenes mini** split (10 scenes, ~4 GB).
It ships real 3D boxes, ego poses, and per-object `instance_token`s, which is
what lets us do *real* prediction evaluation later (predicted vs. GT future).

## 1. Install dependencies

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # now includes nuscenes-devkit, scipy, pydantic
```

## 2. Download the mini split

1. Make a free account at <https://www.nuscenes.org/nuscenes> → **Download**.
2. Grab **Full dataset (v1.0) → Mini** = `v1.0-mini.tgz` (~4 GB).
3. Extract it so the repo sees this layout:

```
data/nuscenes/
  maps/
  samples/          # keyframe sensor data (images live here)
  sweeps/           # intermediate sensor data
  v1.0-mini/        # the JSON tables (sample.json, ego_pose.json, ...)
```

`dataroot` in [configs/default.yaml](../configs/default.yaml) (`dataset.dataroot`)
must point at `data/nuscenes`. Add `data/` to `.gitignore` — never commit the dataset.

## 3. Smoke-test the devkit

```python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=True)
print(len(nusc.scene), "scenes")           # -> 10
s = nusc.scene[0]
print(s["name"], s["nbr_samples"])         # e.g. scene-0061 39
```

If that prints without error, the data is in place.

## 4. Week-1 implementation tasks (you write the cores)

### a) `src/data/nuscenes_loader.py` — 2 TODOs
- `iter_scene_samples` — walk the sample linked list (`first_sample_token` → `next`).
- `extract_gt_agents` — map `category_name` → coarse class, zero out NaN velocity,
  keep `instance_token` (your GT track id).

**Verify:**
```powershell
python -m src.data.nuscenes_loader --scene 0 --config configs/default.yaml `
  --output examples/outputs/nuscenes/scene_0
```
Open `scene_frames.json`: `num_frames` should equal the scene's `nbr_samples`,
and each frame should list `gt_agents` with `instance_token`s that **repeat
across frames** (same object, same token).

### b) `src/geometry.py` — transform TODOs
- `quaternion_to_rotation_matrix`, `global_to_ego` (and `ego_to_global` for the
  round-trip check), plus `relative_zone` / `in_ego_path`.

**Verify (this is the real checkpoint):**
```powershell
python -m src.geometry --scene-frames examples/outputs/nuscenes/scene_0/scene_frames.json `
  --config configs/default.yaml --output examples/outputs/nuscenes/scene_0
```
Open `ego_geometry.json` and sanity-check the ego frame (x forward, y left):
- A car you can see directly **ahead** in the image → large **+x**, small **|y|**, `zone: "ahead"`.
- Round-trip: `ego_to_global(global_to_ego(p)) ≈ p`.
- Optional stronger check: project a GT 3D box back onto the CAM_FRONT image with
  the camera intrinsic and confirm it lands on the right object.

### c) Run YOLO on nuScenes images (reuse, no new code)
The existing `src/detector.py` already takes a `sampled_frames.json`-shaped file.
Quick path: build a tiny list of `{path: <image_path>}` from `scene_frames.json`
and feed `run_detection_on_image` per frame, or point the detector at the image
folder. Goal for Week 1 is just: **detections render on nuScenes CAM_FRONT images.**

## What "done with Week 1" looks like
- `scene_frames.json` loads a full scene with stable `instance_token`s.
- `ego_geometry.json` shows correct ahead/left/right zones (transforms verified).
- YOLO boxes visualized on at least one nuScenes scene.

Then Week 2: `src/tracker.py` (Module 1) + finish `src/geometry.py` kinematics
(velocity / heading / TTC) + assemble `src/scene_state.py`.
