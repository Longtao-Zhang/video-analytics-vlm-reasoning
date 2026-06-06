# Warehouse Safety Video Intelligence (VLM-assisted)

> Turns raw warehouse / operational video into structured safety events,
> interpretable risk levels, and a business-readable report — so safety teams can
> catch near-miss conditions without manually reviewing hours of footage.

## Problem → Solution

Manual CCTV review doesn't scale and misses near-miss events. This pipeline
detects safety-relevant visual patterns (e.g. **person–vehicle proximity**),
scores their risk with transparent rules, and uses a **VLM to verify and explain**
each one in plain language for non-technical stakeholders.

```text
video → frame sampling → YOLO detection → temporal aggregation
      → rule-based safety events → VLM reasoning → safety report (JSON + Markdown)
```

## Key design

- **Hybrid rule → VLM — *rules propose, the VLM verifies*.** Deterministic CV
  rules generate auditable candidate events with evidence frames; the VLM
  explains, verifies (`verified: true/false`), and contextualizes them rather than
  making ungrounded calls.
- **Interpretable risk.** Each event gets a `severity_prior` from a transparent
  weighted score (proximity, duration, confidence, crowding) with a stored
  `score_breakdown` — *"why is this high risk?"* is always answerable.
- **Max-verified-risk rollup.** A video's overall risk is the highest VLM risk
  among events it did **not** reject; VLM-dismissed false positives are demoted but
  kept for audit. (See [FP-001](docs/failure_cases.md) — the VLM correctly
  overturned a 2D-projection proximity false positive.)
- **Mock + API VLM.** `mock` runs with no key or network (clone-and-run demos);
  `api` uses Alibaba Cloud Model Studio (DashScope) Qwen-VL, degrading gracefully
  to mock on error.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the whole pipeline with one command (mock VLM — no API key needed):

```bash
python -m src.pipeline \
  --video examples/input_videos/berlin_tower_3840_2160_30fps.mp4 \
  --config configs/default.yaml \
  --output examples/outputs/demo/berlin_tower \
  --vlm-mode mock
```

This writes all intermediate JSON plus the final **`safety_report.md`** /
`safety_report.json` to the output directory.

To use the real VLM, set your key and switch modes (endpoint/model are configured
under `vlm.api` in [configs/default.yaml](configs/default.yaml)):

```bash
export DASHSCOPE_API_KEY=...     # or put it in a local .env file
python -m src.pipeline --video <v> --config configs/default.yaml --output <dir> --vlm-mode api
```

<details>
<summary>Run individual stages</summary>

```bash
python -m src.frame_sampler       --video <v> --config configs/default.yaml --output <dir>
python -m src.detector            --sampled-frames-json <dir>/sampled_frames.json --config configs/default.yaml --output <dir>
python -m src.temporal_aggregator --detections-json <dir>/detections.json --config configs/default.yaml --output <dir>
python -m src.event_generator     --detections-json <dir>/detections.json --temporal-segments-json <dir>/temporal_segments.json --config configs/default.yaml --output <dir>
python -m src.vlm_reasoner        --safety-events-json <dir>/safety_event_candidates.json --config configs/default.yaml --output <dir> --mode mock
python -m src.report_generator    --vlm-reasoning-json <dir>/vlm_event_reasoning.json --config configs/default.yaml --output <dir>
```
</details>

## Deployment (HTTP service + Docker)

The pipeline is also exposed as a FastAPI service ([app/main.py](app/main.py)) —
`POST /analyze` takes a video upload and returns the safety report as JSON.

```bash
# Local
uvicorn app.main:app --port 8000           # interactive docs at http://localhost:8000/docs

# Docker
docker build -t safety-video .
docker run --rm -p 8000:8000 safety-video                  # mock VLM (no key)
docker run --rm -p 8000:8000 --env-file .env safety-video  # real VLM (injects DASHSCOPE_API_KEY)
# or simply: docker compose up --build
```

Call it:

```bash
curl localhost:8000/health
curl -F "video=@examples/input_videos/berlin_tower_3840_2160_30fps.mp4" \
     "localhost:8000/analyze?vlm_mode=mock"     # or vlm_mode=api
```

`api` mode needs the key **inside the container** (`--env-file .env` or
`-e DASHSCOPE_API_KEY=...`); without it the service silently falls back to mock.
Confirm which ran via `report.technical_notes.reasoning_mode` in the response
(`api:<model>` vs `mock`).

## Evaluation

Three protocols in [docs/evaluation.md](docs/evaluation.md): event-level accuracy,
sampling-rate trade-off, and VLM reasoning quality. Measured finding: **~5 FPS is
the practical balance** between event recall and inference cost — the proximity
event is only recalled at ≥5 FPS, while 10 FPS doubles detection cost without
surfacing new events. Run the benchmark yourself:

```bash
python -m src.sampling_benchmark --video examples/input_videos/berlin_tower_3840_2160_30fps.mp4 \
  --fps 1 2 5 10 --config configs/default.yaml --output examples/outputs/benchmarks/berlin_tower
```

## Limitations

A decision-support prototype, **not** a certified safety system.
- 2D bbox-distance proximity has no depth → projection false positives (the VLM
  catches some; see FP-001). Camera angle and occlusion also affect detection.
- VLM outputs can hallucinate; treat `verified`/`risk_level` as advisory.
- No labeled ground-truth set yet, so event-level precision/recall is protocol-only.
- Targets **near-miss / risk-condition** detection on normal footage, not accident
  detection.

## Repository structure

```text
src/
  frame_sampler.py        # fixed-FPS frame sampling
  detector.py             # YOLO object detection
  temporal_aggregator.py  # frame detections -> per-class time segments
  event_generator.py      # rule-based safety event candidates + interpretable risk
  vlm_reasoner.py         # VLM verify/explain (mock + DashScope/Qwen-VL API)
  report_generator.py     # video-level safety report (JSON + Markdown)
  sampling_benchmark.py   # sampling-rate vs runtime/recall benchmark
  pipeline.py             # end-to-end orchestration (one command)
app/main.py               # FastAPI service (/health, /analyze)
configs/default.yaml      # all tunable parameters
docs/                     # evaluation.md, failure_cases.md, architecture.md
examples/                 # input_videos/ and generated outputs/
Dockerfile, docker-compose.yml
```

## Status

Core pipeline complete (sampling → detection → aggregation → safety events → VLM
reasoning → report), plus a sampling benchmark, failure analysis, and a FastAPI
service packaged with Docker. Planned next: labeled event-level evaluation;
multi-stage image slimming.

## Tech stack

Python · OpenCV · Ultralytics YOLOv8 · Alibaba Cloud Model Studio (Qwen-VL) ·
FastAPI · Docker · PyYAML
