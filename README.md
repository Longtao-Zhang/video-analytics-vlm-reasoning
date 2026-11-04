# Video Safety Event Analysis — rules propose, a VLM verifies

> A prototype that turns raw operational video into structured safety events,
> interpretable risk scores and a readable report. Deterministic computer-vision
> rules propose candidate events; a vision-language model (Qwen-VL) verifies and
> explains each one. Designed for warehouse / industrial near-miss review;
> currently demonstrated on a public street clip (see [Evaluation](#evaluation)).

```text
video → frame sampling → YOLOv8 detection → temporal aggregation
      → rule-based safety events → VLM verification → report (JSON + Markdown)
```

## Key design

- **Rules propose, the VLM verifies.** Rules generate auditable candidate events
  with evidence frames; the VLM returns `verified: true/false`, a risk level and an
  explanation for each candidate. It never invents events on its own.
- **Interpretable risk.** Each candidate has a `severity_prior` from a transparent
  weighted score (proximity, duration, detection confidence, crowding), stored with
  its `score_breakdown`, so "why is this high risk?" is always answerable.
- **One event per continuous episode.** Proximity evidence frames are split with
  the same max-gap logic used for temporal aggregation, so two separate approaches
  are two events, not one long one.
- **Conservative rollup.** Video risk is the highest VLM risk among events the VLM
  did *not* reject. Rejected candidates are kept for audit. Events with no usable
  VLM verdict (API failure, malformed reply) are marked `unverified` and still
  count toward the headline risk, flagged for human review.
- **Honest provenance.** `mock` mode runs with no key or network. `api` mode calls
  Qwen-VL through Alibaba Cloud Model Studio (DashScope); failed calls fall back to
  template text, and the report records how many events actually got an API verdict.

## Requirements

Python ≥ 3.10 (the Docker image uses 3.11).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quickstart

Bring any MP4 with people and vehicles in view (videos are not committed to the
repo). Run the whole pipeline with the mock VLM — no API key needed:

```bash
python -m src.pipeline \
  --video path/to/clip.mp4 \
  --config configs/default.yaml \
  --output examples/outputs/demo/clip \
  --vlm-mode mock
```

This writes every intermediate JSON plus `safety_report.md` / `safety_report.json`
to the output directory. YOLOv8n weights are downloaded on first use.

To use the real VLM, set your key and switch modes (endpoint and model are under
`vlm.api` in [configs/default.yaml](configs/default.yaml)):

```bash
export DASHSCOPE_API_KEY=...     # or put it in a local .env file (see .env.example)
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

## HTTP service (FastAPI + Docker)

[app/main.py](app/main.py) exposes the pipeline: `POST /analyze` takes a video
upload and returns the safety report as JSON.

```bash
uvicorn app.main:app --port 8000           # interactive docs at http://localhost:8000/docs

docker build -t safety-video .
docker run --rm -p 8000:8000 safety-video                  # mock mode only
docker run --rm -p 8000:8000 --env-file .env safety-video  # also enables api mode
# or: docker compose up --build
```

```bash
curl localhost:8000/health
curl -F "video=@clip.mp4" "localhost:8000/analyze?vlm_mode=mock"
curl -F "video=@clip.mp4" -H "Authorization: Bearer $SERVICE_API_TOKEN" \
     "localhost:8000/analyze?vlm_mode=api"
```

Upload handling:
- The client filename is never used as a path; uploads are stored under a random
  name in a per-request temp directory that is deleted afterwards.
- Uploads over `service.max_upload_mb` (default 200 MB) are rejected with 413.
- `vlm_mode=api` spends paid API quota, so it is **disabled unless
  `SERVICE_API_TOKEN` is set** on the server, and then requires
  `Authorization: Bearer <token>`. `mock` mode stays open for demos.
- Check `report.technical_notes.reasoning_mode` in the response to see what
  actually ran: `api:<model>`, `api:<model>+partial_fallback`,
  `api_error_fallback`, or `mock`.

## Tests

```bash
pytest
```

27 unit tests covering episode splitting, strict parsing of the VLM verdict,
fallback accounting, the report rollup, and the upload endpoint (path handling,
size limit, token gate). They stub YOLO and the VLM client, so they need no GPU,
weights or network.

## Evaluation

Three protocols in [docs/evaluation.md](docs/evaluation.md): event-level accuracy,
sampling-rate trade-off, and VLM reasoning quality. Only the sampling benchmark has
measured numbers so far, on a single 12-second 4K street clip: the one
person–vehicle candidate appeared only at ≥ 5 FPS, and 10 FPS doubled detection
cost without adding candidates. The VLM then correctly rejected that candidate as
a 2D-projection false positive ([FP-001](docs/failure_cases.md)). These are
indicative results from one clip, not a validated operating point.

```bash
python -m src.sampling_benchmark --video path/to/clip.mp4 \
  --fps 1 2 5 10 --config configs/default.yaml --output examples/outputs/benchmarks/clip
```

## Limitations

A decision-support prototype, **not** a certified safety system.
- Proximity is 2D bbox-center distance with a fixed pixel threshold: no depth, and
  not normalized for resolution or object size (see FP-001).
- No object tracking: evidence is grouped by time, not by object identity.
- VLM verdicts can be wrong or hallucinated; treat `verified` / `risk_level` as
  advisory. The VLM is also a single point of judgement: a manipulated scene (e.g.
  text placed in view) could push it to dismiss a real event.
- No labeled ground-truth set yet, so event-level precision/recall is protocol-only,
  and no warehouse footage has been evaluated.

## Repository structure

```text
src/
  frame_sampler.py        # fixed-FPS frame sampling
  detector.py             # YOLO object detection
  temporal_aggregator.py  # frame detections -> per-class time segments (+ shared gap splitter)
  event_generator.py      # rule-based safety event candidates + interpretable risk
  vlm_reasoner.py         # VLM verify/explain (mock + DashScope/Qwen-VL API)
  report_generator.py     # video-level safety report (JSON + Markdown)
  sampling_benchmark.py   # sampling-rate vs runtime/recall benchmark
  pipeline.py             # end-to-end orchestration (one command)
app/main.py               # FastAPI service (/health, /analyze)
configs/default.yaml      # all tunable parameters
tests/                    # pytest suite (no GPU / network needed)
docs/                     # evaluation.md, failure_cases.md
Dockerfile, docker-compose.yml
```

## Next steps

Labeled event-level evaluation on real warehouse footage; resolution-normalized,
depth-aware proximity; robustness of the VLM verifier against manipulated scenes.

## Tech stack

Python · OpenCV · Ultralytics YOLOv8 · Qwen-VL (Alibaba Cloud Model Studio) ·
FastAPI · Docker · pytest
