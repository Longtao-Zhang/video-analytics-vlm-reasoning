# Video Analytics Pipeline with VLM-assisted Event Reasoning

## Motivation

This project builds a modular video analytics pipeline for object detection,
temporal event aggregation, and VLM-assisted event reasoning.

It is designed as a practical computer vision and machine learning engineering
portfolio project. The focus is not only on model inference, but also on
structured outputs, evaluation, reproducibility, and deployment-oriented
engineering practices.

## Planned Architecture

```text
Input Video
  -> Frame Sampling
  -> Object Detection
  -> Tracking / Temporal Aggregation
  -> Event Candidate Generation
  -> VLM-assisted Event Reasoning
  -> Structured JSON Report
```

## Current Progress

- [x] Project structure
- [x] GitHub repository setup
- [x] Video metadata extraction
- [x] Frame sampling
- [x] Object detection
- [X] Temporal aggregation
- [X] Event generation
- [ ] VLM-assisted reasoning
- [ ] Evaluation and benchmarking
- [ ] FastAPI inference service
- [ ] Docker deployment

## Repository Structure

```text
video_analytics/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- video_loader.py        # Video loading and metadata extraction
|   |-- frame_sampler.py       # Frame sampling strategies
|   |-- detector.py            # Object detection interface
|   |-- tracker.py             # Tracking and temporal aggregation
|   |-- event_generator.py     # Rule-based event candidate generation
|   |-- vlm_reasoner.py        # VLM-assisted event interpretation
|   |-- pipeline.py            # End-to-end pipeline orchestration
|   `-- utils.py               # Shared utilities
|-- app/
|   `-- main.py                # FastAPI application entry point
|-- configs/
|   `-- default.yaml           # Default pipeline configuration
|-- examples/
|   |-- input_videos/          # Sample videos for local experiments
|   `-- outputs/               # Generated reports and intermediate outputs
|-- docs/
|   |-- architecture.md        # System design notes
|   |-- evaluation.md          # Evaluation protocol and metrics
|   `-- failure_cases.md       # Known limitations and failure analysis
`-- tests/                     # Unit and integration tests
```

## Engineering Practices

This project aims to follow practical software engineering practices for ML/CV
systems:

- Modular Python package structure
- YAML-based configuration
- Structured JSON outputs
- Logging and error handling
- Pytest-based testing
- Reproducible command-line execution
- Evaluation and benchmarking documentation
- Deployment-oriented design with FastAPI and Docker

## Setup

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux (bash):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Roadmap

### Week 1

- Video metadata extraction
- Frame sampling
- YOLO-based object detection
- Temporal aggregation
- Event candidate generation

### Week 2

- VLM-assisted reasoning
- Sampling strategy benchmark
- Failure case analysis
- FastAPI inference endpoint
- Docker packaging

### Week 3

- Testing and refactoring
- Resume and interview preparation
- Project polish

## Usage

### Video Metadata Extraction
```bash
python -m src.video_loader --video examples/input_videos/video.mp4 \
--output examples/outputs/demo
```

### Frame Sampling
```bash
python -m src.frame_sampler --video examples/input_videos/demo.mp4 \
--config configs/default.yaml \
--output examples/outputs/demo
```

### Object Detection

This project uses YOLO for frame-level object detection on sampled video frames.

```bash
python -m src.detector \
  --sampled-frames-json examples/outputs/demo/sampled_frames.json \
  --config configs/default.yaml \
  --output examples/outputs/demo
```
The detector outputs:

```text
examples/outputs/demo/
├── detections.json
└── visualized_frames/
```

Each detection contains:
```json
{
  "label": "person",
  "confidence": 0.91,
  "bbox_xyxy": [120.5, 80.2, 300.7, 420.9]
}
```

### Temporal Aggregation
Converts frame-level YOLO detections into object presence segments.    
```bash
python -m src.temporal_aggregator \
--detection-json example/output/demo/detection.json \
--config configs/default.yaml \
--output example/output/demo
```

### Event Candidate Generation

The event generation module converts temporal object segments and frame-level detections into rule-based event candidates.

```bash
python -m src.event_generator \
  --detections-json examples/outputs/demo/detections.json \
  --temporal-segments-json examples/outputs/demo/temporal_segments.json \
  --config configs/default.yaml \
  --output examples/outputs/demo
```

The output is saved to:  
```text
examples/outputs/demo/event_candidates.json
```

Currently supported event types:  
- person_present
- vehicle_present
- multiple_people_present
- person_near_vehicle

Each event candidate contains:  
```JSON
{
  "event_id": "event_001",
  "event_type": "person_present",
  "start_time": 0.0,
  "end_time": 6.0,
  "duration_seconds": 6.0,
  "objects_involved": ["person"],
  "num_evidence_frames": 7,
  "confidence": 0.84
}
```