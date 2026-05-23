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
- [ ] Video metadata extraction
- [ ] Frame sampling
- [ ] Object detection
- [ ] Temporal aggregation
- [ ] Event generation
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

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Coming soon.

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
