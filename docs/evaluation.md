# Evaluation

How we assess the pipeline's quality, cost, and reliability. Three protocols:
**event-level accuracy**, **sampling-rate trade-off**, and **VLM reasoning
quality**. The system is a prototype, so the emphasis is on an honest,
reproducible methodology rather than headline scores — including where it fails
(see [failure_cases.md](failure_cases.md)).

> Status note: event-level precision/recall requires hand-labeled ground truth,
> which does not exist yet — that section documents the protocol and is seeded
> with one analyzed case. The sampling-rate benchmark below is fully runnable and
> reports **real measured numbers**.

---

## 1. Event-level evaluation (protocol)

**Goal:** does the system detect the safety events that actually occur, without
over-flagging?

**Ground truth:** for a handful of clips, a human annotates each true safety
event as `{event_type, start_time, end_time}` in a small JSON file alongside the
clip. (Not yet collected.)

**Metrics:**
- **Event precision / recall** — a predicted event matches a labeled event if the
  type agrees and their time spans overlap above a temporal-IoU threshold.
- **Temporal IoU** — `intersection / union` of the predicted vs labeled time
  windows; reported as the mean over matched events.
- **False positives / false negatives** — enumerated as cases, not just counts,
  and cross-referenced to [failure_cases.md](failure_cases.md).

**Important:** because rules propose and the VLM verifies, report two precision
numbers — **rule precision** (candidates) and **verified precision** (after
applying the VLM `verified` flag). The gap between them quantifies how much the
VLM layer is worth.

**Seed result:** FP-001 (`berlin_tower`, person–vehicle proximity) is a confirmed
**rule false positive** that the VLM correctly rejected (`verified: false`) — i.e.
it lowers rule precision but not verified precision. See FP-001 for details.

---

## 2. Sampling-rate benchmark (measured)

**Goal:** quantify the trade-off between sampling rate, runtime, and event recall.
Higher FPS captures more evidence but costs more detection runtime and more
downstream VLM calls.

**Run it:**
```bash
python -m src.sampling_benchmark \
  --video examples/input_videos/berlin_tower_3840_2160_30fps.mp4 \
  --fps 1 2 5 10 \
  --config configs/default.yaml \
  --output examples/outputs/benchmarks/berlin_tower
```

**Result** — `berlin_tower_3840_2160_30fps.mp4`, YOLOv8n, CPU (macOS):

| FPS | Sampled frames | Detections | Segments | Safety events | Detect (s) | Total (s) | Frames/s |
|---|---|---|---|---|---|---|---|
| 1  | 12  | 39  | 1 | 0 | 0.74 | 2.84  | 4.22  |
| 2  | 23  | 74  | 1 | 0 | 1.38 | 3.67  | 6.26  |
| 5  | 58  | 192 | 3 | 1 | 3.51 | 6.34  | 9.15  |
| 10 | 115 | 381 | 3 | 1 | 7.29 | 10.99 | 10.47 |

**Reading it:**
- The person–vehicle proximity event is **only recalled at ≥ 5 FPS**. At 1–2 FPS
  the person and vehicle never land in the same sampled frame, so the proximity
  rule cannot fire — a sampling-induced false negative.
- Detection runtime scales roughly linearly with sampled frames; going from 5 → 10
  FPS doubles detection cost (3.5s → 7.3s) but surfaces **no new safety events**
  on this clip.
- (Caveat: the VLM later judged the single recalled event a false positive — so on
  this particular clip the "true" recall is 0. The point stands as a method: recall
  of *candidates* is sampling-sensitive and saturates.)

**Conclusion:** ~5 FPS is the practical balance between event recall and inference
cost for near-real-time safety monitoring; 10 FPS roughly doubles cost without
additional events on this footage. (Validate per-site, since fast-moving forklifts
may need higher rates.)

---

## 3. VLM reasoning quality check (protocol)

**Goal:** the VLM is not trusted blindly — we audit its judgements.

**Protocol:** sample ~20 `vlm_event_reasoning.json` outputs and have a human label
each as **correct / partially correct / incorrect**, recording which common error
it exhibits:
- over-interpreting static objects as active hazards
- missing occluded workers
- hallucinating motion / contact from still frames
- confusing vehicle type
- (positive) correctly rejecting a rule false positive

**Metrics:** share correct/partial/incorrect; agreement between VLM `verified` and
the human verdict; how often `verified: false` correctly overturns a rule candidate.

**Seed result:** in FP-001 the VLM was **correct** — it rejected a 2D-projection
proximity false positive and explained why (separate elevation / barrier), which is
exactly the verification behavior this layer is meant to provide.

---

## Reproducibility

All stages are deterministic CLIs writing JSON; the report and benchmark also emit
Markdown. End-to-end for one clip:

```bash
python -m src.frame_sampler        --video <v> --config configs/default.yaml --output <dir>
python -m src.detector             --sampled-frames-json <dir>/sampled_frames.json --config configs/default.yaml --output <dir>
python -m src.temporal_aggregator  --detections-json <dir>/detections.json --config configs/default.yaml --output <dir>
python -m src.event_generator      --detections-json <dir>/detections.json --temporal-segments-json <dir>/temporal_segments.json --config configs/default.yaml --output <dir>
python -m src.vlm_reasoner         --safety-events-json <dir>/safety_event_candidates.json --config configs/default.yaml --output <dir> --mode mock
python -m src.report_generator     --vlm-reasoning-json <dir>/vlm_event_reasoning.json --config configs/default.yaml --output <dir>
```
