# Failure Cases & Analysis

Concrete failures observed in the pipeline, with root causes and mitigations. The
system is a decision-support prototype, not a certified safety product, so we
track where it gets things wrong.

Because rules propose candidates and the VLM verifies them, failures fall into:
**rule false positive** (rule fires on a non-event — caught by VLM `verified:false`
or human review), **rule false negative** (real event missed, e.g. occlusion /
sampling gap — caught only by human review), and **VLM error** (misreads the
frames — hallucinated motion/contact, wrong object, missed occluded worker).

---

## FP-001 — Person–vehicle proximity, 2D projection (depth ignored)

- **Clip:** `Berlin_tower_3840_2160_30fps.mp4` @ 5 FPS · **Event:** `safety_event_001` `person_vehicle_proximity` · **Window:** 00:08–00:11
- **Category:** Rule false positive — caught by the VLM (`verified: false`)

**Rule:** fired at `min_distance_px=325.07` (threshold 400) → `severity_prior=medium`,
`risk_score=0.3586` (proximity 0.187 · duration 0.180 · confidence 0.815 ·
crowding 0.667). The score barely clears the low/medium bin (0.33); the proximity
factor is actually low — `medium` is driven by confidence and crowding, not closeness.

**VLM (`qwen3-vl-flash`):** rejected it — `verified:false`, `risk_level:low`,
`confidence:0.90`. Pedestrians are on a raised walkway, the car on a lower road,
separated by a barrier/height difference — *"2D projection error without 3D context."*

**Root cause:** the rule uses 2D bbox-center pixel distance
(`generate_person_vehicle_proximity_events` in [src/event_generator.py](../src/event_generator.py)),
which has no depth — objects far apart in 3D can project close together, especially
from an elevated/wide-angle camera.

**Mitigations:** (1) demote `verified:false` events in risk aggregation / reports;
(2) add a depth proxy (bbox foot position + relative size) or a same-plane gate;
(3) normalize the threshold by frame size / person height (325px ≠ same at 4K vs 720p);
(4) add this clip to the eval set as a labeled false positive.

**Why it matters:** this is the hybrid design working as intended — rules give cheap,
auditable recall; the VLM supplies the 3D context they lack and correctly downgrades
a `medium` candidate to a verified non-event.

---

## Template for new cases

```markdown
## <ID> — <short title>
- **Clip:** <video> @ <fps> · **Event:** <event_id> <event_type> · **Window:** MM:SS–MM:SS
- **Category:** Rule false positive | Rule false negative | VLM error

**Rule:** <rule_triggered, key metrics, severity_prior, risk_score/breakdown>
**VLM:** <verified, risk_level, confidence, paraphrased reasoning>
**Root cause:** <why>
**Mitigations:** <next steps>
```
