# Interval Heading and Rigid Constellation Design

## Status

Approved direction. This document defines the next object-only change after
commit `065d6c8c`. It does not authorize publication of a canonical pose.

## Goal

Preserve the full run's recovered suitcase turn while removing rotational
jitter and improving pixel/contact fit. Audio determines when supported motion
occurs, VLM supplies discrete turn and face-topology evidence, and fixed-asset
visual features determine the continuous rigid pose.

The same mechanisms must apply to any fixed rigid asset with declared feature
geometry. No suitcase case branch or object-specific optimizer may be added.

## Current evidence

The fair four-way ablation at commit `065d6c8c` shows:

| Variant | Projection p95 (px) | Point p95 (px) | Contact p95 (m) | Rotation path, frames 111–163 | Rotation p95 |
|---|---:|---:|---:|---:|---:|
| full | 23.827 | 26.307 | 0.08732 | 289.919 deg | 12.533 deg/frame |
| no VLM | 21.245 | 28.533 | 0.09123 | 161.278 deg | 9.584 deg/frame |
| no audio | 23.757 | 28.528 | 0.09097 | 163.769 deg | 9.816 deg/frame |
| vision only | 13.344 | 28.939 | 0.09148 | 159.113 deg | 9.602 deg/frame |

The full method recovers the long turn and improves point/contact fit, but its
frame-level semantic heading pressure conflicts with visual projection and
creates excess angular variation. More global smoothing is not an acceptable
fix because it has already been observed to damage projection fit and erase
real irregular motion.

## Scope and boundaries

- Optimize object state only.
- GVHMR sites remain read-only observations for hand/object relation and
  rendering. Human pose is never refined or published by this work.
- Keep the fixed suitcase body, rails, handle, and wheels in one rigid state.
- Do not change loss family, publication thresholds, or canonical outputs.
- Do not add a case-name dispatch or a suitcase-specific solver.
- VLM may output only forced-choice discrete relations and confidence. It may
  not output position, orientation, angle, or continuous factor weights.
- Audio may output only events/intervals and confidence. It may not output pose.
- Existing full/no-VLM/no-audio/vision-only candidates remain isolated until
  the full result passes hard metrics and visual review.

## Considered approaches

### A. Interval semantics plus rigid feature constellation — selected

Use VLM only to constrain turn topology over an interval and use tracked,
asset-bound points and lines to solve each frame's continuous pose. This keeps
the correct semantic branch without allowing VLM to inject framewise rotation.

### B. Stronger global temporal smoothing — rejected

This reduces visible jitter quickly but also suppresses real start/stop and turn
dynamics and worsens projection. It treats the symptom rather than the evidence
conflict.

### C. Full-trajectory hypothesis generation and VLM selection — deferred

This could resolve monocular ambiguity but adds substantial complexity and
risks recreating a case dispatcher. It is unnecessary before the existing
geometry evidence is consumed correctly.

## Architecture

### 1. Asset-bound rigid constellation

The asset descriptor declares semantic feature groups in local coordinates:

- eight body corners;
- four wheel support points;
- two rail segments, each with upper and lower endpoints;
- one handle center;
- two wide faces and two side faces with stable topology IDs.

These declarations are data, not solver logic. The generic geometry provider
already maps local features through the object state.

At reliable visible frames, Stage 1 associates tracked image points/lines with
these asset feature IDs. Each measurement records confidence, visibility,
source track ID, and association provenance. Unassigned image tracks remain
valid generic tracks but do not become 3D reprojection constraints until an
association is available.

The production solver consumes:

- point reprojection for body corners, wheel points, rail endpoints, and handle;
- line reprojection for the two rails;
- mask silhouette for coarse body support;
- support-plane distance for wheel groups;
- the existing rigid geometry transform, which guarantees constellation
  rigidity without a separate deformable state.

### 2. Interval-level semantic heading

Replace frame-level signed heading pressure with an interval factor. A VLM
relation may declare only:

- clockwise, counterclockwise, stationary, or unclear;
- visible face topology at reliable windows;
- whether occlusion explains missing geometry evidence.

The compiled factor acts on an interval, not every independent frame. It may:

- penalize a cumulative turn with the wrong sign;
- penalize repeated sign reversals above a small dead zone;
- enforce compatibility between interval endpoints and declared face topology;
- abstain when geometry cannot calibrate screen direction to world yaw.

It must not prescribe total angle or per-frame angular velocity. Visual
reprojection, contact, support, and temporal factors determine those values.

The factor is active only when all conditions hold:

1. the VLM answer is not `unclear` and meets the configured confidence tier;
2. visible geometry supplies a consistent screen-to-world sign calibration;
3. the interaction timeline marks the interval as supported moving or attached;
4. no reliable visible geometry contradicts the semantic relation.

Conflicting evidence disables the semantic factor and writes a provenance row;
it never overrides the geometry.

### 3. Audio and interaction-state arbitration

Audio controls motion-state timing only:

- sustained motion, short tug, onset, and offset support moving intervals;
- silence supports static intervals only when sustained visual displacement
  does not contradict it;
- seam clicks remain event evidence and do not create orientation constraints.

The interaction estimator evaluates audio over intervals with visual-motion
hysteresis. A single low-energy frame cannot activate static freeze. When audio
and visual evidence conflict, the state remains non-static and the conflict is
recorded instead of silently choosing audio.

This allows audio to activate the interval heading relation together with VLM,
while preventing incorrect silence labels from freezing genuine motion.

### 4. Continuous solver behavior

The generic sequence solver remains the only continuous optimizer. The state is
the existing rigid translation plus SO(3) rotation.

The factor program uses:

- point, line, mask, and depth factors for observation fit;
- contact distance and contact relative velocity during valid grasp intervals;
- wheel/support-plane factors for ground consistency;
- angular velocity and angular acceleration through the existing manifold-aware
  temporal residuals;
- interval signed-turn topology as a bounded semantic inequality.

Weights remain a small profile-declared tier set: active, downweighted, or
inactive. Reliability changes factor activation/tier; neither VLM nor audio may
generate arbitrary weights.

No global post-solve smoothing is required for acceptance. A local outlier
repair may be evaluated only if it preserves hard metrics and is generic,
provenance-recorded, and inactive when no nonphysical step is present.

### 5. Grasp, release, and support

Persistent grasp constrains handle-to-hand distance and relative velocity, using
read-only human sites. It does not optimize the human.

After a release state, hand contact factors must deactivate immediately. Wheel
support remains active while the suitcase is supported. This avoids forcing the
object toward a hand that no longer holds it and addresses the remaining contact
gap without increasing the contact weight globally.

## Data flow

```text
asset feature declarations
        +
SAM2 mask / CoTracker tracks / rail observations / depth
        |
        v
typed asset-bound point and line measurements
        |
audio intervals ---> InteractionStateIR <--- VLM interval relations
        |                    |
        +----------+---------+
                   v
       generic factor compilation
                   v
       one GenericSequenceExecutor
                   v
 isolated candidate + metrics + provenance + object render
```

## Failure handling

- A feature association without a descriptor-backed local coordinate is not
  consumed by the solver.
- A heading DP with no continuous path fails explicitly; it never selects an
  arbitrary candidate.
- A VLM relation without geometry sign calibration is inactive.
- Contradictory audio silence and sustained visual motion cannot activate
  static freeze.
- Missing or occluded tracks lower visual confidence; they are never filled with
  zero coordinates.
- Hard metric failure blocks publication regardless of VLM preference.

## Verification and acceptance

Use focused assertions, compilation checks, real candidate solves, and renders;
do not add repository pytest files.

The next full candidate must satisfy all of the following before promotion:

- total projection p95 below 24 px;
- point projection p95 below 24 px;
- contact p95 below 0.08 m;
- rotation p95 at most 5 deg/frame on frames 62–108 and 111–163;
- maximum normal rotation step at most 8 deg/frame;
- retain the long physical turn on frames 111–163 without reversing its
  topology;
- no visible wheel-floor float or penetration;
- no static-tail drift;
- full improves point/contact and turn-topology metrics over both no-audio and
  no-VLM under identical solver budgets;
- canonical output remains unchanged until explicit visual approval.

The four ablation variants must continue to differ only in typed evidence
availability. Their pose CSVs and hard metrics are sufficient; only the full
candidate requires rendering during iteration.

## Implementation order

1. Add complete rigid feature declarations to the asset descriptor.
2. Bind reliable point/line tracks to descriptor feature IDs with provenance.
3. Replace framewise semantic heading pressure with the interval topology
   residual and strict activation conditions.
4. Resolve audio/visual motion conflicts before static/heading activation.
5. Correct grasp release and wheel support activation.
6. Run one-variable real probes, then the full four-way ablation.
7. Render only the full candidate and compare against the acceptance gates.
