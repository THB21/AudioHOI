# Rigid Feature Whole-Sequence Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an isolated, physically continuous rigid-asset pose candidate whose whole-video evidence uses 8 body corners, 4 wheel points, 2 rails and 1 grasp point while preserving the approved 0–124 and 164–end trajectory segments.

**Architecture:** Build named, bidirectional feature evidence from the asset descriptor and clear frames, then solve one SE(3) sequence with locked trusted intervals and factor-gated observations. Keep the candidate outside canonical publication until video approval.

**Tech Stack:** Python, NumPy, SciPy, OpenCV, CoTracker, existing AudioHOI GeometryProvider/factor residuals/rendering.

---

### Task 1: Freeze and verify the two approved references

**Files:**
- Create: `output/suitcase_rigid_sequence_v1/reference_manifest.json`
- Create: `output/suitcase_rigid_sequence_v1/reference_pose.csv`

- [x] Resolve Annotation 1 to `generic-solve-c661e65678cf/state.csv` and Annotation 2 to `generic-solve-e1b1a07dff8d/state.csv` by reproducing their historical overlay frames.
- [x] Convert both state schemas to the canonical pose schema.
- [x] Compose frames 0–124 from Annotation 1 and frames 164–end from Annotation 2.
- [x] Record source paths, source hashes, frame ranges and composed-segment hashes in `reference_manifest.json`.
- [x] Verify frames 0–124 and 164–end compare exactly against their source state values.

### Task 2: Materialize named whole-video rigid feature evidence

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/build_rigid_feature_tracks.py`
- Create: `output/suitcase_rigid_sequence_v1/rigid_feature_tracks.csv`
- Create: `output/suitcase_rigid_sequence_v1/rigid_feature_track_metrics.json`

- [x] Read named local 3D features from the asset descriptor; reject missing body, support, rail or grasp features.
- [x] Project features at clear locked frames and at frame 125 using the Annotation 1 frame-125 hypothesis without locking it.
- [x] Track each named projection forward and backward with CoTracker across the complete video.
- [x] Compute cross-bank consistency, mask compatibility, edge compatibility and visibility confidence without changing feature identity.
- [x] Keep visible wheel/corner evidence independent from the partial-mask minimum-area rectangle.

### Task 3: Add the isolated whole-sequence rigid solver

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/run_rigid_feature_sequence_candidate.py`
- Create: `output/suitcase_rigid_sequence_v1/object_pose_candidate.csv`
- Create: `output/suitcase_rigid_sequence_v1/factor_ledger.csv`
- Create: `output/suitcase_rigid_sequence_v1/trajectory_metrics.json`

- [x] Load one root-SE(3) state for every video frame and hard-lock the two approved intervals.
- [x] Compile corner/wheel/grasp point reprojection and rail axis-line residuals from reliable named tracks.
- [x] Add one-sided visible silhouette containment so occlusion cannot shrink the physical cuboid.
- [x] Add hand-handle co-motion, four-wheel support/penetration, and SE(3) velocity/acceleration residuals.
- [x] Optimize frames 125–163 jointly rather than solving independent PnP poses.
- [x] Reject the candidate if locked values move, the support plane is violated, or unexplained single-frame jumps remain.

### Task 4: Render and review the real candidate

**Files:**
- Create: `output/suitcase_rigid_sequence_v1/object_only/overlay.mp4`
- Create: `output/suitcase_rigid_sequence_v1/keyframes_113_165.jpg`

- [x] Verify the canonical accepted pose SHA256 remains `b86b0834d8c7b61687f8bef24c5d5f9afdaaddc35262d32a8b1ba4fdc7ea648c`.
- [x] Compare reference and candidate values on every locked frame.
- [x] Report maximum translation/rotation step, wheel-plane gap, penetration and feature reprojection error for frames 125–163.
- [x] Render the full overlay and a dense 113–165 contact sheet.
- [x] Do not publish the candidate without explicit user approval.

## Current execution status

The whole-sequence architecture is implemented and rendered. The final review candidate is deliberately rejected rather than promoted: both locked segments are textually exact, maximum translation step is 0.0757 m, and maximum wheel penetration is 0.00523 m, but the optimizer reached its 100-evaluation limit and the maximum rotation step is 8.045° against the declared 8.0° gate. The canonical accepted pose remains unchanged.
