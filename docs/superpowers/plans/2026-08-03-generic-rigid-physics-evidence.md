# Generic Rigid Physics Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce typed, case-independent rigid silhouette, relative-depth, feature-track, and pose-hypothesis evidence with a hard pre-solve validation manifest.

**Architecture:** A focused measurement module owns immutable evidence records and validation. A command-line builder adapts existing mask/depth, CoTracker, and MegaPose artifacts into CSV/JSONL outputs. This phase never reads or writes an accepted pose and never runs a solver.

**Tech Stack:** Python 3.10, dataclasses, NumPy, pandas, existing Measurement IR conventions.

---

## File structure

- Create `scripts/shared/generic_contact_pipeline/core/measurements/rigid_physics.py`: typed evidence records, JSON-safe serialization, and manifest validation.
- Modify `scripts/shared/generic_contact_pipeline/core/measurements/__init__.py`: export the new contract.
- Create `scripts/shared/generic_contact_pipeline/tools/build_rigid_physics_evidence.py`: artifact adapter and diagnostic producer.
- Generate `output/suitcase_rigid_physics_evidence/`: ignored real-run evidence used for verification, never committed.

### Task 1: Add the typed rigid physics evidence contract

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/measurements/rigid_physics.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/measurements/__init__.py`

- [x] **Step 1: Define frame-level silhouette and depth evidence**

Implement frozen `RigidSilhouetteEvidence` and `RelativeDepthEvidence`
dataclasses. Silhouette rows contain sample/frame/time, visibility, centroid,
stable body bbox, mask area, log width/height/area/aspect, reliability, and
source artifact. Depth rows contain raw and log metric depth, confidence, and
source artifact. Reject non-positive dimensions/depth, invalid visibility,
invalid confidence, and non-finite values.

- [x] **Step 2: Define feature-track and pose-hypothesis evidence**

Implement frozen `RigidFeatureTrackEvidence` and
`RigidPoseHypothesisEvidence`. Track rows retain candidate feature identities,
role, coordinates, tracker confidence, boundary distance, cross-bank error,
anchor trust, usability, and rejection reason. Pose rows retain every provider
hypothesis rather than publishing one selected pose.

- [x] **Step 3: Define the blocking manifest**

Implement `RigidPhysicsEvidenceManifest` with coverage and contamination counts,
input hashes, named gates, and `ready_for_solver`. Validate that
`ready_for_solver == all(gates.values())`.

- [x] **Step 4: Verify imports and invalid-value rejection without pytest**

Run `py_compile`, instantiate valid rows, and confirm zero-width silhouette and
non-positive depth constructors raise `ValueError`.

- [x] **Step 5: Commit the contract**

Commit only the new contract and package exports with message
`feat: add rigid physics evidence contract`.

### Task 2: Build silhouette and relative-depth evidence

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/build_rigid_physics_evidence.py`

- [x] **Step 1: Add generic artifact inputs**

The CLI accepts `--sample-id`, `--object-observations`, `--feature-tracks`,
`--megapose-hypotheses`, `--trusted-anchor-intervals`, and `--output-dir`.
It validates required columns and hashes every input.

- [x] **Step 2: Adapt silhouette rows**

Derive stable width, height, log scale, and aspect from body bbox columns. Write
`rigid_silhouette_evidence.csv`. Only visible rows with observation confidence
above 0.5 set `scale_reliable=1`; partial and occluded rows remain present.

- [x] **Step 3: Adapt relative-depth rows**

Accept finite positive `object_ref_depth_m` with positive `depth_conf`. Write
`relative_depth_evidence.csv`, preserving raw and log depth without aligning it
to a solved pose.

- [x] **Step 4: Emit trend diagnostics**

For consecutive reliable frames write `scale_depth_trend.csv` containing
delta-log-depth, delta-log-height, delta-log-area, and whether farther depth is
consistent with smaller silhouette scale.

### Task 3: Audit CoTracker and MegaPose evidence

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/tools/build_rigid_physics_evidence.py`

- [x] **Step 1: Validate trusted anchors**

Parse interval strings such as `1-124,164-240`. Reject tracks whose anchor does
not fall inside a trusted interval with reason `untrusted_anchor`.

- [x] **Step 2: Validate feature roles**

Use boundary limits of 12 px for body/support points, 10 px for line endpoints,
and 18 px for grasp points. A track is usable only when its anchor is trusted,
tracker visibility is at least 0.65, cross-bank error is at most 18 px, it is
mask compatible, and its boundary distance satisfies the declared role. Write
`rigid_feature_track_evidence.csv` with explicit rejection reasons.

- [x] **Step 3: Normalize MegaPose hypotheses**

Write every row to `rigid_pose_hypotheses_evidence.jsonl`. Mark a frame
ambiguous when the best two mask-IoU hypotheses differ by at most 0.08 or the
provider reports blocked visual evidence. Never publish the provider-selected
hypothesis as a trajectory.

- [x] **Step 4: Build the blocking manifest**

Require clear scale and relative-depth coverage, zero contaminated usable
tracks, zero usable tracks from untrusted anchors, at least two usable rigid
feature frames, and at least two trusted rail frames. Report MegaPose ambiguity
without treating one hypothesis as truth.

- [x] **Step 5: Verify on the real suitcase artifacts**

Run the builder with trusted intervals `1-124,164-240`. Confirm anchor 125 is
untrusted, deep-interior body tracks are rejected, frames 155 and 163 report
blocked MegaPose evidence, and the manifest blocks solving when trusted rail
coverage is absent.

- [x] **Step 6: Commit the builder**

Commit only the builder with message `feat: build rigid physics evidence audit`.

### Task 4: Record evidence-phase status

**Files:**
- Modify: `docs/superpowers/plans/2026-08-03-generic-rigid-physics-evidence.md`

- [x] **Step 1: Record exact manifest results**

Check completed steps and append real counts and failed gate names. Do not mark
solver integration complete in this plan.

- [x] **Step 2: Verify isolation**

Run `git diff --check`, inspect `git status --short`, and verify the accepted
pose SHA256 remains
`b86b0834d8c7b61687f8bef24c5d5f9afdaaddc35262d32a8b1ba4fdc7ea648c`.

- [x] **Step 3: Commit plan status**

Commit only this plan update with message `docs: record rigid evidence audit status`.

## Real suitcase evidence audit (2026-08-03)

The builder was run against the current suitcase observation, CoTracker, and
MegaPose artifacts with trusted anchor intervals `1-124,164-240`. It did not
read or write the accepted pose and did not invoke a solver.

- Frames: 240; solve interval: `125-163`.
- Clear scale rows in the solve interval: 29; relative-depth rows: 39.
- Usable rigid-feature frames: 6; trusted rail frames: 0.
- Trusted orientation frames in the solve interval: `[161]`.
- Maximum visible orientation gap: 29 frames; allowed maximum: 12 frames.
- Input tracks rejected as contaminated: 1,368; tracks rejected for an
  untrusted anchor: 198.
- Ambiguous or visually blocked MegaPose frames: `1,106,118,155,163,200,226`.
- Passing gates: clear scale coverage, relative-depth coverage, no
  contaminated usable tracks, no untrusted-anchor usable tracks, and minimum
  rigid-feature coverage.
- Failing gates: `trusted_rail_coverage` and
  `bounded_visible_orientation_gap`.
- Publication status: `solver_blocked`; `ready_for_solver=false`.

This phase therefore establishes the exact missing evidence instead of
publishing another hand-shaped yaw trajectory. Solver integration is
deliberately not marked complete. The next implementation phase must regenerate
persistent rail/corner identities from clear frames and compile signed-face,
scale-depth, and support factors before the blocked interval is solved.
