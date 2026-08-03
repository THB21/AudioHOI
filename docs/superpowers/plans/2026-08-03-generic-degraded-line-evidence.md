# Generic Degraded Line Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve generic rigid-object orientation when a declared parallel-line pair degrades to one visible line.

**Architecture:** Extend the existing mask observation policy with an unassigned single-axis fallback, then let the isolated generic rigid executor minimize over descriptor-declared line candidates. Keep visual reliability independent from contact state and use existing support and temporal factors after release.

**Tech Stack:** Python, OpenCV, NumPy, pandas, SciPy least-squares, existing GeometryProvider descriptor and factor ledger.

---

### Task 1: Publish degraded single-line evidence

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/components/observation/policies/rigid_mask_track.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [ ] Add a helper that extracts the longest reliable feature-region line when `_parallel_line_pair` returns no pair.
- [ ] Require the configured minimum span, mask support, and direction limits; return no observation when these checks fail.
- [ ] Write one `line_observation_mode=unassigned_axis` row with the descriptor-declared candidate feature IDs and confidence.
- [ ] Rebuild Stage-1 object and line observations in an isolated result artifact and verify frames 141–147 gain single-line evidence without changing earlier paired rows.

### Task 2: Consume an unassigned line without object dispatch

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/tools/run_rigid_feature_sequence_candidate.py`

- [ ] Load paired rows exactly as before and load a single unassigned axis row separately.
- [ ] For each unassigned row, project every descriptor-declared candidate line.
- [ ] Compute axis distance and direction residuals for each candidate and append only the minimum-cost candidate residual.
- [ ] Record `unassigned_rail_axis_line` and `unassigned_rail_direction` in the factor ledger, including the selected candidate index.

### Task 3: Keep fallback mask orientation weak and independent

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/tools/run_rigid_feature_sequence_candidate.py`

- [ ] Compute a main-body principal axis from the SAM2 mask after removing thin appendage rows.
- [ ] Compute the projected principal axis of the eight descriptor body corners.
- [ ] Activate the sign-invariant sine residual only where no line observation exists, weighted by mask visibility and anisotropy.
- [ ] Keep exact mask bounds restricted to visually `visible` frames and never gate them on contact state.

### Task 4: Run real candidate validation

**Files:**
- Create: `output/suitcase_rigid_sequence_v11/object_pose_candidate.csv`
- Create: `output/suitcase_rigid_sequence_v11/factor_ledger.csv`
- Create: `output/suitcase_rigid_sequence_v11/trajectory_metrics.json`
- Create: `output/suitcase_rigid_sequence_v11/review/object_only/overlay.mp4`

- [ ] Confirm frame 160 is `inactive` and frames 146–163 contain zero grasp-related factor rows.
- [ ] Confirm every consumed named feature is mask-compatible.
- [ ] Compare 143–150 rotation steps with v10 and locate the maximum step.
- [ ] Render the lightweight object overlay and inspect frames 137, 143–150, 154, 160, and 164.
- [ ] Reject publication unless locked values are exact, support penetration is below 1 cm, the optimizer converges, the declared rotation-step gate passes, and visual branch continuity is correct.
- [ ] Recompute the accepted-pose SHA256 and verify it remains `b86b0834d8c7b61687f8bef24c5d5f9afdaaddc35262d32a8b1ba4fdc7ea648c`.

