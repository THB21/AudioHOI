# Interval Heading and Rigid Constellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the suitcase full run's recovered turn while reducing rotational jitter and bringing point projection and contact gap through the existing publication gates using generic asset-bound geometry, interval VLM semantics, and audio/visual motion arbitration.

**Architecture:** Bind CoTracker observations to descriptor-declared rigid feature points at reliable visible anchors, then consume those typed point/line measurements in the existing generic sequence solver. Replace framewise semantic heading pressure with interval cumulative-sign and reversal inequalities; audio controls motion-state timing only and cannot freeze a visually moving interval. Keep the rigid object as the only optimized state and publish only isolated candidates until all hard gates and visual review pass.

**Evidence-driven correction (2026-08-05):** The original generic mask/depth initializer projected the fixed asset at roughly twice the observed image height (`z=1.8–2.4m`), so an 18px association gate yielded zero matches. Existing high-IoU MegaPose keyframe hypotheses (`z=3.3–4.8m`) are valid Stage 0 external observations and recover the correct projection scale without reading a final pose. Because the existing CoTracker queries are SAM-interior grid points rather than literal corners, association uses one globally unique best reliable anchor per track/feature pair instead of requiring the same imperfect pose hypothesis to win at two anchors. The 18px gate remains unchanged.

**Second evidence-driven correction (2026-08-05):** The first production solve proved that a SAM-interior CoTracker point cannot be renamed as a descriptor corner/wheel/rail endpoint merely because it is close in one frame: the resulting 2D/3D correspondences were mutually inconsistent (`point p95=161.47px`, frames 111–163 rotation p95 `33.57deg`). The corrected contract assigns each selected CoTracker point its own `track_local:<track_id>` feature by ray-casting it to the descriptor body surface at one reliable anchor. MegaPose contributes translation and heading, while descriptor-declared upright/support axes remove its cuboid pitch/roll symmetry before ray-casting; otherwise almost every ray incorrectly lands on the suitcase bottom. Local XYZ is immutable across all emitted rows, the asset descriptor is not mutated with video-specific points, and production augments the provider from the typed artifact. Named rails remain line factors; four wheels remain descriptor support/visibility geometry rather than falsely identified texture tracks. The full artifact retains provenance while optimization uses at most 16 spatially distributed tracks every 3 frames.

**Tech Stack:** Python 3, NumPy, SciPy, OpenCV, existing typed MeasurementIR/InteractionStateIR/factor runtime, Qwen forced-choice SemanticRelationIR, ffmpeg, YAML/JSON/CSV provenance.

---

## Execution constraints

- Work only in `/mnt/hdd/AudioHOI-object-stage0-solver-integration` on branch `integrate/object-stage0-solver-update`.
- Do not reset, clean, bulk-stage, or overwrite unrelated dirty files.
- Do not push.
- Do not add pytest files or run repository-wide pytest. Use focused temporary assertions, `py_compile`, real solves, metrics, and renders.
- Do not read canonical/final pose as a Stage 1–4 input.
- Do not optimize human state. GVHMR sites are read-only observations.
- Do not modify loss family, publication thresholds, or canonical `object_pose.csv`.
- Render only the full candidate during iteration. Ablations retain pose CSV and metrics only.

## File map

### Create

- `scripts/shared/generic_contact_pipeline/core/measurements/rigid_feature_tracks.py` — generic association of tracked 2D points to descriptor-declared local rigid features using reliable initializer projections.
- `scripts/shared/generic_contact_pipeline/tools/bind_rigid_feature_tracks.py` — materialize `rigid_feature_measurements.csv` and its provenance manifest without reading a final pose.

### Modify

- `scripts/shared/generic_contact_pipeline/configs/assets/suitcase_fixed_rigid.json` — give every corner, wheel, rail endpoint, and handle point a stable feature ID and declare trackable groups.
- `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml` — register the typed track artifact and interval-heading/audio arbitration capability parameters.
- `scripts/shared/generic_contact_pipeline/core/measurements/configured.py` — load descriptor-bound point tracks as `Point2DMeasurement` rows.
- `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py` — register the generic rigid-feature binding task and cache contract.
- `scripts/shared/generic_contact_pipeline/core/solver/semantic_factor_inputs.py` — build interval cumulative-turn and bounded reversal inputs instead of a positive increment target on every frame.
- `scripts/shared/generic_contact_pipeline/core/solver/factor_residuals.py` — evaluate interval sign and excessive-reversal inequalities.
- `scripts/shared/generic_contact_pipeline/core/solver/residual_inputs.py` — pass the new interval payload keys through the existing residual boundary.
- `scripts/shared/generic_contact_pipeline/core/interaction/estimator.py` — arbitrate complete silence intervals against sustained visual displacement before activating static freeze.
- `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py` — compile only calibrated, geometry-consistent, moving interval semantics and corrected grasp/release activation.
- `scripts/shared/generic_contact_pipeline/core/factors/activation.py` — keep interval heading inactive outside persistent moving grasp and immediately after release.
- `scripts/shared/generic_contact_pipeline/tools/prepare_generic_object_problem.py` — retain repeated ablation flags and expose preparation ledgers used by focused verification.
- `output/suitcase_evidence_ablations/candidate_comparison.md` — append the new fair four-way metrics and explicit remaining blockers.

## Task 1: Freeze the current failure and input boundary

**Files:**
- Read: `output/suitcase_evidence_ablations/candidate_comparison.md`
- Read: `output/suitcase_evidence_ablations/candidates/*/generic_object_pose_candidate.csv`
- Read: `samples_known_object/15_suitcase_drag/results/tracking/rigid_point_tracks.csv`
- Create generated diagnostic: `/tmp/suitcase_interval_heading_baseline.json`

- [x] **Step 1: Record the current full and ablation metrics without modifying them**

Run:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python - <<'PY'
import csv, glob, json, math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

root = Path('output/suitcase_evidence_ablations/candidates')
summary = {}
for variant in ('full', 'no_vlm', 'no_audio', 'vision_only'):
    pose = list(csv.DictReader((root / variant / 'generic_object_pose_candidate.csv').open()))
    metrics = json.loads(next((root / variant).glob('hard_metrics.json')).read_text())['metrics']
    rotations = {
        int(row['frame']): Rotation.from_quat([
            float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])
        ])
        for row in pose
    }
    steps = [
        math.degrees((rotations[frame - 1].inv() * rotations[frame]).magnitude())
        for frame in range(112, 164)
    ]
    summary[variant] = {
        'projection_p95_px': metrics['projection_p95_px'],
        'point_projection_p95_px': metrics['point_projection_p95_px'],
        'contact_gap_p95_m': metrics['contact_gap_p95_m'],
        'rotation_p95_deg': float(np.quantile(steps, 0.95)),
        'rotation_max_deg': float(np.max(steps)),
    }
Path('/tmp/suitcase_interval_heading_baseline.json').write_text(
    json.dumps(summary, indent=2, sort_keys=True) + '\n'
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
```

Expected: full remains approximately `23.827 px` total projection, `26.307 px` point projection, `0.08732 m` contact p95, and `12.533 deg/frame` rotation p95 on frames 111–163.

- [x] **Step 2: Prove raw tracker rows are not yet bound to asset features**

Run:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python - <<'PY'
import csv
from pathlib import Path
p = Path('samples_known_object/15_suitcase_drag/results/tracking/rigid_point_tracks.csv')
rows = list(csv.DictReader(p.open()))
assert rows
assert all(not row.get('semantic_feature_id', '').strip() for row in rows)
assert len({row['track_id'] for row in rows}) >= 8
print('unbound rigid tracks reproduced:', len(rows), 'rows')
PY
```

Expected: PASS with all semantic feature IDs empty. This is the failing boundary the next task fixes.

- [x] **Step 3: Verify canonical output remains outside the candidate write set**

Run:

```bash
git status --short samples_known_object/15_suitcase_drag/results/object_pose.csv
```

Expected: no output.

## Task 2: Declare and bind the rigid feature constellation

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/configs/assets/suitcase_fixed_rigid.json`
- Create: `scripts/shared/generic_contact_pipeline/core/measurements/rigid_feature_tracks.py`
- Create: `scripts/shared/generic_contact_pipeline/tools/bind_rigid_feature_tracks.py`

- [x] **Step 1: Split compound asset features into stable point IDs**

Add these descriptor entries while retaining existing aggregate features for compatibility:

```json
"object:body_corner_nx_ny_bottom": [[-0.19, -0.115, 0.07]],
"object:body_corner_nx_py_bottom": [[-0.19, 0.115, 0.07]],
"object:body_corner_px_ny_bottom": [[0.19, -0.115, 0.07]],
"object:body_corner_px_py_bottom": [[0.19, 0.115, 0.07]],
"object:body_corner_nx_ny_top": [[-0.19, -0.115, 0.53]],
"object:body_corner_nx_py_top": [[-0.19, 0.115, 0.53]],
"object:body_corner_px_ny_top": [[0.19, -0.115, 0.53]],
"object:body_corner_px_py_top": [[0.19, 0.115, 0.53]],
"object:wheel_nx_ny": [[-0.145, -0.085, 0.035]],
"object:wheel_nx_py": [[-0.145, 0.085, 0.035]],
"object:wheel_px_ny": [[0.145, -0.085, 0.035]],
"object:wheel_px_py": [[0.145, 0.085, 0.035]],
"object:rail_left_bottom": [[-0.07, 0.125, 0.570]],
"object:rail_left_top": [[-0.07, 0.125, 0.890]],
"object:rail_right_bottom": [[0.07, 0.125, 0.570]],
"object:rail_right_top": [[0.07, 0.125, 0.890]]
```

Declare `visual_tracking_features.point_feature_ids` as the ordered list of the 16 entries plus `object:handle`. Do not encode frame numbers or screen colors.

- [x] **Step 2: Implement the generic association contract**

Create `rigid_feature_tracks.py` with these public records and function:

```python
@dataclass(frozen=True)
class RigidTrackAssociation:
    track_id: str
    geometry_feature_id: str
    anchor_frame: int
    anchor_error_px: float
    confidence: float

@dataclass(frozen=True)
class RigidFeatureTrackBinding:
    associations: tuple[RigidTrackAssociation, ...]
    measurement_rows: tuple[dict[str, object], ...]
    reliable_anchor_frames: tuple[int, ...]
    rejected_by_reason: Mapping[str, int]

The public function is named `bind_rigid_feature_tracks` and accepts keyword-only
`track_rows`, `states_by_frame`, `cameras`, `geometry_provider`, `feature_ids`,
`reliable_anchor_frames`, `maximum_anchor_error_px`, and
`minimum_track_visibility`; it returns `RigidFeatureTrackBinding`. Keep parsing,
projection, assignment, consensus filtering, and row emission in separate private
helpers so each phase can be checked with focused assertions.
```

At each reliable high-IoU external pose anchor, project every one-point feature. Retain the lowest-error anchor for each track/feature pair, then use `scipy.optimize.linear_sum_assignment` once across the sequence to enforce unique track and feature identities. Accept an association only when its projected distance is within `maximum_anchor_error_px`. Emit one row per visible associated track observation with `frame,time,u,v,geometry_feature_id,semantic_role,track_id,confidence,source_anchor_frames`.

Do not use `object_pose.csv`, any final pose, or a hand-edited orientation.

- [x] **Step 3: Add the materialization tool**

`bind_rigid_feature_tracks.py` must accept:

```text
--case
--result-name
--track-artifact
--output-csv
--output-manifest
--maximum-anchor-error-px
--minimum-track-visibility
```

It prepares the capability initializer for provenance, loads `selected_by_visual_geometry` MegaPose hypotheses whose official render-mask IoU passes the declared reliability gate, and passes those external observation states plus descriptor feature IDs into `bind_rigid_feature_tracks`. It atomically writes the CSV and a manifest containing input/output hashes, pose-source hash, association count, feature coverage, reliable anchor frames, rejected reasons, `baseline_pose_read=false`, and `human_state_optimized=false`.

- [x] **Step 4: Run a focused synthetic association check**

Run an inline assertion using four known local points projected through an identity rigid state, perturb each image observation by less than 1 px, and assert that all four tracks map to distinct feature IDs. Then perturb one observation by 100 px and assert it is rejected. Expected: `rigid track binding synthetic check: PASS`.

- [x] **Step 5: Materialize real suitcase bindings**

Run:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/bind_rigid_feature_tracks.py \
  --case suitcase_drag \
  --result-name suitcase_evidence_full \
  --track-artifact samples_known_object/15_suitcase_drag/results/tracking/rigid_point_tracks.csv \
  --output-csv samples_known_object/15_suitcase_drag/results/suitcase_evidence_full/rigid_feature_measurements.csv \
  --output-manifest samples_known_object/15_suitcase_drag/results/suitcase_evidence_full/rigid_feature_measurements_manifest.json \
  --maximum-anchor-error-px 18 \
  --minimum-track-visibility 0.5
```

Expected: at least two rails/rail endpoints, two body corners, and two wheel/support points are bound over reliable visible frames. If this coverage is not achieved, stop and report the actual coverage; do not loosen the radius until wrong associations are accepted.

- [x] **Step 6: Commit only descriptor and binding implementation**

```bash
git add \
  scripts/shared/generic_contact_pipeline/configs/assets/suitcase_fixed_rigid.json \
  scripts/shared/generic_contact_pipeline/core/measurements/rigid_feature_tracks.py \
  scripts/shared/generic_contact_pipeline/tools/bind_rigid_feature_tracks.py
git commit -m "feat: bind rigid asset features to tracked points"
```

## Task 3: Load bound tracks as production MeasurementIR

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/measurements/configured.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`
- Modify: `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py`

- [x] **Step 1: Add the `rigid_feature_points_v1` configured adapter**

In `adapt_configured_supplemental_measurements`, parse each bound row into:

```python
meta = MeasurementMeta(
    measurement_id=f"{profile.case_name}:{frame}:rigid_track:{row['track_id']}:{row['geometry_feature_id']}",
    sample_id=profile.case_name,
    frame=frame,
    time=float(row['time']),
    feature=FeatureRef(str(row['semantic_role']), str(row['geometry_feature_id'])),
    coordinate_frame=CoordinateFrame.IMAGE_PIXELS,
    unit=Unit.PIXEL,
    confidence=float(row['confidence']),
    source=SourceRef(source_path, ('u', 'v', 'track_id', 'geometry_feature_id'), adapter),
)
measurements.append(Point2DMeasurement(meta, float(row['u']), float(row['v'])))
```

Reject an empty feature ID, duplicate `(frame, track_id)`, nonfinite coordinates, confidence outside `[0,1]`, and any feature not declared by the descriptor-backed geometry provider at problem preparation.

- [x] **Step 2: Register the supplemental artifact**

Add to `suitcase_drag.yaml`:

```yaml
supplemental_measurements:
  - adapter: physical_line_endpoints_v1
    artifact: line_observations.csv
    feature_id_field: feature_id
    semantic_role_field: semantic_role
    fps: 24.0
  - adapter: rigid_feature_points_v1
    artifact: rigid_feature_measurements.csv
    fps: 24.0
```

Extend `measurement_roles.point_reprojection` with `rigid_body_corner`, `rigid_wheel_center`, `rigid_rail_endpoint`, and `handle_center`.

- [x] **Step 3: Register the preprocessing cache contract**

Add a generic `bind_rigid_feature_tracks` task after CoTracker and before Stage 1 typed measurement materialization. Inputs are track CSV/manifest, asset descriptor, initializer observations, and camera/profile hashes. Outputs are the bound CSV/manifest. Failure must name missing feature coverage rather than silently falling back to fabricated points.

- [x] **Step 4: Verify real typed coverage without solving**

Prepare the full problem and assert:

```python
roles = Counter(
    measurement.meta.feature.semantic_role
    for measurement in prepared.measurements
    if isinstance(measurement, Point2DMeasurement)
)
assert roles['rigid_body_corner'] > 0
assert roles['rigid_wheel_center'] > 0
assert roles['rigid_rail_endpoint'] > 0
```

Also assert every consumed geometry feature resolves to exactly one local 3D point.

- [x] **Step 5: Commit measurement production**

```bash
git add \
  scripts/shared/generic_contact_pipeline/core/measurements/configured.py \
  scripts/shared/generic_contact_pipeline/core/preprocess/registry.py \
  scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml
git commit -m "feat: consume rigid feature tracks as typed measurements"
```

## Task 4: Replace framewise heading pressure with interval topology

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/semantic_factor_inputs.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/factor_residuals.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/residual_inputs.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [x] **Step 1: Change the heading factor input contract**

Replace `minimum_increment_rad` with:

```python
minimum_cumulative_turn_rad: float = 0.0
maximum_reverse_increment_rad: float = 0.02
```

Validation requires both values nonnegative. `minimum_cumulative_turn_rad` remains `0.0` in the production profile so VLM never prescribes turn magnitude.

- [x] **Step 2: Build one cumulative residual and bounded reversal rows per interval**

For each active calibrated interval:

```python
signed_steps = [
    float(_relative_rotvec(states[f - 1], states[f]) @ normal) * interval.world_yaw_sign
    for f in range(interval.start_frame + 1, interval.end_frame + 1)
    if f - 1 in states and f in states
]
cumulative_agreement.append(float(np.sum(signed_steps)))
reverse_agreement.extend(signed_steps)
```

Return:

```python
{
    'cumulative_signed_turn_rad': cumulative_agreement,
    'reverse_signed_increment_rad': reverse_agreement,
    'cumulative_weight': cumulative_weights,
    'reverse_weight': reverse_weights,
    'minimum_cumulative_turn_rad': factor.minimum_cumulative_turn_rad,
    'maximum_reverse_increment_rad': factor.maximum_reverse_increment_rad,
    'sigma_rad': factor.sigma_rad,
}
```

Do not emit a positive minimum increment for every frame.

- [x] **Step 3: Evaluate only wrong total sign and excessive reverse steps**

Implement:

```python
def heading_topology(
    self,
    cumulative_signed_turn_rad,
    reverse_signed_increment_rad,
    *,
    cumulative_weight,
    reverse_weight,
    minimum_cumulative_turn_rad,
    maximum_reverse_increment_rad,
    sigma_rad,
):
    cumulative = np.asarray(cumulative_signed_turn_rad, dtype=float).reshape(-1)
    reverse = np.asarray(reverse_signed_increment_rad, dtype=float).reshape(-1)
    total_residual = (
        _row_weights(cumulative_weight, len(cumulative))
        * _smooth_hinge(cumulative, minimum_cumulative_turn_rad)
        / sigma_rad
    )
    reverse_residual = (
        _row_weights(reverse_weight, len(reverse))
        * _smooth_hinge(reverse, -maximum_reverse_increment_rad)
        / sigma_rad
    )
    return np.concatenate((total_residual, reverse_residual))
```

This permits small local negative increments and does not determine total angle.

- [x] **Step 4: Restrict compilation to calibrated moving intervals**

An interval is consumed only when it has a world sign, `geometry_consistent=true`, at least one reliable geometry frame, and overlapping `SUPPORTED_MOVING` plus persistent/occluded-hold grasp state. Split mixed moving/static intervals at interaction boundaries; do not propagate one VLM answer across a static section.

- [x] **Step 5: Run focused residual assertions**

Assert that:

1. a cumulative positive turn with small `-0.01 rad` local reversals has near-zero residual;
2. the same trajectory with negative cumulative sign has nonzero interval residual;
3. a `-0.05 rad` reversal exceeds the `0.02 rad` tolerance;
4. doubling every positive increment does not create a penalty, proving magnitude is not prescribed.

Expected: `interval heading residual checks: PASS`.

- [x] **Step 6: Run an isolated full solve and stop if the long turn disappears**

Use the existing 80-evaluation full command. Require frames 111–163 to retain a cumulative physical turn substantially above both no-VLM and no-audio while rotation p95 falls below the `12.533 deg/frame` baseline. Do not tune weights in this step.

- [ ] **Step 7: Commit interval topology**

```bash
git add \
  scripts/shared/generic_contact_pipeline/core/solver/semantic_factor_inputs.py \
  scripts/shared/generic_contact_pipeline/core/solver/factor_residuals.py \
  scripts/shared/generic_contact_pipeline/core/solver/residual_inputs.py \
  scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py \
  scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml
git commit -m "fix: constrain semantic heading at interval level"
```

## Task 5: Arbitrate audio silence against sustained visual motion

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/interaction/estimator.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/semantic_factor_inputs.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [x] **Step 1: Compute visual motion over complete audio intervals**

Before `_frame_state`, compute the existing one-frame/four-frame speed trace. For every silence interval, calculate median and 75th-percentile visual speed over frames that have valid observations. Add profile fields:

```yaml
audio_visual_arbitration:
  sustained_visual_motion_px_per_frame: 1.5
  minimum_conflict_frames: 4
  conflict_fraction: 0.5
```

Mark a silence interval conflicting when at least four frames exceed the visual threshold and their fraction is at least 0.5.

- [x] **Step 2: Prevent conflicting silence from activating static freeze**

Pass `silence_visual_conflict` into `_frame_state`. For support frames:

```python
elif audio_silence and not silence_visual_conflict and not visual_moving:
    contact_mode = InteractionContactMode.SUPPORT
    motion_mode = MotionMode.SUPPORTED_STATIC
elif visual_moving or silence_visual_conflict:
    contact_mode = InteractionContactMode.ROLLING
    motion_mode = MotionMode.SUPPORTED_MOVING
```

Record interval ID, median speed, conflict fraction, and the decision in provenance.

- [x] **Step 3: Keep audio factor inputs consistent with state arbitration**

Set `AudioMotionInterval.visual_speed_is_low=false` for conflicting silence intervals so `build_audio_motion_inputs` omits their zero-speed residuals. Audio onset/offset and sustained-motion rows remain unchanged.

- [x] **Step 4: Verify known intervals**

For the real artifact, assert that the incorrectly silent but visibly moving opening interval cannot create a long static freeze, while the final low-motion silence interval remains supported static. Print exact interval decisions; do not assert frame numbers in core code.

- [ ] **Step 5: Commit audio/visual arbitration**

```bash
git add \
  scripts/shared/generic_contact_pipeline/core/interaction/estimator.py \
  scripts/shared/generic_contact_pipeline/core/solver/semantic_factor_inputs.py \
  scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml
git commit -m "fix: arbitrate silence with visual motion"
```

## Task 6: Correct grasp release and wheel support activation

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/interaction/estimator.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/factors/activation.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`

- [x] **Step 1: Derive release from typed contact continuity**

Retain an active/occluded-hold grasp only while a typed contact row remains active or a short occlusion hold is supported by previous active contact plus handle visibility loss. A semantic `released` relation may close an existing contact interval when confidence passes the fixed tier, but a semantic `active` relation may not invent contact without typed proximity.

- [x] **Step 2: Deactivate hand factors at release**

Ensure `_persistent_contact_state` returns inactive for `RELEASE` and `INACTIVE`. Build `active_frames` for contact distance/relative velocity from the corrected timeline, not every legacy contact row. Record removed frames and the state transition source.

- [x] **Step 3: Preserve wheel support independently**

Support activation remains based on support contact IDs and descriptor support features after grasp release. Use nearest valid wheel/axle group and retain penetration protection. Hand release must not disable wheel-floor support.

- [x] **Step 4: Focused real assertions**

Print every grasp transition and assert no hand factor row exists after the first confirmed release until a new typed active contact appears. Assert support factors remain active in those same frames. Expected: `release/support separation: PASS`.

- [ ] **Step 5: Commit interaction boundary correction**

```bash
git add \
  scripts/shared/generic_contact_pipeline/core/interaction/estimator.py \
  scripts/shared/generic_contact_pipeline/core/factors/activation.py \
  scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py
git commit -m "fix: separate grasp release from object support"
```

## Task 7: Run the fair four-way object ablation and gate promotion

**Files:**
- Modify: `output/suitcase_evidence_ablations/candidate_comparison.md`
- Generated only: `output/suitcase_interval_heading_full_candidate/**`
- Generated only: isolated pose/metrics for no-VLM, no-audio, and vision-only

- [ ] **Step 1: Compile all touched Python files**

Run `python -m py_compile` on every modified/created Python file. Expected: exit 0.

- [ ] **Step 2: Run four isolated solves with identical budgets**

Use `prepare_generic_object_problem.py --max-nfev 80` with:

- full: no ablation flags, required VLM arbitration;
- no-VLM: `--ablation-flag disable_vlm_semantic_evidence`;
- no-audio: `--ablation-flag disable_audio_events`;
- vision-only: both flags.

Before solving, hash the shared observation, rigid feature, line, descriptor, StateSpec, weights, loss, bounds, and iteration budget. Assert all hashes match across variants except evidence availability.

- [ ] **Step 3: Compute the acceptance table**

Report for each variant:

- total and point projection p95;
- contact gap p95/max;
- support gap and penetration;
- rotation p95/max and cumulative turn for frames 62–108 and 111–163;
- static-tail translation and rotation drift;
- active semantic/audio factor IDs and evidence IDs.

- [ ] **Step 4: Render only full**

Write the full object-only overlay to:

```text
output/suitcase_interval_heading_full_candidate/object_only/overlay.mp4
```

Also write a keyframe contact sheet containing reliable visible, turn, occluded, release, and final static frames selected from provenance rather than hard-coded inside core code.

- [ ] **Step 5: Enforce promotion gates**

Do not publish unless all are true:

```text
projection_p95_px < 24
point_projection_p95_px < 24
contact_gap_p95_m < 0.08
rotation_p95_deg_per_frame <= 5 on both turn intervals
rotation_max_deg_per_frame <= 8 on normal frames
no wheel-floor float or penetration
no static-tail drift
full retains the long turn and improves point/contact over no-audio and no-VLM
```

If any gate fails, write the exact blocker and keep the result candidate-only.

- [ ] **Step 6: Update the comparison report and commit verified code/report only**

Stage only files named by Tasks 2–7 and the Markdown comparison. Do not add generated pose/video directories to the commit unless explicitly requested. Run `git diff --cached --check`, inspect the staged diff, and commit with:

```bash
git commit -m "feat: stabilize interval-conditioned rigid reconstruction"
```

Do not push.

## 2026-08-05 execution evidence

- Rebuilt Stage-4 semantic evidence as a five-frame earlier-to-later RGB/typed strip plus a profile-declared asset identity reference. Qwen responses retain prompt, image, model, response, and SHA-256 provenance.
- Disabled production consumption of `visible_face` for this asset after repeated Qwen answers contradicted the visible broad/narrow body aspect. The raw answers remain audit evidence. VLM cannot invent pose or contact.
- Expanded risk coverage to frames 65–113 and 118–171. Only camera-calibrated `turn_direction_screen` and persistent-contact `facing_relation` enter the solver; VLM grasp answers remain non-authoritative conflict diagnostics.
- Re-ran the fair 80-evaluation full candidate. Compared with the existing 80-evaluation no-VLM run, VLM increased frames 111–163 net rotation from `115.552 deg` to `118.764 deg`; rotation p95 was `9.599 deg/frame`. Projection p95 was `28.159 px` and contact p95 was `0.09145 m`, so promotion remained blocked.
- Re-ran full at 160 evaluations only as a convergence diagnostic. The solver converged, but projection/contact stayed at `28.158 px` / `0.09145 m`. This proves the remaining fit gap is observation geometry, not iteration budget. The 160-evaluation result is not used as a fair ablation comparison.
- Rejected an attempted point/line/mask weight increase because projection worsened to `42.796 px` and the frames 62–108 physical turn collapsed. The configuration was reverted.
- Audio gain remains large and interpretable: frames 179–240 net rotation is about `0.01 deg` in full versus `73.681 deg` without audio; translation path is `0.0181 m` versus `0.4402 m`.
- Rendered candidate-only full evidence at `output/suitcase_vlm_audio_refined_full_candidate_160/object_only/overlay.mp4`. Canonical output was not written; hard gates still block promotion.
