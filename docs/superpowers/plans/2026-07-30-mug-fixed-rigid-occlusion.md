# Mug Fixed-Rigid Occlusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the 240-frame mug trajectory with the single generic sequence solver, treating body and handle as one fixed rigid asset and preserving pose through handle occlusion with read-only hand contact evidence.

**Architecture:** Replace the nine-dimensional root-plus-phase production state with an eight-dimensional fixed-rigid state. A descriptor-driven axial-rigid initializer composes body tilt and off-axis feature phase into one root quaternion; typed visual factors run only where their measurements exist, while persistent-contact and temporal factors carry the object through semantic handle occlusion. Qwen marks 60 frame-local hidden states; after combining that gate with missing thin-handle detections, 131 handle points remain and 109 frames contain no fabricated handle point.

**Tech Stack:** Python, NumPy, SciPy, OpenCV, typed Measurement/Interaction/Factor IR, `GenericSequenceExecutor`, URDF/OBJ asset descriptors, FFmpeg.

---

## File structure

- Modify `scripts/shared/generic_contact_pipeline/configs/assets/mug_periodic_rigid.json`: declare the fixed-rigid state, axial body proxy, fixed semantic feature points, and generic initializer capability.
- Modify `scripts/shared/generic_contact_pipeline/configs/cases/mug.yaml`: select typed visual/depth/contact factors and remove the production periodic seed/prior.
- Modify `scripts/shared/generic_contact_pipeline/core/state/asset_state_contract.py`: parse descriptor-declared rigid feature points and axial initializer metadata without object names.
- Modify `scripts/shared/generic_contact_pipeline/core/solver/capability_initializers.py`: implement the reusable axial-rigid feature-correspondence initializer.
- Modify `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`: route any matching descriptor capability through `build_asset_state_contract()` and the generic initializer/provider.
- Modify `scripts/shared/generic_contact_pipeline/core/factors/adapters.py`: select periodic factors from declared state capabilities rather than `profile.case_name == "mug"`.
- Modify the existing Articraft mug render backend only if required to consume the fixed-rigid pose with zero relative phase; this remains presentation compatibility, not solver logic.
- Modify `docs/interaction_state_conditioned_generic_solver_plan.md`: record evidence, remaining automatic gates, and five-case promotion coverage.

### Task 1: Convert the asset to a fixed-rigid contract

- [x] **Step 1: Record the current production contradiction**

Run:

```bash
rg -n 'observation_periodic_rigid|periodic_phase_prior|profile.case_name == "mug"' \
  scripts/shared/generic_contact_pipeline/core \
  scripts/shared/generic_contact_pipeline/configs/cases/mug.yaml
```

Expected: production reads `observation_seed/body_pose.csv`, `observation_seed/axial_phase.csv`, and constructs a periodic prior.

- [x] **Step 2: Declare the eight-dimensional state and fixed features**

Update the asset descriptor with this contract shape:

```json
{
  "geometry_id": "asset:fixed_axial_rigid",
  "state_contract": {
    "spec_id": "rigid7_scale:fixed_axial_asset",
    "state_model": "rigid_se3_scale",
    "dofs": [
      {"dof_id": "root.translation", "kind": "translation", "fields": ["tx", "ty", "tz"]},
      {"dof_id": "root.rotation", "kind": "rotation_so3", "fields": ["qw", "qx", "qy", "qz"]},
      {"dof_id": "scale", "kind": "scalar", "field": "scale", "bounds": [0.75, 1.5], "default": 1.0, "observable": false}
    ]
  },
  "initializer": {
    "kind": "axial_rigid_feature_correspondence",
    "body_center_role": "object_center",
    "body_mask_role": "object_body_mask",
    "depth_role": "object_center_depth",
    "off_axis_feature_role": "handle_center"
  }
}
```

Also declare the body radius/height proxy, local symmetry axis, and fixed local points for `object:center`, `object:handle`, and all contact feature IDs used by `object_contact_points.csv`.

- [x] **Step 3: Make periodic-factor selection capability-driven**

In `core/factors/adapters.py`, replace the object-name condition with a state-contract query equivalent to:

```python
periodic_dofs = tuple(
    dof for dof in _asset_state_dofs(profile)
    if dof.get("kind") == "periodic"
)
periodic_phase_prior_enabled = bool(periodic_dofs) and (
    "periodic_phase_prior" in configured_sequence_factors
)
```

Guard the existing periodic-factor construction block with `periodic_phase_prior_enabled`. For the fixed-rigid mug descriptor this produces no periodic production factor.

- [x] **Step 4: Verify the contract boundary**

Run the asset-contract builder and assert from its printed record:

```text
state width = 8
periodic DOF count = 0
initializer = axial_rigid_feature_correspondence
```

- [x] **Step 5: Commit the contract slice**

```bash
git add scripts/shared/generic_contact_pipeline/configs/assets/mug_periodic_rigid.json \
  scripts/shared/generic_contact_pipeline/configs/cases/mug.yaml \
  scripts/shared/generic_contact_pipeline/core/factors/adapters.py \
  scripts/shared/generic_contact_pipeline/core/state/asset_state_contract.py
git commit -m "refactor: model axial assets as fixed rigid states"
```

### Task 2: Implement the generic axial-rigid initializer

- [x] **Step 1: Add descriptor-backed initialization inputs**

Extend `InitializationRequest` usage so `axial_rigid_feature_correspondence` receives typed center, mask, depth, visibility, and fixed off-axis feature measurements plus the descriptor-built `RigidFeatureGeometryProvider`.

- [x] **Step 2: Generate finite physical pose hypotheses**

Implement:

```python
def _axial_rigid_feature_correspondence(
    request: InitializationRequest,
) -> InitializationResult:
    """Fit body translation/scale/tilt and compose axial feature angle into root SO(3)."""
```

For each frame:

```text
body center + metric depth -> translation
body proxy bbox            -> scale and tilt hypotheses
off-axis feature pixel     -> axial-angle hypotheses
root rotation              -> R_tilt @ R_axis(angle), stored once as quaternion
```

Use a fixed descriptor-declared angle grid and retain a finite number of lowest-reprojection hypotheses. Select a temporally consistent sequence with wrapped angular transition cost. Do not call an object-named solver and do not read seed pose/phase files.

- [x] **Step 3: Handle hidden feature intervals without fake observations**

If no visible feature measurement exists for a frame, do not create a pixel target. Initialize the state by quaternion interpolation or nearest valid hold, then allow contact/temporal factors to determine the final hidden-frame state.

The hypothesis ledger must record:

```python
{
    "initializer_kind": "axial_rigid_feature_correspondence",
    "visible_feature_frame_count": 131,
    "hidden_feature_frame_count": 109,
    "fabricated_feature_measurement_count": 0,
    "baseline_pose_read": False,
    "historical_phase_read": False,
    "case_dispatch_used": False,
    "human_state_optimized": False,
}
```

- [x] **Step 4: Route production preparation by capability**

In `prepare_capability_object_problem()`, include `axial_rigid_feature_correspondence` in the descriptor-contract path. Build `RigidFeatureGeometryProvider` from descriptor semantic points with `scale_state_index=7`; do not install `PeriodicFeatureRule`.

- [x] **Step 5: Run preparation-only verification**

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/prepare_generic_object_problem.py \
  --case mug --result-name benchmark_vlm_qwen \
  --output /tmp/audiohoi-mug-fixed-rigid/preparation.json
```

Expected: 240 states, width 8, no periodic factor, no solved-pose/phase read, and no case dispatch.

- [x] **Step 6: Commit the initializer slice**

```bash
git add scripts/shared/generic_contact_pipeline/core/solver/capability_initializers.py \
  scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py \
  scripts/shared/generic_contact_pipeline/core/state/geometry_provider.py
git commit -m "feat: initialize fixed axial assets from typed features"
```

### Task 3: Compile visible and occluded-hold factors

- [x] **Step 1: Enable typed measurement factors**

Configure:

```yaml
measurement_factors: [point_reprojection, metric_depth]
measurement_roles:
  point_reprojection: [object_center, handle_center]
  metric_depth: [object_center_depth]
sequence_factors:
  - temporal_velocity
  - temporal_acceleration
  - static_freeze
  - contact_relative_velocity
```

Keep `contact_distance` from the existing typed contact constraints.

- [x] **Step 2: Preserve per-measurement visibility**

The point factor input contains exactly the available typed rows. Center observations remain active according to object visibility. Handle rows exist only for visible measurements; the 81 hidden frames contribute zero handle reprojection rows, not zero-valued targets.

- [x] **Step 3: Activate persistent grasp mechanics**

Compile contact distance and relative velocity from the same fixed handle/contact feature and exact-frame read-only hand-site track. For hidden persistent-grasp frames:

```text
handle reprojection: inactive because no measurement exists
contact distance: active/downweighted from interaction provenance
contact relative velocity: active/downweighted from interaction provenance
temporal factors: active according to motion mode
```

- [x] **Step 4: Inspect initial residual provenance**

Verify selected factor IDs, row counts, input measurement IDs, visibility gates, and absence of `periodic_phase_prior:observation_seed_axial_phase`.

- [x] **Step 5: Commit the factor slice**

```bash
git add scripts/shared/generic_contact_pipeline/configs/cases/mug.yaml \
  scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py \
  scripts/shared/generic_contact_pipeline/core/solver/problem_factory.py \
  scripts/shared/generic_contact_pipeline/core/solver/residual_inputs.py
git commit -m "feat: constrain hidden rigid assets with persistent contact"
```

### Task 4: Solve and diagnose the full sequence

- [x] **Step 1: Run an isolated solve**

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/prepare_generic_object_problem.py \
  --case mug --result-name benchmark_vlm_qwen \
  --output /tmp/audiohoi-mug-fixed-rigid/preparation.json \
  --solve --candidate-dir /tmp/audiohoi-mug-fixed-rigid --max-nfev 200
```

- [x] **Step 2: Audit physical and occlusion metrics**

Record:

```text
quaternion norm error
scale min/max
visible handle reprojection median/p95/max
hidden-frame contact gap median/p95/max
hidden-frame hand/object relative velocity median/p95/max
translation acceleration p95/max
rotation step p95/max
reappearance correction at every hidden -> visible boundary
```

- [x] **Step 3: Localize any failure before changing code**

For each failed metric, inspect initial state, factor residuals, activation rows, and solved state at the same frames. Change only a generic capability or descriptor value justified by asset geometry. Do not add a mug branch, change global gate thresholds, or reintroduce pose/phase seeds.

- [x] **Step 4: Verify source boundaries**

```bash
git diff -U0 -- scripts/shared/generic_contact_pipeline/core | \
  rg '^\+.*(profile\.case_name|case_name ==|mug)' && exit 1 || true
```

Expected: no new object-name dispatch in core.

### Task 5: Decode, render, and publish only after review

- [x] **Step 1: Provide phase-zero render compatibility**

The root quaternion already contains the axial angle. Generate a derived render-only zero-relative-phase track, or update the Articraft render adapter to use zero relative phase for a fixed-rigid state. The artifact is never a solver input.

- [x] **Step 2: Render all 240 frames**

Produce:

```text
object_only/overlay.mp4
object_only/camera3d.mp4
object_only/side_yz.mp4
with_human/overlay.mp4
with_human/camera3d.mp4
with_human/side_yz.mp4
keyframes_montage.png
```

The skeleton is read-only relationship visualization.

- [x] **Step 3: Verify render artifacts**

Use `ffprobe` to confirm 240 frames at 24 fps and full decode. Record SHA-256 for all six videos.

- [ ] **Step 4: Request user visual acceptance**

Provide the six paths and call out hidden spans, reappearance boundaries, persistent hand contact, handle/body rigidity, jitter, and final support state. Do not publish canonical output before explicit acceptance unless every established automatic gate passes and promotion was already authorized.

- [ ] **Step 5: Publish through the sole publisher after authorization**

Use `AcceptedObjectOutputPublisher`; preserve automatic gate results and any manual override provenance. Verify 240 canonical rows, `source=generic_sequence_executor`, solve attempt ID, accepted SHA-256, `case_dispatch_used=false`, and `human_state_optimized=false`.

- [ ] **Step 6: Maintain the main plan and commit**

Update `docs/interaction_state_conditioned_generic_solver_plan.md` with capabilities, solve metrics, render hashes, promotion state, and remaining zero-shot gaps. Stage only task-owned files and commit locally; do not push.
