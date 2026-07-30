# Chair Articulated Initializer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace chair `legacy_state_artifact` initialization with a descriptor-selected, observation-derived articulated initializer consumed by the existing generic sequence executor.

**Architecture:** Parse the asset descriptor into a `StateSpec`, geometry provider, and initializer declaration. Build finite root/joint hypotheses from typed lines, contact-site correspondences, camera, and descriptor defaults; score them through generic residual evaluators and interpolate the selected sequence without reading any object pose CSV.

**Tech Stack:** Python dataclasses, NumPy/SciPy, JSON/YAML asset configuration, existing MeasurementIR/ContactConstraintIR/GeometryProvider/GenericSequenceExecutor.

**Verification constraint:** Add no test files; use existing suites, direct production smokes, artifact validation, and full-video decoding.

---

### Task 1: Descriptor-driven articulated StateSpec

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/state/asset_state_contract.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/state/__init__.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/assets/chair_articulated.json`

- [ ] **Step 1: Add the chair state contract declaration**

Declare root translation/quaternion, two bounded joints, initializer kind, joint defaults, semantic correspondence roles, and support features in the asset JSON. Do not include a case name or solved pose path.

```json
"state_contract": {
  "spec_id": "articulated6:urdf",
  "dofs": [
    {"dof_id": "root.translation", "kind": "translation", "fields": ["tx", "ty", "tz"]},
    {"dof_id": "root.rotation", "kind": "rotation_so3", "fields": ["qw", "qx", "qy", "qz"]},
    {"dof_id": "joint.front_to_rear", "kind": "joint_angle", "field": "rear_joint_angle", "default": 0.0, "bounds_from_urdf": "front_to_rear"},
    {"dof_id": "joint.front_to_seat", "kind": "joint_angle", "field": "seat_joint_angle", "default": 0.0, "bounds_from_urdf": "front_to_seat"}
  ]
},
"initializer": {
  "kind": "articulated_correspondence",
  "line_feature_roles": ["backrest:top_edge", "seat:front_edge", "leg:front_left", "leg:front_right", "leg:rear_left", "leg:rear_right"],
  "minimum_line_count": 2,
  "joint_hypotheses": "descriptor_default_and_observed",
  "missing_frame_policy": "interpolate_then_nearest_hold"
}
```

- [ ] **Step 2: Implement the asset state-contract parser**

Create these public types and function:

```python
@dataclass(frozen=True)
class AssetStateContract:
    state_spec: StateSpec
    initializer: Mapping[str, object]
    resource_path: str
    descriptor_sha256: str

def build_asset_state_contract(
    descriptor_path: Path,
    repository_root: Path,
) -> AssetStateContract:
    ...
```

The parser reads URDF joint limits for `bounds_from_urdf`, adds a positive camera-depth bound to root translation, validates unique DOF IDs/source fields, and rejects unknown kinds or absent URDF joints.

- [ ] **Step 3: Export and smoke the parser**

Run:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python - <<'PY'
from pathlib import Path
from scripts.shared.generic_contact_pipeline.core.state import build_asset_state_contract
c = build_asset_state_contract(Path('scripts/shared/generic_contact_pipeline/configs/assets/chair_articulated.json'), Path('.'))
assert c.state_spec.spec_id == 'articulated6:urdf'
assert tuple(d.dof_id for d in c.state_spec.dofs) == ('root.translation','root.rotation','joint.front_to_rear','joint.front_to_seat')
print('chair_state_contract_ok')
PY
```

Expected: `chair_state_contract_ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/shared/generic_contact_pipeline/core/state scripts/shared/generic_contact_pipeline/configs/assets/chair_articulated.json
git commit -m "feat: build articulated state from asset descriptor"
```

### Task 2: Generic capability initializer contract

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/solver/capability_initializers.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/__init__.py`

- [ ] **Step 1: Add initializer request/result contracts**

```python
@dataclass(frozen=True)
class InitializationRequest:
    state_spec: StateSpec
    geometry_provider: GeometryProvider
    measurements: tuple[Measurement, ...]
    contact_constraints: tuple[ContactConstraint, ...]
    human_sites: tuple[HumanSiteMeasurement, ...]
    cameras: Mapping[int, PinholeCamera]
    initializer: Mapping[str, object]

@dataclass(frozen=True)
class InitializationResult:
    states_by_frame: Mapping[int, tuple[float, ...]]
    template_rows: tuple[Mapping[str, object], ...]
    hypothesis_ledger: Mapping[str, object]
    input_artifact_ids: tuple[str, ...]
```

- [ ] **Step 2: Implement `articulated_correspondence`**

Use descriptor defaults for joint hypotheses, visible `Line2DMeasurement` semantics for image constraints, GVHMR palms only as read-only contact targets, and available metric depth/support evidence. Generate finite depth/orientation/joint hypotheses, score through existing line/contact/joint/support residual functions, select the lowest finite score per evidence interval, then linearly interpolate translation/joints and SLERP quaternion between observed frames.

The ledger must contain selected/rejected hypothesis IDs, input measurement IDs, score blocks, and `case_dispatch_used=false`. Raise `ValueError("articulated_correspondence has insufficient typed evidence")` instead of reading a pose fallback.

- [ ] **Step 3: Run a synthetic capability smoke**

Use two typed lines, two local feature segments, two cameras, and descriptor-default joints. Assert 3 frames are initialized, quaternions are normalized, and the ledger reports no baseline read.

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python scripts/shared/generic_contact_pipeline/tools/smoke_capability_initializer.py --geometry-descriptor scripts/shared/generic_contact_pipeline/configs/assets/chair_articulated.json --initializer articulated_correspondence
```

Expected: JSON containing `"case_dispatch_used": false`, `"baseline_pose_read": false`.

- [ ] **Step 4: Commit**

```bash
git add scripts/shared/generic_contact_pipeline/core/solver/capability_initializers.py scripts/shared/generic_contact_pipeline/core/solver/__init__.py scripts/shared/generic_contact_pipeline/tools/smoke_capability_initializer.py
git commit -m "feat: add generic articulated correspondence initializer"
```

### Task 3: Remove the chair legacy production branch

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/chair.yaml`
- Modify: `scripts/shared/generic_contact_pipeline/core/factors/adapters.py`

- [ ] **Step 1: Switch chair configuration**

Replace:

```yaml
initializer: legacy_state_artifact
initializer_artifacts: [object_pose_init.csv, object_contact_points.csv, object_observations.csv]
```

with:

```yaml
initializer: articulated_correspondence
initializer_artifacts: [object_local_segments.csv, object_contact_points.csv, object_observations.csv]
```

- [ ] **Step 2: Route all descriptor initializers through the registry**

Remove the `legacy_state_artifact` early return. Build `AssetStateContract`, geometry provider, GVHMR sites, cameras, then call:

```python
initialization = initialize_from_capabilities(
    InitializationRequest(
        state_spec=contract.state_spec,
        geometry_provider=provider,
        measurements=tuple(measurements),
        contact_constraints=tuple(constraints),
        human_sites=tuple(gvhmr_sites.measurements),
        cameras=cameras,
        initializer=contract.initializer,
    )
)
```

Populate `initializer_input_sha256` from typed input artifact hashes and the hypothesis ledger, never `object_pose_init.csv`.

- [ ] **Step 3: Remove production chair-name factor selection**

Replace `profile.case_name == "chair"` checks with capabilities derived from `StateSpec` (`DofKind.JOINT_ANGLE`) and descriptor feature availability. Chair-specific audit modules may remain compatibility-only but cannot be imported by production preparation.

- [ ] **Step 4: Verify no pose seed is opened**

Create a temporary result directory containing symlinks/copies of `object_observations.csv`, `object_contact_points.csv`, `object_local_segments.csv`, VLM artifacts if required, and no `object_pose*.csv`. Run preparation and assert:

```json
{"initializer_kind":"articulated_correspondence","baseline_pose_read":false,"case_dispatch_used":false}
```

- [ ] **Step 5: Commit**

```bash
git add scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py scripts/shared/generic_contact_pipeline/core/factors/adapters.py scripts/shared/generic_contact_pipeline/configs/cases/chair.yaml
git commit -m "refactor: remove chair legacy state initializer"
```

### Task 4: Isolated chair solve and render evidence

**Files:**
- Modify: `docs/interaction_state_conditioned_generic_solver_plan.md`
- Generate in isolated result/evidence directory; do not promote canonical without explicit approval.

- [ ] **Step 1: Run the same executor**

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python scripts/shared/generic_contact_pipeline/tools/prepare_generic_object_problem.py --case chair --result-name benchmark_vlm_qwen --output /tmp/audiohoi-chair-capability/preparation.json --solve --candidate-dir /tmp/audiohoi-chair-capability --max-nfev 200
```

- [ ] **Step 2: Validate evidence**

Check attempt status, hard metrics, factor ledger, no baseline reads, no human optimization, and normalized quaternions/joint bounds.

- [ ] **Step 3: Render candidate**

Use the configured URDF renderer with candidate pose, current sample frames, and read-only GVHMR skeleton. Verify all required MP4 files decode for the full frame count.

- [ ] **Step 4: Record and commit**

Append exact command, attempt ID, metrics, render hashes, and remaining automatic blockers to the master plan, then commit code/config/docs without promoting the candidate.
