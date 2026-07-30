# Mug Periodic Typed Initializer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move rigid-body and periodic-phase initialization from mug-named seed files and code into typed DOF measurements selected by the asset descriptor.

**Architecture:** Add a generic state-DOF measurement to MeasurementIR, adapt current parity inputs into that schema outside the solver, and initialize any descriptor-declared periodic rigid asset by DOF ID with wrapped interpolation. The generic executor and existing periodic factor remain unchanged.

**Tech Stack:** Python dataclasses, JSONL/CSV adapters, NumPy/SciPy rotations, existing StateSpec/PeriodicFeatureRule/GenericSequenceExecutor.

**Verification constraint:** Add no test files; use existing suites, typed round-trip smokes, isolated solves, and full-video decoding.

---

### Task 1: Add typed state-DOF measurements

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/measurements/types.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/measurements/__init__.py`
- Create: `scripts/shared/generic_contact_pipeline/core/measurements/state_dof.py`

- [ ] **Step 1: Define the measurement**

```python
@dataclass(frozen=True)
class StateDofMeasurement:
    meta: MeasurementMeta
    frame: int
    time: float
    dof_id: str
    values: tuple[float, ...]
    confidence: float
    interpolation_allowed: bool = True
```

Validate nonempty finite values, `[0,1]` confidence, and nonempty DOF ID.

- [ ] **Step 2: Add JSONL loader/publisher**

Implement `load_state_dof_measurements(path)` and `write_state_dof_measurements(path, measurements)` with provenance source fields and stable ordering by frame/DOF.

- [ ] **Step 3: Run schema smoke**

Round-trip translation, quaternion, scale, and periodic phase measurements and assert byte-stable second serialization.

- [ ] **Step 4: Commit**

```bash
git add scripts/shared/generic_contact_pipeline/core/measurements
git commit -m "feat: add typed state dof measurements"
```

### Task 2: Descriptor-declared periodic state

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/configs/assets/mug_periodic_rigid.json`
- Modify: `scripts/shared/generic_contact_pipeline/core/state/asset_state_contract.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/state/geometry_provider.py`

- [ ] **Step 1: Add periodic DOFs and observation roles**

The descriptor must declare root translation, root quaternion, scale, `assembly.phase`, state bounds, periodic axis/origin, feature rule, and initializer kind `periodic_rigid_observation`. Map observation roles to DOF IDs; do not mention mug.

- [ ] **Step 2: Generalize state contract parsing**

Parse scalar and periodic DOFs, validate periodic `[-pi,pi]` bounds, and return DOF-index lookup helpers. Build `RigidFeatureGeometryProvider` using descriptor feature points and `PeriodicFeatureRule` whose phase index is resolved from `assembly.phase` rather than hardcoded `8`.

- [ ] **Step 3: Smoke descriptor resolution**

Assert state width 9, phase index 8 obtained by ID, and a 90-degree phase rotates the declared periodic feature around the declared axis.

- [ ] **Step 4: Commit**

```bash
git add scripts/shared/generic_contact_pipeline/configs/assets/mug_periodic_rigid.json scripts/shared/generic_contact_pipeline/core/state
git commit -m "feat: describe periodic rigid state in asset manifest"
```

### Task 3: Generic periodic observation initializer

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_initializers.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/mug.yaml`

- [ ] **Step 1: Implement DOF assembly by ID**

Select `StateDofMeasurement` rows by descriptor-required DOF IDs. Normalize quaternion observations, linearly interpolate translation/scale, SLERP rotation, unwrap/interpolate/wrap periodic phase, and nearest-hold boundaries. Never index phase by numeric constant.

- [ ] **Step 2: Replace `_periodic_seed()`**

Delete the fixed reads of `observation_seed/body_pose.csv` and `observation_seed/axial_phase.csv`. The production function loads a configured typed artifact such as `state_dof_measurements.jsonl`, then calls `initialize_from_capabilities()`.

- [ ] **Step 3: Change case configuration**

Use:

```yaml
generic_object_problem:
  initializer: periodic_rigid_observation
  initializer_artifacts:
    - state_dof_measurements.jsonl
    - object_contact_points.csv
```

- [ ] **Step 4: Verify fixed paths are absent from production**

Run:

```bash
rg -n 'observation_seed/body_pose|observation_seed/axial_phase|mug_observation_seed|projected_periodic_sequence' scripts/shared/generic_contact_pipeline/core/{solver,state,factors}
```

Expected: no production matches.

- [ ] **Step 5: Commit**

```bash
git add scripts/shared/generic_contact_pipeline/core/solver scripts/shared/generic_contact_pipeline/configs/cases/mug.yaml
git commit -m "refactor: initialize periodic rigid assets from typed dofs"
```

### Task 4: Current-data parity adapter and isolated solve

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/publish_state_dof_measurements.py`
- Modify: `docs/interaction_state_conditioned_generic_solver_plan.md`

- [ ] **Step 1: Publish typed parity evidence outside solver**

The CLI accepts explicit `--body-pose-csv`, `--phase-csv`, and `--output`; it maps CSV columns to descriptor DOF IDs and records both input hashes. It contains no optimization and no case-name branch.

- [ ] **Step 2: Generate current mug typed measurements**

Run the CLI with the current two observation-seed artifacts and write `benchmark_vlm_qwen/state_dof_measurements.jsonl`.

- [ ] **Step 3: Prove seed independence**

Prepare from a temporary result directory containing the typed JSONL and required observation/contact artifacts but no `observation_seed` directory and no object pose CSV. Assert `baseline_pose_read=false` and initializer kind `periodic_rigid_observation`.

- [ ] **Step 4: Solve and render**

Run 200 evaluations through `prepare_generic_object_problem.py`, then decode object and GVHMR skeleton-plus-object full videos. Preserve automatic publication gates.

- [ ] **Step 5: Record and commit**

Commit typed input/provenance, code/config/docs, and isolated attempt evidence only when paths are repository-stable. Do not promote without explicit visual approval.
