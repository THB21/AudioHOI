# Generic Rigid-Mesh Visual Pose Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an isolated suitcase candidate from persistent SAM2/CoTracker observations and MegaPose RGB mesh-aware keyframe pose hypotheses without changing the accepted pose or adding a case-specific solver.

**Architecture:** Stage 0 keeps one stable CoTracker query set across the sequence and records visibility/provenance. A SAM-PT-style refinement task may use those tracks to improve visible masks. MegaPose runs as an external RGB rigid-mesh pose provider on reliable keyframes; typed hypotheses enter the existing generic sequence executor as measurements.

**Tech Stack:** Python 3.10/3.9, PyTorch, official CoTracker3, SAM2/SAM-PT method, MegaPose RGB multi-hypothesis, OpenCV, trimesh, existing AudioHOI typed IR and GenericSequenceExecutor.

---

## File map

- Modify `scripts/shared/generic_contact_pipeline/tools/run_cotracker_object_points.py`: persistent full-sequence query identities and new rigid-track artifacts.
- Modify `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py`: declare persistent-track artifacts and validation.
- Modify `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`: generic tracker/provider configuration only.
- Create `scripts/shared/generic_contact_pipeline/tools/run_sam_pt_mask_refine.py`: optional point-prompt mask refinement producing a separate candidate mask directory.
- Create `scripts/shared/generic_contact_pipeline/tools/export_rigid_asset_mesh.py`: convert a fixed-state URDF visual assembly into one mesh in millimetres for external pose providers.
- Create `scripts/shared/generic_contact_pipeline/tools/run_megapose_rigid_pose.py`: prepare official MegaPose input, run RGB multi-hypothesis inference, and normalize output into typed JSONL.
- Modify `scripts/shared/generic_contact_pipeline/configs/runtime_envs.yaml`: register the isolated MegaPose environment.
- Modify `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py`: register mesh export and MegaPose keyframe tasks after their standalone diagnostics pass.
- Create `scripts/shared/generic_contact_pipeline/core/measurements/pose_hypotheses.py`: parse and validate provider-neutral external SE(3) hypotheses.
- Modify `scripts/shared/generic_contact_pipeline/core/measurements/configured.py`: register the schema-capability adapter without `case_name` dispatch.
- Modify `scripts/shared/generic_contact_pipeline/core/factors/adapters.py`: compile the selected external SE(3) measurement through the existing pose-prior factor kind.
- Update this plan's checkboxes and `docs/generic_pipeline_v2_mainline_design_cn.md` with measured results only.

### Task 1: Freeze baseline and persistent-track contract

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/tools/run_cotracker_object_points.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [ ] **Step 1: Record the immutable accepted-pose hash**

Run:

```bash
sha256sum samples_known_object/15_suitcase_drag/results/pure_solver_no_audio_no_vlm/object_pose.csv
```

Expected hash:

```text
b86b0834d8c7b61687f8bef24c5d5f9afdaaddc35262d32a8b1ba4fdc7ea648c
```

- [ ] **Step 2: Replace chunk-local query initialization with one persistent query set**

Implement these focused helpers in `run_cotracker_object_points.py`:

```python
def sample_persistent_queries(mask: np.ndarray, *, grid_size: int) -> list[tuple[str, np.ndarray]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Empty SAM2 mask")
    x_grid = np.linspace(float(xs.min()), float(xs.max()), grid_size)
    y_grid = np.linspace(float(ys.min()), float(ys.max()), grid_size)
    queries: list[tuple[str, np.ndarray]] = []
    for row, y in enumerate(y_grid):
        for col, x in enumerate(x_grid):
            xi = int(np.clip(round(x), 0, mask.shape[1] - 1))
            yi = int(np.clip(round(y), 0, mask.shape[0] - 1))
            if mask[yi, xi] > 0:
                queries.append((f"grid_{row:02d}_{col:02d}", np.array([x, y], dtype=np.float64)))
    if len(queries) < 8:
        raise RuntimeError(f"Persistent rigid tracking requires at least 8 mask-interior queries, got {len(queries)}")
    return queries
```

Call official CoTracker3 offline once for the complete resized video:

```python
queries = torch.zeros((1, len(points), 3), dtype=torch.float32, device=device)
queries[0, :, 1:] = torch.from_numpy(points).to(device)
pred_tracks, pred_visibility = cotracker(video, queries=queries)
```

Do not loop over independent 32-frame ranges. Keep `track_id` unchanged for every output frame.

- [ ] **Step 3: Write the new persistent artifact without breaking legacy outputs**

Write `results/tracking/rigid_point_tracks.csv` with this exact schema:

```text
frame,time,track_id,query_frame,x,y,visible,confidence,semantic_feature_id,source,attempt_id
```

Write `results/tracking/rigid_point_tracks_manifest.json` containing tracker model, query mask hash, query count, query frame, frame count, resize scale, and `reinitialization_frames: []`. Derive legacy center and boundary CSVs from the persistent tracks for compatibility; do not assign fake local 3D coordinates.

- [ ] **Step 4: Register and validate artifacts**

Add `ArtifactSpec` entries and require:

```python
_validate_frame_csv(rigid_tracks.path, count, allow_multiple=True)
```

The manifest is a required file. Include `query_policy`, `grid_size`, and `sequence_mode: persistent_offline` in the task fingerprint.

- [ ] **Step 5: Run only Stage 0 tracking for suitcase**

Use the existing case-ingestion entry point with the suitcase config, forcing only the CoTracker task to regenerate. Expected evidence:

- 240 frames in `rigid_point_tracks.csv`;
- identical track-ID set across frames 127, 128, and 129, except points marked invisible rather than renamed;
- no reinitialization at frames 33, 65, 97, 129, 161, 193, or 225;
- accepted pose hash unchanged.

- [ ] **Step 6: Commit the persistent tracker change**

Stage only the three source/config files and commit:

```text
fix: keep rigid cotracker identities across sequence
```

### Task 2: Add SAM-PT-style mask refinement as isolated evidence

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/run_sam_pt_mask_refine.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [ ] **Step 1: Build visible positive/negative prompts from persistent tracks**

For each configured checkpoint frame, select visible tracked points inside the current mask as positive prompts. Select negative prompts from a narrow dilated ring outside the mask and from explicit human-occluder pixels when available. Preserve source point IDs in the manifest.

- [ ] **Step 2: Refine into a separate mask directory**

Use the existing SAM2 video predictor API and write only:

```text
results/segmentation/sam_pt_candidate_masks/{frame:05d}_mask.png
results/segmentation/sam_pt_candidate_manifest.json
```

Never overwrite `results/segmentation/masks` during diagnostics. Invisible points are omitted rather than treated as negative object evidence.

- [ ] **Step 3: Validate visible-mask behavior**

Generate contact sheets for frames 118, 125, 127, 128, 140, 154, 155, and 163. Confirm that original object pixels are retained, obvious person pixels are not absorbed, and the mask artifact records uncertainty during occlusion.

- [ ] **Step 4: Commit the isolated mask-refinement task**

```text
feat: add point-propagated sam2 mask candidate
```

### Task 3: Prepare the official MegaPose RGB runtime and fixed mesh

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/export_rigid_asset_mesh.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/runtime_envs.yaml`

- [ ] **Step 1: Install the official repository outside tracked source**

Clone `megapose6d/megapose6d` into `third-party/megapose6d`, create its official `megapose` conda environment from `conda/environment_full.yaml`, install the package editable, and download official MegaPose models. Record repository commit and model checksums in a runtime manifest; do not vendor MegaPose source into this commit.

- [ ] **Step 2: Export a fixed-state mesh**

Parse the asset descriptor and URDF visual geometry, apply the declared fixed joint state, concatenate all visual meshes in the root object coordinate frame, and export millimetre units. The command is:

```bash
python scripts/shared/generic_contact_pipeline/tools/export_rigid_asset_mesh.py \
  --asset-descriptor scripts/shared/generic_contact_pipeline/configs/assets/suitcase_fixed_rigid.json \
  --output samples_known_object/15_suitcase_drag/articraft/megapose/suitcase_fixed_mm.ply
```

The sidecar JSON records source URDF hash, fixed joint state, metre-to-millimetre scale, output mesh hash, bounds, and vertex/face counts.

- [ ] **Step 3: Verify asset geometry visually**

Render front, side, and top views of the exported mesh. Confirm constant handle extension, two rails, rigid body, and wheel support before any pose inference.

- [ ] **Step 4: Register the runtime**

Add a `megapose` environment entry pointing at `/home/yang/miniconda3/envs/megapose/bin/python`, with `megapose`, `torch`, `numpy`, `pandas`, `PIL`, and `trimesh` as required imports.

- [ ] **Step 5: Commit mesh export and runtime registration**

```text
feat: prepare generic rigid assets for megapose
```

### Task 4: Add MegaPose RGB as an external generic pose provider

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/run_megapose_rigid_pose.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [ ] **Step 1: Implement provider-neutral input preparation**

Read RGB frame, camera intrinsics, asset mesh, and typed mask bbox. Create official MegaPose input files with mesh units in millimetres and run `megapose-1.0-RGB-multi-hypothesis`. The runner accepts only generic arguments: sample directory, asset descriptor, keyframe list, mask source, camera JSON, model name, and output directory.

- [ ] **Step 2: Normalize official outputs**

Write `rigid_pose_hypotheses.jsonl` with frame, hypothesis ID, score, translation in camera metres, normalized quaternion, model/repository identity, frame/mask/camera/mesh hashes, and evidence paths. Preserve incompatible hypotheses separately; do not average rotations.

- [ ] **Step 3: Run visible keyframe diagnostics**

Run frames 1, 80, 106, 118, 125, 127, 155, 163, 200, and 226 first. Render each hypothesis as official MegaPose contour/mesh overlays. Frames with insufficient visible evidence are marked blocked instead of receiving a fabricated pose.

- [ ] **Step 4: Register the provider task only after standalone success**

The task depends on frames, masks, camera, and exported mesh. Its outputs remain under the isolated candidate directory and are not requirements for legacy five-case Stage 0.

- [ ] **Step 5: Commit the provider**

```text
feat: add megapose rgb rigid pose hypotheses
```

### Task 5: Consume external SE(3) hypotheses in the generic solver

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/measurements/pose_hypotheses.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/measurements/configured.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/factors/adapters.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [ ] **Step 1: Add a provider-neutral typed pose measurement**

Map every valid hypothesis to a full SE(3) pose measurement with confidence, frame interval, measurement IDs, input hashes, and provider provenance. The adapter key is capability-based (`external_rigid_pose_hypotheses`), never object- or case-named.

Use this immutable record at the adapter boundary:

```python
@dataclass(frozen=True)
class ExternalRigidPoseHypothesis:
    measurement_id: str
    frame: int
    hypothesis_id: str
    translation_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    confidence: float
    provider: str
    input_ids: tuple[str, ...]
    provenance: Mapping[str, object]
```

Reject non-finite translation, non-unit quaternion, confidence outside `[0, 1]`, duplicate measurement IDs, and frames outside the declared sequence.

- [ ] **Step 2: Compile pose-prior factors from selected hypotheses**

Use existing pose-prior residual machinery. Observation reliability selects active/downweighted/inactive tiers; no provider may directly supply solver weights.

- [ ] **Step 3: Run one isolated sequence attempt**

Keep support, penetration, persistent grasp, temporal velocity, and temporal acceleration active. Do not enable the rejected full-sequence linear-loss override. Write all state, factor ledger, hard metrics, and provenance under a new attempt ID.

- [ ] **Step 4: Render without publishing**

Generate ordinary overlay, VLM evidence overlay, object-only camera 3D, and human-skeleton HOI camera 3D. Compare against real pixels at the requested failure interval and verify the accepted pose hash is unchanged.

- [ ] **Step 5: Commit generic solver consumption**

```text
feat: consume external rigid pose hypotheses
```

### Task 6: Document measured result and promotion status

**Files:**
- Modify: `docs/generic_pipeline_v2_mainline_design_cn.md`
- Modify: this plan

- [ ] **Step 1: Record actual evidence**

Document tracker identity continuity, MegaPose keyframe success/failure counts, mesh/render hashes, candidate path, and accepted pose hash.

- [ ] **Step 2: Stop for visual approval**

Do not promote automatically. Present the isolated videos and keyframe contact sheet. Promotion requires explicit approval.

- [ ] **Step 3: Commit documentation**

```text
docs: record rigid mesh pose provider evidence
```
