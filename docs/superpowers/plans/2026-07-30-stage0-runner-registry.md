# Fresh Stage 0 Runner Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize every required object-solver preprocessing role from `video.mp4` and configured assets/environments in a fresh result directory, then publish typed IR with complete provenance.

**Architecture:** A declarative task registry resolves artifact-role dependencies into a DAG. Each task runs with the Python selected by `runtime_envs.yaml`, validates outputs, hashes inputs/outputs/models, and publishes a single Stage 0 manifest; typed publishers sit after model tasks and before `GenericSequenceExecutor` preparation.

**Tech Stack:** Python subprocess/dataclasses, YAML runtime config, ffmpeg/OpenCV, existing SAM2/CoTracker/GVHMR/audio code, Depth Anything 3 API, JSON/JSONL/CSV provenance.

**Verification constraint:** Add no test files; use direct DAG smokes, existing validation commands, fresh-directory execution, and full-video decoding.

---

### Task 1: Preprocess task contracts and DAG

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/__init__.py`
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/types.py`
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py`
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/runtime.py`

- [ ] **Step 1: Define task and attempt types**

```python
@dataclass(frozen=True)
class PreprocessTaskSpec:
    task_id: str
    runtime_env: str
    input_roles: tuple[str, ...]
    output_roles: tuple[str, ...]
    required: bool
    command_builder: Callable[[PreprocessContext], tuple[str, ...]]
    validator: Callable[[PreprocessContext], Mapping[str, object]]

@dataclass(frozen=True)
class PreprocessTaskAttempt:
    task_id: str
    status: str
    command: tuple[str, ...]
    runtime_python: str
    input_sha256: Mapping[str, str]
    output_sha256: Mapping[str, str]
    error: str
```

- [ ] **Step 2: Implement role resolution/topological sort**

Reject duplicate producers, unknown roles, and cycles. Reuse only when the previous task manifest has identical input hashes and every output validator passes.

- [ ] **Step 3: Add dry-run CLI**

Create `tools/plan_stage0.py --case <name> --result-name <name>` returning ordered tasks and `reuse/run/blocked` decisions without mutation.

- [ ] **Step 4: Verify five-case DAGs and commit**

Run dry-run for all five configs, assert no cycle/duplicate producer, then commit core registry and CLI.

### Task 2: Register extraction, SAM2, and CoTracker

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/tasks.py`
- Modify: `scripts/shared/generic_contact_pipeline/stages/main/stage0_preprocess.py`

- [ ] **Step 1: Register frame/audio extraction**

Use ffmpeg with explicit output directories and validate sequential frame numbering, nonzero frame count, FPS, and WAV duration.

- [ ] **Step 2: Wrap existing SAM2/CoTracker commands**

Move current command construction into task specs. SAM2 depends on frames; CoTracker depends on frames and masks. Preserve explicit detector prompt/box configuration.

- [ ] **Step 3: Replace ad-hoc Stage 0 branching**

`stage0_preprocess.run()` creates a context, resolves required roles, executes the registry, and writes task attempts. It must not contain `_maybe_run_*` model-specific branches.

- [ ] **Step 4: Dry-run/reuse verification and commit**

Run on an existing case, confirm existing valid outputs are reused; run in a temporary result root and confirm extraction/SAM2/CoTracker are planned in dependency order.

### Task 3: DA3 repository-owned runner

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/run_da3_scene_depth.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/preprocess/tasks.py`

- [ ] **Step 1: Implement explicit DA3 CLI**

Load `depth_anything_3.api.DepthAnything3`, model name/path and process resolution from profile configuration, process ordered frames in bounded chunks, and atomically write per-frame NPY, `index.csv`, and `meta.json`.

- [ ] **Step 2: Register `da3_scene_depth`**

Use `runtime_python("da3")`; validate one finite depth file per frame, consistent shapes, model metadata, and index/frame alignment.

- [ ] **Step 3: Smoke with a bounded frame subset**

Run 2 frames into `/tmp`, validate outputs, then run the full selected canonical case only when the smoke passes.

- [ ] **Step 4: Commit**

Commit wrapper, registry entry, and documented runtime provenance.

### Task 4: GVHMR and audio event tasks

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/preprocess/tasks.py`
- Create: `scripts/shared/generic_contact_pipeline/tools/publish_audio_events.py`

- [ ] **Step 1: Register existing GVHMR runner**

Use the configured `gvhmr` Python and `run_gvhmr_local.py --sample-dir`. Validate `result.pkl`, frame count, camera matrices, finite SMPL parameter arrays, and required body model hashes. Record read-only role; do not modify human code.

- [ ] **Step 2: Register audio detection and typed publication**

Run `src.audio` in the `audiohoi` environment. Convert detector events into `AudioEventIR` rows with event type `unknown` unless classification evidence supports a fixed vocabulary value. Write `results/events/audio_events.csv` and `audio_events.jsonl`.

- [ ] **Step 3: Respect ablation disable**

When audio is disabled, emit a valid task attempt with `status=disabled`, no fake event file, and no blocking error for the optional audio role.

- [ ] **Step 4: Verify existing and fresh runs, then commit**

Check reuse on an existing case and execute GVHMR/audio in an isolated sample/result directory when outputs are absent.

### Task 5: Typed IR publication

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/publish.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/preprocess/tasks.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`

- [ ] **Step 1: Publish typed artifacts**

After model tasks, adapt validated observations/tracks/depth/GVHMR sites/audio into `measurements.jsonl`, `contact_constraints.jsonl`, `audio_events.jsonl`, and `interaction_timeline.jsonl`. Every row carries source artifact hash and producer task attempt ID.

- [ ] **Step 2: Add manifest role resolver to solver preparation**

Production preparation resolves typed artifact paths from `stage0_inputs_manifest.json`. If the manifest declares typed roles, it must not fall back to result-local legacy paths. Legacy fallback remains compatibility-only and is explicitly recorded.

- [ ] **Step 3: Validate schema and provenance**

Fail when frames/timestamps disagree, missing/occluded values are encoded as zeros, provenance hashes do not resolve, or any typed row references another case directory.

- [ ] **Step 4: Commit**

Commit typed publisher, solver resolver, and updated Stage 0 manifest schema.

### Task 6: Fresh-directory end-to-end object preparation

**Files:**
- Modify: `docs/interaction_state_conditioned_generic_solver_plan.md`

- [ ] **Step 1: Create an isolated result root**

Use a canonical sample's `video.mp4`, metadata, asset/profile config, installed models, and an empty result directory. Do not copy result artifacts.

- [ ] **Step 2: Run Stage 0**

Execute registered tasks to materialize required roles. Record elapsed times, reuse decisions, commands, runtime environments, and hashes.

- [ ] **Step 3: Prepare and solve object candidate**

Run typed preparation and the same generic executor. Assert `case_dispatch_used=false`, `baseline_pose_read=false`, `human_state_optimized=false`.

- [ ] **Step 4: Render and verify**

Generate object and skeleton-plus-object full videos; verify frame count/duration. Do not promote without explicit visual approval.

- [ ] **Step 5: Update plan and commit**

Document exact completed roles and any external model/checkpoint blocker. Commit code/config/docs and stable provenance artifacts locally; do not push.
