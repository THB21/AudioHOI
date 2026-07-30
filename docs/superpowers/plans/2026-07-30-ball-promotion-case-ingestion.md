# Ball Promotion and Case Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish basketball and football through the unified object solver and make new-case ingestion generate every required preprocessing artifact before solving.

**Architecture:** Add a capability-driven preprocessing DAG with hashed task manifests and environment-specific adapters, then make Stage 0 execute and validate that DAG. Separately rerun the existing sphere capability problems, render the complete object trajectories, and publish them through the unique atomic publisher after evidence review.

**Tech Stack:** Python dataclasses, pathlib, subprocess, hashlib/json/csv, ffmpeg/ffprobe, Depth-Anything-3, GVHMR, NumPy/OpenCV, existing GenericSequenceExecutor and AcceptedObjectOutputPublisher.

---

### Task 1: Materialize the local DA3 runtime resource

**Files:**
- Runtime resource: `third-party/Depth-Anything-3/` (Git-ignored; copied from `/mnt/hdd/AudioHOI/third-party/Depth-Anything-3/`)
- Inspect: `scripts/shared/generic_contact_pipeline/configs/runtime_envs.yaml`

- [ ] **Step 1: Verify source and target paths**

Run:
```bash
test -f /mnt/hdd/AudioHOI/third-party/Depth-Anything-3/src/depth_anything_3/cli.py
test ! -e third-party/Depth-Anything-3
```
Expected: both commands exit 0.

- [ ] **Step 2: Copy the ignored DA3 checkout without touching the original worktree**

Run:
```bash
cp -a /mnt/hdd/AudioHOI/third-party/Depth-Anything-3 third-party/Depth-Anything-3
```
Expected: the target CLI and model checkout exist; Git status remains unchanged because `third-party/` is ignored.

- [ ] **Step 3: Verify the configured environment can import the copied checkout**

Run:
```bash
PYTHONPATH="$PWD/third-party/Depth-Anything-3/src" \
  /home/yang/miniconda3/envs/da3/bin/python -c \
  'from depth_anything_3.api import DepthAnything3; print("da3_import_ok")'
```
Expected: `da3_import_ok`.

### Task 2: Define preprocessing task and manifest contracts

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/__init__.py`
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/types.py`
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/manifest.py`

- [ ] **Step 1: Add immutable task/result types**

Implement `ArtifactSpec`, `PreprocessTask`, `TaskExecutionRecord`, and `CaseIngestionResult`. `PreprocessTask` must declare `task_id`, `runtime_env`, dependencies, input/output paths, command builder, required flag, and config fingerprint. Reject duplicate task IDs, unsafe output paths, missing dependencies, and required tasks without outputs.

- [ ] **Step 2: Add canonical hash helpers**

Implement file SHA-256, deterministic directory-tree SHA-256, JSON canonical hashing, and a task cache-key function over schema version, input hashes, config fingerprint, command, runtime environment, runner source, and model/checkpoint identity.

- [ ] **Step 3: Add atomic manifest I/O**

Implement `write_ingestion_manifest_atomic()` using a temporary sibling and `os.replace`, plus `validate_ingestion_manifest()` that verifies every required task is `generated` or `reused`, every declared output hash matches, frame count/FPS are present, and GVHMR records `human_state_role=read_only_observed`.

- [ ] **Step 4: Compile and inspect the new contract modules**

Run:
```bash
/home/yang/miniconda3/envs/audiohoi/bin/python -m py_compile \
  scripts/shared/generic_contact_pipeline/core/preprocess/types.py \
  scripts/shared/generic_contact_pipeline/core/preprocess/manifest.py
```
Expected: exit 0.

- [ ] **Step 5: Commit the contracts**

```bash
git add scripts/shared/generic_contact_pipeline/core/preprocess
git commit -m "feat: define case ingestion contracts"
```

### Task 3: Add generic runner adapters

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/run_da3_scene_depth.py`
- Create: `scripts/shared/generic_contact_pipeline/tools/run_audio_event_extract.py`
- Reuse: `scripts/shared/human/gvhmr/run_gvhmr.py`
- Reuse: `scripts/shared/generic_contact_pipeline/tools/run_sam2_object.py`
- Reuse: `scripts/shared/generic_contact_pipeline/tools/run_cotracker_object_points.py`

- [ ] **Step 1: Implement DA3 normalization**

The tool must run DA3 `images <sample>/frames --export-format mini_npz`, load `exports/mini_npz/results.npz`, validate a finite `[N,H,W]` depth array, and atomically publish `results/da3/scene_depth/00001.npy...` plus `index.csv`. It must require exactly the sample frame count and record the source NPZ/model identity in a JSON summary.

- [ ] **Step 2: Implement audio-event normalization**

The tool must invoke the existing generic audio detector, then write the solver-compatible columns `event,audio_time,audio_frame,peak,prominence,rms_rise,sharpness,audio_score` to `results/events/audio_events.csv`. It must derive FPS from `video.mp4`, preserve detector provenance, reject non-finite fields, and write at least a valid header when a clip has no detected event.

- [ ] **Step 3: Validate adapter CLI boundaries**

Run:
```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_audio_event_extract.py --help
/home/yang/miniconda3/envs/da3/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_da3_scene_depth.py --help
```
Expected: both exit 0 and show `--sample-dir`.

- [ ] **Step 4: Commit the adapters**

```bash
git add scripts/shared/generic_contact_pipeline/tools/run_da3_scene_depth.py \
  scripts/shared/generic_contact_pipeline/tools/run_audio_event_extract.py
git commit -m "feat: add generic DA3 and audio ingestion adapters"
```

### Task 4: Build and execute the preprocessing DAG

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py`
- Create: `scripts/shared/generic_contact_pipeline/core/preprocess/runner.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/preprocess/__init__.py`
- Modify: `scripts/shared/generic_contact_pipeline/stages/main/stage0_preprocess.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/base/schema.py`

- [ ] **Step 1: Register the fixed generic DAG**

Register tasks in this dependency order: frame extraction, audio extraction, SAM2, CoTracker, DA3, GVHMR, audio-event extraction. Resolve commands with `runtime_python()` and case-config capabilities; do not branch on object or case names.

- [ ] **Step 2: Implement hash-valid cache execution**

For each task, compute its cache key, reuse only when all output hashes match the previous record, otherwise execute it in dependency order. Capture stdout/stderr, environment name, command, start/end timestamps, return code, input/output hashes, frame count, FPS, and failure reason. Stop at the first failed required task and never invoke Stage 1.

- [ ] **Step 3: Refactor Stage 0 into ingestion execution**

Replace `_maybe_run_sam2`, `_maybe_run_cotracker`, and all three `external_*_required` placeholders with `run_case_ingestion(profile)`. Continue writing the existing Stage 0 metrics/manifest paths for compatibility, and additionally write the accepted hashed ingestion manifest declared by `stage_paths()`.

- [ ] **Step 4: Enforce solver entry validation**

Before Stage 1, validate the accepted ingestion manifest. A missing, stale, or failed required task must raise with its `task_id`; `disable_audio_events` is the only accepted reason for an absent audio-event artifact.

- [ ] **Step 5: Compile the integrated path**

Run:
```bash
/home/yang/miniconda3/envs/audiohoi/bin/python -m py_compile \
  scripts/shared/generic_contact_pipeline/core/preprocess/registry.py \
  scripts/shared/generic_contact_pipeline/core/preprocess/runner.py \
  scripts/shared/generic_contact_pipeline/stages/main/stage0_preprocess.py \
  scripts/shared/generic_contact_pipeline/run_pipeline.py
```
Expected: exit 0.

- [ ] **Step 6: Commit the DAG**

```bash
git add scripts/shared/generic_contact_pipeline/core/preprocess \
  scripts/shared/generic_contact_pipeline/stages/main/stage0_preprocess.py \
  scripts/shared/generic_contact_pipeline/core/base/schema.py
git commit -m "feat: run required preprocessing during case ingestion"
```

### Task 5: Verify fresh ingestion and caching

**Files:**
- Generated only: isolated result and task-attempt directories under `/tmp/audiohoi-case-ingestion-*`
- Modify: `docs/interaction_state_conditioned_generic_solver_plan.md`

- [ ] **Step 1: Run ingestion in an isolated empty result context**

Use a temporary case-config override that points `result_name` to an empty result directory while retaining immutable sample video/assets. Run Stage 0 with the declared environment registry. Expected: every required task is generated or hash-valid reused from sample-level immutable inputs, and an accepted manifest is written.

- [ ] **Step 2: Run the identical ingestion again**

Expected: every task reports `reused`, commands are not rerun, and output hashes equal the first manifest.

- [ ] **Step 3: Verify precise DA3 resource failure**

Temporarily override the DA3 root to a nonexistent isolated path and run ingestion against another empty result. Expected: the DA3 task fails with the exact missing root, later tasks do not run, and no accepted ingestion manifest or canonical object pose is written.

- [ ] **Step 4: Record evidence and commit documentation**

Append task counts, cache results, hashes, and failure evidence to the maintained plan; do not add test files.

```bash
git add docs/interaction_state_conditioned_generic_solver_plan.md
git commit -m "docs: record fresh case ingestion evidence"
```

### Task 6: Produce fresh unified basketball and football attempts

**Files:**
- Generated: isolated attempt roots under `/tmp/audiohoi-unified-ball-promotion/`
- Modify after acceptance: `samples_known_object/01_basketball/results/benchmark_vlm_qwen/object_pose.csv`
- Modify after acceptance: `samples_known_object/10_football/results/benchmark_vlm_qwen/object_pose.csv`
- Create after acceptance: each result directory's `generic_object_publication.json`
- Persist after acceptance: each result directory's selected `generic_sequence_solver_attempts/<attempt-id>/`

- [ ] **Step 1: Solve basketball through the capability path**

Run `prepare_generic_object_problem.py --case basketball --result-name benchmark_vlm_qwen --solve` into an isolated candidate root without accepted-write authorization. Expected: one generic attempt, `case_dispatch_used=false`, `human_state_optimized=false`, and no canonical modification.

- [ ] **Step 2: Solve football through the identical capability path**

Use the same command and parameters except `--case football`. Expected: the same sphere capability initializer/factor vocabulary, no football solver, and no canonical modification.

- [ ] **Step 3: Evaluate and persist attempt evidence**

Verify solver termination, objective nonincrease, projection/contact gates, state row counts of 192 and 242, finite states, normalized quaternions where present, and candidate provenance. Persist factor ledger, hard metrics, residuals, states, and VLM gate records.

### Task 7: Render, review, and publish both balls

**Files:**
- Generated: `/tmp/audiohoi-unified-ball-promotion/renders/{basketball,football}/`
- Modify: two canonical `object_pose.csv` files and their publication records
- Modify: `docs/interaction_state_conditioned_generic_solver_plan.md`

- [ ] **Step 1: Render complete object evidence**

For each candidate generate object-only overlay/camera-3D and read-only GVHMR skeleton relation overlay/camera-3D videos. Renderers must consume the candidate CSV directly and must not optimize human state.

- [ ] **Step 2: Verify media and geometry evidence**

Use ffprobe to verify expected FPS/frame count and ffmpeg to full-decode all eight videos. Record SHA-256 hashes. Compare object centers/contact frames with the source overlays and verify hard metrics remain within the existing case-independent gates.

- [ ] **Step 3: Obtain explicit full-video acceptance**

Present both sets of videos to the user. Record the exact authorization text separately from automatic numeric gates; do not convert a failed automatic gate into a pass.

- [ ] **Step 4: Publish atomically through the unique publisher**

Call `AcceptedObjectOutputPublisher` for each accepted attempt. Persist the attempt directory and publication JSON. Verify candidate and accepted SHA-256 equality, canonical row counts, one `generic_solve_attempt_id`, `source=generic_sequence_executor`, `case_dispatch_used=false`, and `human_state_optimized=false`.

- [ ] **Step 5: Update maintained status and commit**

Record five-of-five generic canonical coverage, while keeping chair/stick automatic-gate exceptions explicitly nonblocking.

```bash
git add samples_known_object/01_basketball/results/benchmark_vlm_qwen \
  samples_known_object/10_football/results/benchmark_vlm_qwen \
  docs/interaction_state_conditioned_generic_solver_plan.md
git commit -m "data: publish unified basketball and football trajectories"
```

### Task 8: Final object-only verification

**Files:**
- Inspect only: five canonical result directories and Stage 0 manifests

- [ ] **Step 1: Verify five canonical publishers**

Run a one-off CSV/JSON audit over basketball, football, mug, chair, and stick. Expected: every canonical source is `generic_sequence_executor`, every result has one solve-attempt ID, and all publication records state `case_dispatch_used=false` and `human_state_optimized=false`.

- [ ] **Step 2: Verify ingestion definition of done**

Search production Stage 0 for `external_da3_required`, `external_gvhmr_required`, and `external_audio_events_required`. Expected: no matches. Validate the fresh-run accepted manifest and its second-run cache hits.

- [ ] **Step 3: Verify repository scope**

Run Python compilation, `git diff --check` excluding generated CRLF CSVs, and inspect staged paths. Confirm no human refinement/handoff code and no unrelated chair VLM, golden, or stick render files are staged.

- [ ] **Step 4: Commit any final documentation-only correction**

If the maintained plan needs a final factual correction, commit only that file with `docs: finalize case ingestion and five-case promotion evidence`.
