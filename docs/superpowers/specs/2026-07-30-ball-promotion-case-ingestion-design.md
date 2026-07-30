# Basketball/Football Promotion and Case-Ingestion Design

## Goal

Close two currently blocking gaps without changing the continuous solver:

1. publish basketball and football canonical object trajectories from the existing capability-conditioned `GenericSequenceExecutor`; and
2. make every newly registered case prepare all required object-side inputs before it is considered ingestible.

The scope remains object reconstruction. GVHMR produces a read-only skeleton used by object contact factors and relation renders. No human refinement, human trajectory publication, or downstream human handoff is added.

## Current State

- Mug, chair, and stick canonical `object_pose.csv` files are published by `AcceptedObjectOutputPublisher` with `source=generic_sequence_executor`.
- Basketball and football already have capability-based sphere candidates, but their canonical files still record `source=sequence_se3_optimizer`.
- `stage0_preprocess.py` can generate missing SAM2 and CoTracker artifacts. DA3, GVHMR, and audio events are still reported as `external_*_required`, even though the repository has an audio pipeline, a GVHMR runner, configured Conda environments, and a local ignored Depth-Anything-3 checkout.
- A newly copied worktree does not automatically contain ignored third-party DA3 code. The ingestion contract must therefore validate the configured DA3 root and fail clearly when that runtime resource has not been installed or copied.

## Architecture

### Case ingestion is the preparation boundary

`Stage 0` becomes the case-ingestion entrypoint rather than a passive missing-file reporter. A case is ready for the solver only after this DAG succeeds:

```text
video.mp4 + case config
  -> frames + audio.wav
  -> SAM2 masks
  -> CoTracker object tracks
  -> DA3 scene depth
  -> GVHMR read-only skeleton
  -> generic audio events
  -> contract validation + ingestion manifest
```

The solver does not silently generate these inputs midway through Stage 1–4. It consumes an accepted ingestion manifest or fails before creating a solve attempt.

### Task registry

Add a focused preprocessing registry under `core/preprocess/`:

- `types.py`: immutable task specification, input/output artifact declarations, execution result, cache status, and failure status.
- `registry.py`: registers the fixed generic tasks and resolves the dependency DAG.
- `runner.py`: selects the Python interpreter through `runtime_envs.yaml`, evaluates cache hashes, executes missing tasks, validates declared outputs, and writes task provenance.
- `manifest.py`: creates and validates the atomic case-ingestion manifest.

Task selection depends on declared capabilities and required artifacts, never on `case_name == basketball/football/...`.

### Existing runner adapters

Small tools normalize existing code into stable task contracts:

- frame/audio extraction uses ffmpeg and writes deterministic frame/audio artifacts;
- SAM2 and CoTracker reuse `run_sam2_object.py` and `run_cotracker_object_points.py` in the `audiohoi` environment;
- DA3 uses the `da3` environment and the configured `third-party/Depth-Anything-3` checkout, exports dense depth, then normalizes it to `results/da3/scene_depth/<frame>.npy` plus `index.csv`;
- GVHMR invokes `scripts/shared/human/gvhmr/run_gvhmr.py` in the `gvhmr` environment and validates `results/gvhmr/result.pkl` frame count and required camera/SMPL fields;
- audio invokes the generic `src.audio` detector in the `audiohoi` environment and normalizes events to `results/events/audio_events.csv` and typed `AudioEventIR` fields.

The adapters may translate schemas and file layouts, but they must not solve object pose or add object-name-specific thresholds.

## Cache and Provenance

Each task cache key is the SHA-256 of:

- task schema version;
- input artifact hashes;
- relevant case-config subtree;
- command and arguments;
- Python executable and environment record;
- runner source hash;
- model/checkpoint identity when locally available.

A cache hit is valid only when every declared output exists and matches its recorded hash. A stale or partial output is regenerated into an isolated temporary directory and atomically installed after validation.

The accepted manifest is written under the selected result directory and records:

- task dependency order;
- started/completed/reused/failed status;
- command without secrets;
- runtime environment;
- input and output hashes;
- frame count and FPS;
- model/checkpoint identity;
- failure message;
- `human_state_role=read_only_observed` for GVHMR.

## Failure Semantics

- Missing `video.mp4`, missing case config, missing required asset, or an unavailable configured environment fails ingestion before any solver attempt.
- An absent ignored DA3 checkout reports the exact expected path and installation/copy requirement; it is not recorded as a valid `missing` artifact.
- A task process failure retains its log and leaves canonical outputs unchanged.
- Output frame-count mismatch, empty CSV, corrupt PKL/NPY, or missing required fields fails contract validation.
- Audio can be intentionally disabled only through the existing `disable_audio_events` ablation flag; otherwise missing audio events are a failed required task.
- Existing valid artifacts are never silently replaced; hash mismatch triggers a new isolated attempt and atomic replacement only after validation.

## Basketball and Football Promotion

For each ball case:

1. materialize a fresh capability-conditioned problem from its current typed inputs;
2. run the same `GenericSequenceExecutor` used by mug/chair/stick;
3. persist the isolated attempt ledger, states, residuals, hard metrics, and VLM gate record;
4. render the full object-only overlay and camera-3D videos, plus the read-only GVHMR skeleton relation views;
5. verify full video decode, frame count, projection/contact metrics, and absence of case dispatch or human optimization;
6. publish through the unique `AcceptedObjectOutputPublisher` only after the existing numeric gates and explicit user authorization are recorded.

No basketball/football solver, threshold override, or ball-only residual is introduced. Sphere geometry and initialization remain capability-selected.

## Verification Without New Test Files

Per the project constraint, this work does not add test files. Verification uses existing commands and one-off contract checks:

- import/compile the preprocessing modules in their declared environments;
- run ingestion against an isolated empty result directory for one existing case while reusing immutable sample inputs;
- run ingestion a second time and prove every task is a hash-valid cache hit;
- intentionally point a copied config at a missing DA3 root and verify a precise task failure without canonical writes;
- compare generated artifact frame counts and hashes with the manifest;
- full-decode all basketball/football promotion videos;
- verify both canonical CSVs have the expected row counts, `source=generic_sequence_executor`, a single solve attempt ID, candidate/accepted hash equality, `case_dispatch_used=false`, and `human_state_optimized=false`.

## Completion Criteria

- Five of five canonical object trajectories are published by `AcceptedObjectOutputPublisher` from `GenericSequenceExecutor` attempts.
- New-case ingestion prepares or hash-validates frames, audio, SAM2, CoTracker, DA3, GVHMR, and audio events before solving.
- `stage0_preprocess.py` contains no `external_da3_required`, `external_gvhmr_required`, or `external_audio_events_required` placeholders.
- A fresh result directory can reach an accepted ingestion manifest using only the sample video, case config, asset resources, installed runtime environments, and configured checkpoints.
- GVHMR remains read-only evidence, and no human refinement or human output contract is introduced.
- Chair/stick automatic gate gaps remain documented nonblocking exceptions and are not modified in this work.
