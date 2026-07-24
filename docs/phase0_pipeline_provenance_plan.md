# Phase 0: Pipeline Provenance Foundation

Branch: `refactor/pipeline-provenance-foundation`
Base: `53ae3a3470cbef0df47f5d4d8427328c85566627`

## Goal

Freeze reproducible evidence for the basketball, football, mug, chair, and stick
canonical runs before later structural refactors.

## Constraints

- Work only in `/mnt/hdd/AudioHOI-refactor`.
- Do not change losses, thresholds, or solver algorithms.
- Do not begin artifact-store, typed-contracts, or capability-plugin refactors.
- Preserve the original worktree and its uncommitted evaluation files.

## Steps

| Step | Status | Verification | Files |
| --- | --- | --- | --- |
| Isolate branch/worktree at the requested base | done | `git worktree list`; `git rev-parse HEAD` | worktree metadata only |
| Repair pytest collection and runtime environment | done | `pytest.ini` limits collection to `tests/`; generated-data assertion skips cleanly when its untracked inputs are absent | `pytest.ini`, `tests/test_final_hoi_evaluation.py` |
| Define a five-case golden manifest | done | canonical case/order/schema test | `tests/golden/pipeline_v1_five_cases.json` |
| Record inputs, per-stage artifacts, contact/gate state, pose, decoded render, and output hashes | done | capture plus byte/decoded verification CLI | `core/provenance/golden.py`, `tools/manage_golden_manifest.py` |
| Align ablation flags with actual consumers and recorded mechanisms | done | unknown/dead flags rejected; historical unsupported variants are labeled | `core/base/ablation.py`, ablation registry/runner |
| Add stage attempt/provenance records for reruns | done | rerun test produces immutable `000001.json`/`000002.json` with parent linkage | `core/provenance/attempts.py`, `run_pipeline.py` |
| Run focused and full regression checks; document results | done | commands and results below | tests and this document |

## Golden Contract

The manifest freezes the existing `benchmark_vlm_qwen` directory for each of the
five cases. It records:

- repository-relative input paths, presence, sizes, SHA-256, and recursive
  directory hashes;
- historical launch metadata exactly as recorded in each pipeline manifest
  (the result-directory name is not treated as proof that VLM was enabled);
- selected Stage -1 through Stage 5 artifacts, including CSV columns and row
  counts;
- contact candidates, anchor/contact state, VLM gates, and stage-audit gates;
- final pose/phase files; and
- both container-file SHA-256 and decoded RGB24 SHA-256 for all six canonical
  render videos per case.

Capture is an explicit operation. Normal verification never rewrites the
golden file.

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/manage_golden_manifest.py --verify \
  --input-root /mnt/hdd/AudioHOI
```

`--input-root` is read-only and supplies generated/ignored DA3, GVHMR, SAM2,
tracking, event, and legacy seed inputs that are intentionally absent from the
new worktree. Use `--skip-decoded-renders` for a fast file/hash check. Use
`--capture` only after intentionally accepting a new baseline.

Hydrate a future worktree from a known canonical source with a dry run first:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/sync_golden_inputs.py \
  --source-root /mnt/hdd/AudioHOI

/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/sync_golden_inputs.py \
  --source-root /mnt/hdd/AudioHOI --apply
```

The sync refuses to overwrite a destination whose hash differs from the
golden manifest.

## Stage Attempt Layout

Each executed stage writes append-only completed records under:

```text
<result>/provenance/stages/<stage>/attempts/000001.json
<result>/provenance/stages/<stage>/attempts/000002.json
<result>/provenance/stages/<stage>/active_attempt.json
```

An audit-triggered rerun receives a new attempt id and records the triggering
attempt as `parent_attempt_id`. Each record contains before/after artifact
hashes and changed paths. Existing stage output locations remain unchanged in
Phase 0.

## Verification Log

- `python -m compileall -q scripts/shared/generic_contact_pipeline tests`: pass.
- `pytest --collect-only -q`: repository-owned tests only.
- `pytest -q`: `37 passed, 2 skipped` in the hydrated worktree. One skip is the
  pre-existing optional render-mask path; one is the generated final-result
  contact-data check whose inputs are not tracked by Git.
- Golden verification without decoded video: pass for five cases.
- Golden verification with all 30 videos decoded to RGB24: pass for five cases.
- Golden input sync audit: 68 inputs verified locally, 0 copies required, 0 errors.
- `git diff --check`: pass.

No loss, threshold, or solver implementation was changed.

## Continuation Audit

| Step | Status | Evidence |
| --- | --- | --- |
| Audit golden verifier coverage and failure behavior | done | verifier detects missing external roots, changed hashes, CSV/schema changes; sync refuses overwrite conflicts |
| Exercise provenance through the real `run_case` orchestration with isolated fake stages | done | smoke test runs under the `audiohoi` runtime and records a linked two-attempt audit rerun |
| Re-run complete Phase 0 regression and inspect solver/loss diff boundary | done | 39 collected; 37 passed, 2 environment/data skips; decoded golden verification passed; no component/stage solver/loss files changed |
| Commit the Phase 0 foundation as one reviewable checkpoint | done | commit subject: `Add pipeline provenance regression foundation` |
| Create the next isolated artifact-store branch/worktree from the Phase 0 checkpoint | in_progress | pending |
