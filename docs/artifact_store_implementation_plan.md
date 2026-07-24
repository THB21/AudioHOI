# Artifact Store Implementation Plan

Branch: `refactor/artifact-store`
Base: Phase 0 commit `ebb69e8b` (`Add pipeline provenance regression foundation`)

## Goal

Add immutable, content-addressed snapshots for stage-attempt artifacts while
preserving every existing canonical pipeline path and solver behavior.

## Constraints

- Do not change losses, thresholds, optimization algorithms, or stage schemas.
- Do not move or rename current stage outputs.
- Do not begin typed-contract or capability-plugin refactors.
- Artifact capture must be additive: a failure to persist provenance may fail
  the run loudly, but it must never silently substitute different solver data.
- Every change must pass the Phase 0 five-case golden verifier.

## Steps

| Step | Status | Evidence | Files |
| --- | --- | --- | --- |
| Hydrate ignored canonical inputs in the new worktree | done | 41 existing inputs verified, 27 copied, 0 errors | ignored data only |
| Define a content-addressed artifact-store API and layout | done | deduplication and immutable-blob tests | `core/provenance/artifact_store.py` |
| Link completed stage attempts to immutable stored artifacts | done | orchestration smoke test verifies stored references | `core/provenance/attempts.py` |
| Add artifact-store integrity verification | done | corruption and missing-reference tests | `tools/verify_artifact_store.py` |
| Run full tests and five-case decoded golden verification | done | 40 collected; 38 passed, 2 environment/data skips; 68 inputs and 30 decoded renders verified | this document |
| Commit the artifact-store checkpoint | done | commit subject: `Add immutable stage artifact store` | branch checkpoint |

## Intended Layout

```text
<result>/provenance/artifact_store/sha256/<first-two>/<sha256>
<result>/provenance/stages/<stage>/attempts/<id>.json
```

Attempt records will continue to contain before/after canonical-path hashes and
will additionally reference stored blobs by digest. Canonical files such as
`object_pose.csv` remain where existing stages and evaluators expect them.

## Verification Log

- Artifact-store focused tests: 7 passed.
- Full repository tests: 38 passed, 2 environment/data skips.
- Golden input sync audit: 68 verified, 0 copies required, 0 errors.
- Five-case golden verification including 30 decoded RGB24 videos: pass.
- `compileall` and `git diff --check`: pass.
- No component, main-stage solver, loss, threshold, or optimization file changed.
