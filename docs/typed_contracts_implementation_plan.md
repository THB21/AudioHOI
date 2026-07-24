# Typed Contracts Implementation Plan

Branch: `refactor/typed-contracts`
Base: artifact-store commit `9410fe56` (`Add immutable stage artifact store`)

## Goal

Make Stage 1 through Stage 4 artifact semantics explicit and machine-checkable
across basketball, football, mug, chair, and stick without changing current
solver outputs or silently treating incompatible columns as equivalent.

## Constraints

- No loss, threshold, optimization, or solver behavior changes.
- Existing CSV files remain byte-for-byte canonical outputs.
- Case-specific fields are preserved through named adapters, never dropped.
- A contract must distinguish absent, optional, and semantically incompatible
  fields instead of filling invented values.
- The Phase 0 five-case golden verifier remains the regression gate.

## Steps

| Step | Status | Evidence | Files |
| --- | --- | --- | --- |
| Hydrate ignored canonical inputs | done | 41 existing inputs verified, 27 copied, 0 errors | ignored data only |
| Inventory real Stage 1-4 schemas for five cases | done | five-case schema audit | `docs/typed_contract_schema_audit.md` |
| Define typed contracts and explicit case adapters | done | point/line/SE3 adapter tests | `core/contracts/stage_artifacts.py` |
| Validate contracts at stage-attempt completion | done | invalid schema persists `contract_failed` and blocks completion | `core/provenance/attempts.py` |
| Export contract audit into pipeline provenance | done | attempt schema v3 and CLI verifier | attempt records, `tools/verify_stage_contracts.py` |
| Run full tests and decoded golden regression | done | 43 collected; 41 passed, 2 environment/data skips; 20 contract audits, 68 inputs and 30 decoded renders verified | below |
| Commit typed-contracts checkpoint | done | commit subject: `Add typed stage artifact contracts` | branch checkpoint |

## Verification Log

- Typed-contract and provenance focused tests: 10 passed.
- Five cases × Stage 1-4 contract audits: 20 passed.
- Full repository tests: 41 passed, 2 environment/data skips.
- Golden input sync audit: 68 verified, 0 copies required, 0 errors.
- Five-case golden verification including 30 decoded RGB24 videos: pass.
- `compileall` and `git diff --check`: pass.
- No component, main-stage solver, loss, threshold, or optimization file changed.
