# Capability Plugins Implementation Plan

Branch: `refactor/capability-plugins`
Base: typed-contracts commit `987c247b` (`Add typed stage artifact contracts`)

## Goal

Replace implicit module-name dispatch and ad hoc policy maps with an explicit,
validated capability-plugin registry while invoking the exact same existing
implementation modules in the same order.

## Constraints

- No solver, loss, threshold, optimization, or policy algorithm changes.
- Existing case YAML selectors remain valid compatibility inputs.
- Plugin declarations must expose consumed and produced contract capabilities.
- Missing/unknown capabilities fail before stage execution.
- Do not remove mug/chair solved-seed dependencies in this branch.
- The five-case decoded golden verifier remains the regression gate.

## Steps

| Step | Status | Evidence | Files |
| --- | --- | --- | --- |
| Hydrate ignored canonical inputs | done | 41 existing inputs verified, 27 copied, 0 errors | ignored data only |
| Inventory selectors and active modules for five cases | done | plugin matrix | `docs/capability_plugin_matrix.md` |
| Implement explicit capability registry | done | registry/duplicate/unknown/missing-capability tests | `core/plugins/registry.py` |
| Route Stage 1-4 compatibility dispatch through registry | done | runtime dispatch smoke test preserves Stage 4 order and marker skip | mainline adapters |
| Record resolved plugin capabilities in provenance | done | attempt schema v4 and pipeline manifest | attempts, `run_pipeline.py` |
| Run full tests, contracts, artifact integrity and decoded golden regression | done | 47 collected; 45 passed, 2 environment/data skips; 24 plugins loaded, 5 chains resolved, 20 contracts/68 inputs/30 decoded renders verified | below |
| Commit capability-plugin checkpoint | done | commit subject: `Add explicit capability plugin registry` | branch checkpoint |

## Verification Log

- Capability-plugin, typed-contract and provenance focused tests: 14 passed.
- Registered plugin entrypoints loaded in `audiohoi` runtime: 24 passed.
- Complete five-case plugin chains: 5 resolved.
- Five cases × Stage 1-4 typed contract audits: 20 passed.
- Full repository tests: 45 passed, 2 environment/data skips.
- Golden input sync audit: 68 verified, 0 copies required, 0 errors.
- Five-case golden verification including 30 decoded RGB24 videos: pass.
- `compileall` and `git diff --check`: pass.
- No solver, loss, threshold, or optimization implementation file changed.
