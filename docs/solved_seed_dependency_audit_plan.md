# Solved-Seed Dependency Audit Plan

Branch: `refactor/solved-seed-dependency-audit`
Base: capability-plugins commit `54d77caa` (`Add explicit capability plugin registry`)

## Goal

Make the mug and chair solved pose/phase dependency chains explicit and
machine-verifiable before changing either solver path. The audit must identify
the source actually selected for an existing canonical result and the source
that would be selected for a fresh result directory.

## Constraints

- No solver, loss, threshold, optimization, or policy algorithm changes.
- Do not move or reinterpret stage artifacts.
- Preserve the five canonical decoded outputs byte-for-byte.
- Classify missing historical inputs as evidence; do not invent replacements.
- Split mug and chair dependency removal into later, independent branches.

## Steps

| Step | Status | Evidence | Files |
| --- | --- | --- | --- |
| Hydrate ignored canonical inputs | done | 41 existing inputs verified, 27 copied, 0 errors | ignored data only |
| Inventory mug/chair seed candidates and runtime reads | done | existing/fresh source-chain audit | `docs/solved_seed_dependency_audit.md` |
| Add a read-only dependency audit and CLI | done | deterministic, side-effect-free JSON output | `core/provenance/seed_dependencies.py`, audit CLI |
| Freeze expected canonical and fresh-run classifications | done | 3 focused tests pass | `tests/golden/solved_seed_dependency_expectations.json` |
| Run full tests, contracts, artifact integrity and decoded golden regression | done | all applicable gates pass; canonical results predate attempt directories | below |
| Commit audit checkpoint | done | commit subject: `Audit mug and chair solved-seed dependencies` | branch checkpoint |

## Verification Log

- Seed dependency focused tests: 3 passed; fresh audit created no result directory.
- Full repository tests: 49 passed, 1 generated-data skip (`contact_points_csv`).
- Registered plugin entrypoints loaded in the `audiohoi` runtime: 24 passed.
- Complete five-case plugin chains: 5 resolved.
- Five cases x Stage 1-4 typed contract audits: 20 passed.
- Golden input audit: 68 verified, 0 copies required, 0 errors.
- Five-case golden verification including 30 decoded RGB24 videos: pass.
- Artifact-store deduplication, immutable reference and corruption detection tests: pass.
- The canonical result directories have no `provenance/stages` because they
  predate the attempt/artifact-store feature; the integrity CLI therefore has
  no historical attempt records to verify and reports that absence explicitly.
- `compileall`, `git diff --check`, and fresh-directory absence check: pass.
