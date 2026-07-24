# Mug Solved-Seed Removal Plan

Branch: `refactor/mug-seed-removal`
Base: solved-seed audit commit `48e4b599` (`Audit mug and chair solved-seed dependencies`)

## Goal

Remove the mug pipeline's reads of historical M17/final handle phase and M18
solved pose. A fresh result directory must derive Stage 1, Stage 3, and Stage 4
inputs from canonical observations and declared geometry only.

## Constraints

- Do not preserve a solved pose/phase CSV under a new input name.
- Do not change loss weights, thresholds, optimizer bounds, or solver methods
  until observation sufficiency and a before/after comparison are recorded.
- Keep chair and all other case behavior unchanged.
- Compare phase, pose, contact/gates, and decoded renders independently.
- If observations cannot identify a degree of freedom, expose that capability
  gap rather than silently substituting an identity or historical solution.

## Steps

| Step | Status | Evidence | Files |
| --- | --- | --- | --- |
| Hydrate ignored canonical inputs | done | 20 partial/missing directories safely merged; 88 verified, 0 pending, 0 errors | `tests/golden/pipeline_v1_runtime_inputs.json` |
| Audit mug observation/geometry sufficiency for body pose and handle phase | done | 240 body/depth frames, 159 visible handle frames; axial gauge identified | `docs/mug_observation_seed_design.md` |
| Define fresh-run acceptance and non-regression comparisons | done | explicit readiness, typed-row, gate, geometry and overlay criteria | design note |
| Implement observation-derived seed path or explicit capability failure | done | 240-row pose/phase; no historical fallback or Stage 4 phase constants | mug components |
| Execute isolated fresh mug run and compare with canonical output | done | all 21 acceptance checks pass after full rerender | `docs/mug_seed_removal_comparison.md` |
| Run repository-wide regression gates | done | pytest, contracts, plugins, artifact store, golden and hydration checks pass | below |
| Commit mug checkpoint | done | `Remove mug solved-seed dependencies` | branch checkpoint |

## Verification Log

- `python -m pytest -q`: 54 passed, 1 skipped.
- Five canonical cases, Stage 1-4 typed contracts: 20 passed.
- Fresh mug Stage 1-4 typed contracts: 4 passed.
- Capability plugins: all declared plugins load and all five case profiles resolve.
- Fresh mug immutable artifact store: verified.
- Phase 0 golden plus runtime-input supplement: five cases verified.
- Runtime-input hydration dry run: 88 verified, 0 pending, 0 errors.
- Fresh mug seed audit: ready, no solved-seed dependency.
- Qwen Stage 0-2 gates: passed. Stage 3 model loading is blocked by available
  GPU memory while another user service owns 2.47 GiB; no service was stopped
  and no gate/runtime parameter was changed.
