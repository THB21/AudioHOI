# Chair Solved-Seed Removal Plan

Branch: `refactor/chair-seed-removal`
Base: mug seed-removal commit `6e64015f`

后续阶段：chair 接受并提交后进入
[`generalized_reconstruction_kernel_plan.md`](generalized_reconstruction_kernel_plan.md)，
不得把当前 chair 专用修复直接宣称为跨类别泛化。

## Goal

Remove the chair Stage 4 read of the cached/historical physical6d pose. A fresh
result directory must use only current-run Stage 1-3 artifacts, canonical
observations and declared chair geometry.

## Constraints

- Do not rename a solved pose into a new input artifact.
- Do not change loss weights, bounds, thresholds or solver algorithms before a
  fresh-run before/after comparison is recorded.
- Keep all non-chair behavior unchanged.
- Compare semantic 2D fit, palm/contact residuals, freeze behavior and decoded
  renders independently.
- Treat the canonical solved pose as evaluation-only, never as optimizer input.

## Steps

| Step | Status | Evidence | Files |
| --- | --- | --- | --- |
| Create isolated worktree and hydrate runtime inputs | done | 89 verified; 47 initial directories plus missing chair tracking directory copied | ignored data only |
| Audit current chair Stage 1-4 dataflow and hidden baseline reads | done | seed and quality-gate baseline both affect Stage 4 selection | chair components |
| Establish observation sufficiency and acceptance criteria | done | rigid chord lower bound, 2D gauge and canonical evaluation-only comparison recorded | this note |
| Replace Stage 4 seed with current-run Stage 3 artifact | done | 125/125 constrained gauge solves pass; no historical seed/fallback | chair components |
| Execute empty-directory chair run and compare canonical | done | all standard canonical comparison checks pass; six formal Stage 5 videos regenerated | `chair_seed_removal_comparison.md` |
| Run repository-wide regressions and commit checkpoint | done | 55 passed, 2 skipped; five-case encoded/decoded golden and 89-input dry-run pass | branch checkpoint |

## Initial finding

Stage 3 already fits `object_pose_init.csv` from current-run semantic tracks.
Stage 4 first refines that artifact, but its pair-propagation optimizer then
uses `resolve_chair_physical6d_seed()` as both reference and initialization.
The selected canonical artifact is classified as a cached derivative of a
historical solved pose; a fresh result is blocked because the historical
mainline files are absent.

## Current diagnostics

- The empty-directory Stage 1-3 run succeeds after adding the full ignored
  `results/tracking` directory to the runtime-input manifest.
- Fresh and canonical `object_pose_init.csv` have numerically identical pose
  fields for all 192 frames; hashes differ only because provenance text differs.
- Current Stage 3 semantic-2D median error is 12.86 px, better than the cached
  historical physical6d seed at 23.16 px.
- Current generic Stage 4 improves semantic-2D fit to 14.44 px but has 0.718 m
  median palm-contact gap, so pairprop remains necessary.
- The first root-only pairprop from the current Stage 3 seed, using unchanged canonical parameters,
  reaches 19.66 px semantic-2D median and 0.0425 m contact median, but contact
  P90 is 0.189 m versus canonical 0.0693 m. Its current-run quality gate rejects
  it. This rejected experiment established the missing degrees of freedom; it
  is not the selected chair path.
- The two top-rail endpoints form a 0.382 m local chord. Palm-chord mismatch
  gives a directly measurable contact lower bound: 0.02054 m median, 0.04471 m
  P90 and 0.06205 m maximum. All are below canonical contact error.
- The accepted initializer aligns the local chord to the current GVHMR palms,
  then resolves the remaining twist gauge and two chair joints from current-run
  Stage-3 2D projections. It has no solved-pose input.
- The invariant gate requires all 125 active frames to initialize, every 2D
  gauge solve to succeed without increasing its objective, contact to reach the
  geometric lower bound, and the boundary freeze check to pass.
- The generic sequence smoother initially damaged frames 120 and 123 after the
  constrained solve. A result-owned `pose_lock_reason` now prevents downstream
  smoothing from rewriting a pose that has passed the constraint gate.
- Final standard comparison against canonical passes all checks. See
  [`chair_seed_removal_comparison.md`](chair_seed_removal_comparison.md).
