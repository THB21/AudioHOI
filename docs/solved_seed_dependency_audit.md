# Mug/Chair Solved-Seed Dependency Audit

## Scope

This audit describes the read chain at capability-plugin checkpoint `54d77caa`.
It compares a rerun in the existing `benchmark_vlm_qwen` result directory with
a run in an empty result directory. It does not execute or change a solver.

Run the machine-readable audit with:

```bash
python scripts/shared/generic_contact_pipeline/tools/audit_seed_dependencies.py
```

## Findings

| Case/context | Runtime selection | Classification | Readiness |
| --- | --- | --- | --- |
| mug/existing | generated zero-angle `identity_handle_phase.csv`; missing historical M18 pose | phase would change; pose input is a historical solved seed | blocked |
| mug/fresh | same as existing | phase would change; pose input is a historical solved seed | blocked |
| chair/existing | cached `physical6d_rebuild_from_mainline_saved2d/physical6d_pose.csv` | derivative of historical solved mainline pose | runnable but solved-seed dependent |
| chair/fresh | missing `mainline_0425/physical6d_seed/physical6d_pose.csv` | historical solved seed | blocked |

### Mug

`resolve_mug_m17_phase` tries, in order: an M17 reconstruction inside the
current result, a preserved snapshot, historical M17, historical final phase,
then a generated identity phase. The first four inputs are absent. The object
observation CSV exists, so a rerun would write a zero-angle identity phase and
continue with different phase semantics.

Stage 3 independently invokes `mug_opening_2d_pose_correction.py` with the
configured M18 pose as `--pose-csv`. That historical solved pose is absent, so
both an existing-directory rerun and a fresh run are blocked. Existing
`object_pose_init.csv` does not remove this dependency because the Stage 3
builder invokes the correction solver unconditionally.

Stage 4 normally consumes the new Stage 3 pose but resolves the same phase
chain again. Consequently, copying only the final pose or the Stage 4 output
would conceal, not remove, the dependency.

### Chair

Stage 3 (`semantic_graph_6d`) fits a pose from the current observation/contact
artifacts and is not the solved-seed dependency identified here. Stage 4
(`small_se3`) supplies the same resolved physical6d CSV as both `--ref-pose-csv`
and `--init-pose-csv` to the pair-propagation solver.

The existing canonical result contains the cached rebuilt physical6d CSV, so
Stage 4 can select it even though three inputs recorded by its reconstruction
report are now absent: historical mainline pose, historical semantic local
segments, and historical 2D observations. The cache is therefore runnable but
is still a derivative of a historical solved pose. An empty result directory
has no cache or snapshot and falls through to the absent historical physical6d
seed.

## Removal Branch Boundaries

1. `refactor/mug-seed-removal` must replace both the handle-phase chain and the
   M18 body-pose seed with observation-derived inputs. Its regression must
   compare phase, pose, contact/gates, and decoded renders separately.
2. `refactor/chair-seed-removal` must make Stage 4 initialize/reference the
   current Stage 3 observation-derived pose (or a new observation-only
   derivation), without reading the cached mainline-derived physical6d pose.
3. Neither branch may silently preserve the current solved CSV under a new
   filename. Any temporary compatibility snapshot remains classified as a
   solved seed by the audit.
4. Losses, thresholds, and solver parameters remain unchanged until each
   branch has an explicit before/after result and an accepted regression
   decision.

## Frozen Evidence

`tests/golden/solved_seed_dependency_expectations.json` freezes selection and
readiness, while `tests/test_seed_dependencies.py` verifies both existing and
fresh contexts and proves the audit creates no fresh result directory.
