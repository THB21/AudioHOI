# Mug Observation-Derived Pose/Phase Design

## Runtime dependency inventory

The seed audit originally exposed the historical M17/final phase chain and the
M18 Stage 3 pose. Static inspection of the Stage 1 exporter found a third read:
`export_mug_articraft_contact_points.py` defaults to
`proxy/mug_body_only_cylinder_pose_segmented_sequence.csv`. That tracked CSV is
also a historical solved pose. The machine audit and frozen expectations now
include all three dependencies.

## Available canonical observations

| Evidence | Coverage | Role |
| --- | ---: | --- |
| mug body center and bbox | 240/240 frames | translation, projected scale, tilt silhouette |
| DA3/object proxy depth | 240/240 frames | metric depth and backprojection |
| visible handle center and side | 159/240 frames | axial orientation |
| handle hidden state | 81/240 frames | temporal interpolation/occlusion constraint |
| GVHMR camera intrinsics | available | projection/backprojection |
| materialized body/handle meshes | available | geometric reprojection |
| manual rim/bottom annotations | hydrated from canonical source | semantic/evaluation evidence |
| VLM handle-visibility CSV | hydrated from canonical source | contact/gate semantics |
| segmentation masks | 240/240 frames after hydration | overlay evaluation |

The last three inputs were absent only from the newly created worktree because
they are Git-ignored. They exist in the canonical source worktree and are now
covered by `tests/golden/pipeline_v1_runtime_inputs.json`. The observation seed itself
uses the raw body/handle boxes, proxy depth, intrinsics and meshes; the hydrated
annotations and masks remain pipeline-semantic and evaluation inputs.

The body is approximately axially symmetric. Image evidence therefore does not
identify body yaw and the existing `phase` as two independent variables. Both
are rotations about the mug-local vertical axis before projection. The fresh
path must use the explicit gauge `body_yaw = 0` and store the observable axial
angle in `phase`; raw yaw/phase values must not be compared independently.

## Read-only feasibility probes

- Scanning the axial angle against the 159 visible handle centers while using
  the existing body-only geometry trajectory gives 5.21 px median and 9.42 px
  P90 handle-center reprojection error.
- The canonical M18/M17 combination gives 10.81 px median and 43.31 px P90
  under the same mesh-center diagnostic. This is a diagnostic rather than a
  replacement for the full overlay evaluator, but confirms that the observed
  handle signal is strong enough to estimate the combined axial angle.
- The legacy global body-only least-squares implementation has 1,680 variables
  and a dense numerical Jacobian. With annotations/masks absent it also tries
  to write a propagated-annotation file during fitting. A read-only 240-frame
  probe did not finish its first practical solve within two minutes. It is not
  suitable as the fresh pipeline path unchanged.

## Implemented path

1. Build a 240-row body pose directly from body center/bbox, proxy depth and
   intrinsics, then perform bounded per-frame silhouette refinement. This avoids
   the dense global numerical Jacobian and writes only inside the active result
   directory.
2. Fit one observable axial angle on visible frames by projecting the declared
   handle mesh. Resolve the front/back branch using observed side and temporal
   continuity; interpolate hidden spans without inventing a second yaw degree
   of freedom.
3. Feed that same result-owned body pose and phase to the Stage 1 contact
   exporter and Stage 3. The opening-frame correction refines translation from
   current observations and does not accept a baseline pose.
4. Record the gauge, observation coverage, residuals, and every source hash in
   Stage 1/3 provenance.
5. Stage 4 consumes the result-owned phase directly, with the existing
   7-degree-per-frame velocity guard. The historical 30-point phase curve and
   PCHIP reconstruction were removed.

## Fresh-run acceptance

Mandatory gates:

- The seed audit selects no `historical_solved_*`, `preserved_solved_snapshot`,
  or cached historical derivative, and selects no missing path.
- Stage 1, Stage 3 and Stage 4 each produce 240-row typed artifacts in an empty
  result directory.
- Contact/gate labels and blocking decisions are compared frame-by-frame with
  the canonical result; any difference is reported, never hidden by a hash.
- Gauge-invariant geometry is evaluated after composing pose rotation and
  axial phase. Raw yaw and phase are not used as a correctness metric.
- The standard overlay-mask evaluation must not fall below canonical mean IoU
  0.4509453 or mean coverage 0.4982529, and must not exceed canonical mean false
  coverage 0.1594556.
- All six decoded output video hashes are recorded. Exact equality is expected
  only for unaffected cases; the mug change requires metric acceptance plus a
  documented visual comparison.

No loss weight, optimizer bound, or threshold is changed merely to pass these
gates. If the first observation-derived implementation misses them, its result
remains an experiment rather than replacing the canonical path.

## Generic projected-periodic migration checkpoint

The observation-derived implementation now delegates numerical solving to the
object-agnostic `core/solver/projected_periodic_sequence.py`. Mug-specific code
only adapts detector rows and supplies the declared body/handle geometry. The
kinematic contract explicitly marks the handle as a fixed periodic feature of
a rigid assembly (`physical_joint=false`, `relative_motion_allowed=false`), so
the legacy phase cannot be misread as handle articulation.

Frozen switched-run evidence:

- input hashes: observations `f7877ab449f6...`, proxy depth `51d5a084c72c...`;
- body pose `3ea905dfe8de...` and axial phase `3dc05829ef95...`, both byte-identical
  to the fresh observation-derived baseline;
- all Stage 1–4 observation/local-point/init/pre-smooth/final/phase/contact CSVs
  are byte-identical to that baseline;
- Stage 1 attempt provenance stores body pose, axial phase and seed report in the
  content-addressed artifact store;
- repository regression: 129 passed, 3 skipped.

This closes the historical phase fallback for fresh Stage 1. It does not yet
claim a single factor executor for Mug Stage 3/4; Stage 5–7 and decoded renders
remain the next promotion gate.
