# Rigid Feature Whole-Sequence Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an isolated, physically continuous rigid-asset pose candidate whose whole-video evidence uses 8 body corners, 4 wheel points, 2 rails and 1 grasp point while preserving the approved 0–124 and 164–end trajectory segments.

**Architecture:** Build named, bidirectional feature evidence from the asset descriptor and clear frames, then solve one SE(3) sequence with locked trusted intervals and factor-gated observations. Keep the candidate outside canonical publication until video approval.

**Tech Stack:** Python, NumPy, SciPy, OpenCV, CoTracker, existing AudioHOI GeometryProvider/factor residuals/rendering.

---

### Task 1: Freeze and verify the two approved references

**Files:**
- Create: `output/suitcase_rigid_sequence_v1/reference_manifest.json`
- Create: `output/suitcase_rigid_sequence_v1/reference_pose.csv`

- [x] Resolve Annotation 1 to `generic-solve-c661e65678cf/state.csv` and Annotation 2 to `generic-solve-e1b1a07dff8d/state.csv` by reproducing their historical overlay frames.
- [x] Convert both state schemas to the canonical pose schema.
- [x] Compose frames 0–124 from Annotation 1 and frames 164–end from Annotation 2.
- [x] Record source paths, source hashes, frame ranges and composed-segment hashes in `reference_manifest.json`.
- [x] Verify frames 0–124 and 164–end compare exactly against their source state values.

### Task 2: Materialize named whole-video rigid feature evidence

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/build_rigid_feature_tracks.py`
- Create: `output/suitcase_rigid_sequence_v1/rigid_feature_tracks.csv`
- Create: `output/suitcase_rigid_sequence_v1/rigid_feature_track_metrics.json`

- [x] Read named local 3D features from the asset descriptor; reject missing body, support, rail or grasp features.
- [x] Project features at clear locked frames and at frame 125 using the Annotation 1 frame-125 hypothesis without locking it.
- [x] Track each named projection forward and backward with CoTracker across the complete video.
- [x] Compute cross-bank consistency, mask compatibility, edge compatibility and visibility confidence without changing feature identity.
- [x] Keep visible wheel/corner evidence independent from the partial-mask minimum-area rectangle.

### Task 3: Add the isolated whole-sequence rigid solver

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/run_rigid_feature_sequence_candidate.py`
- Create: `output/suitcase_rigid_sequence_v1/object_pose_candidate.csv`
- Create: `output/suitcase_rigid_sequence_v1/factor_ledger.csv`
- Create: `output/suitcase_rigid_sequence_v1/trajectory_metrics.json`

- [x] Load one root-SE(3) state for every video frame and hard-lock the two approved intervals.
- [x] Compile corner/wheel/grasp point reprojection and rail axis-line residuals from reliable named tracks.
- [x] Add one-sided visible silhouette containment so occlusion cannot shrink the physical cuboid.
- [x] Add hand-handle co-motion, four-wheel support/penetration, and SE(3) velocity/acceleration residuals.
- [x] Optimize frames 125–163 jointly rather than solving independent PnP poses.
- [x] Reject the candidate if locked values move, the support plane is violated, or unexplained single-frame jumps remain.
- [x] Compile the asset-declared persistent grasp-facing relation so trusted
  boundary states and read-only human motion recover rotation winding.

### Task 4: Render and review the real candidate

**Files:**
- Create: `output/suitcase_rigid_sequence_v1/object_only/overlay.mp4`
- Create: `output/suitcase_rigid_sequence_v1/keyframes_113_165.jpg`

- [x] Verify the canonical accepted pose SHA256 remains `b86b0834d8c7b61687f8bef24c5d5f9afdaaddc35262d32a8b1ba4fdc7ea648c`.
- [x] Compare reference and candidate values on every locked frame.
- [x] Report maximum translation/rotation step, wheel-plane gap, penetration and feature reprojection error for frames 125–163.
- [x] Render the full overlay and a dense 113–165 contact sheet.
- [x] Do not publish the candidate without explicit user approval.

## Current execution status

The whole-sequence architecture is implemented and rendered. The final review candidate is deliberately rejected rather than promoted: both locked segments are textually exact, maximum translation step is 0.0757 m, and maximum wheel penetration is 0.00523 m, but the optimizer reached its 100-evaluation limit and the maximum rotation step is 8.045° against the declared 8.0° gate. The canonical accepted pose remains unchanged.

### Task 5: Replace tilt compensation with generic rigid physics factors

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/tools/run_rigid_feature_sequence_candidate.py`
- Create: `output/suitcase_rigid_sequence_v32/`

- [x] Consume the typed rigid silhouette and relative-depth evidence produced by
  `build_rigid_physics_evidence.py`; refuse to solve when its blocking manifest
  is absent or declares contaminated usable tracks.
- [x] Add full visible-body width and height residuals instead of horizontal-only
  containment, preserving partial/occluded measurements as one-sided evidence.
- [x] Add a descriptor-driven support-upright factor using
  `initializer.upright_axis_local` and the observed support plane. This prevents
  the optimizer from explaining a narrowing rigid silhouette with a 3D tip.
- [x] Add relative-depth ordering over reliable visible windows. This is a weak
  direction/rank constraint, not an absolute DA3-to-camera-depth calibration.
- [x] Keep the signed handle-face relation active only while contact is active;
  after release, use rigid temporal continuity and the locked late boundary.
- [x] Write per-factor provenance and reject the candidate on depth-order,
  upright, support, locked-segment, or motion-jump gate failure.
- [x] Run the isolated real candidate, verify the accepted pose hash remains
  unchanged, and render only if all evidence and trajectory gates pass.

Task 5 completed with isolated attempt v32. The accepted pose remained at
SHA256 `b86b0834d8c7b61687f8bef24c5d5f9afdaaddc35262d32a8b1ba4fdc7ea648c`.
The converged candidate preserves both locked intervals exactly, has maximum
translation step 0.11174 m, maximum rotation step 7.518 degrees, maximum wheel
penetration 0.000535 m, maximum upright tilt 11.65 degrees, and relative-depth
rank correlation 0.708. The 137-142 review frames now contain a visible signed
heading change driven by body aspect plus the projected handle/body lever arm.
The candidate remains unpromoted pending explicit video approval.

### Task 6: Preserve signed heading through the complete visible-to-occluded turn

The later frame-by-frame review superseded the earlier assumption that frames
111-124 were trustworthy locks.  Numeric audit of v32 found a real winding
reversal: heading advanced in one direction through frame 137, reversed over
137-142, and changed direction again afterward.  The cause was structural:
mask aspect supplied heading magnitude but not sign, rail axes were unsigned,
and independent per-frame quaternion corrections allowed robust optimization to
sacrifice a soft direction residual.

- [x] Expand the audited solve interval to 111-163 while preserving frames
  1-110 and 164-240 exactly.
- [x] Rebuild rigid evidence for that interval; all seven evidence gates pass,
  with 43 clear-scale rows, 53 depth rows, 13 usable feature frames, 36 trusted
  rail frames, and a maximum visible orientation gap of 6 frames.
- [x] Record a signed winding decision and its source rather than inferring a
  new sign independently inside the occlusion.
- [x] Parameterize support-plane heading as same-sign positive increments, so
  a reverse step is outside the continuous solver's feasible state space.
- [x] Resolve the periodic endpoint branch using both clear boundaries and fix
  the total turn; optimization only redistributes that turn over time.
- [x] Include the locked right boundary in reversal accounting.

The v45 diagnostic verifies the corrected topology across both boundaries:
signed turn is -113.092 degrees, reversal count is zero, depth-rank correlation
is 0.9901, maximum translation step is 0.11788 m, and maximum wheel penetration
is 0.00377 m.  It remains an isolated, unpromoted candidate because it is an
initializer verification run, the maximum full-SE(3) step is 9.37 degrees, and
the legacy 18-degree upright gate incorrectly rejects the visibly tilted pull
state.  The accepted pose hash remains unchanged.

#### Superseding visual-face audit

The subsequent original-frame audit rejected v45 despite its zero reversal
count.  It had mapped screen-counterclockwise directly to support sign -1 and
therefore selected the shortest periodic endpoint branch (-113 degrees).  That
branch reaches an equivalent endpoint quaternion but cannot traverse the
observed named-face sequence.  The actual sequence is `+X oblique side -> -Y
outer broad face -> -X narrow side -> +Y handle broad face`, which maps through
the camera projection to support sign +1 and the +246.908-degree branch.

- [x] Map clockwise/counterclockwise screen semantics to support-plane sign by
  projecting a positive one-degree support rotation through the camera.
- [x] Add explicit projected separation for paired parallel rails.
- [x] Treat one detected rail axis on a fully visible object as collapsed-bundle
  evidence, requiring both physical rails to project to that axis.
- [x] Make silhouette boundary sampling fixed-size by arc length; a changing
  cuboid hull may no longer change the least-squares residual dimension.
- [x] Verify v50 follows the required face sequence at frames 123, 139/142 and
  155/160 with no signed reversal.

The isolated v50 review has a +246.908-degree signed turn, zero reversals,
0.11514 m maximum translation step, 0.000266 m maximum wheel penetration, and
0.9651 relative-depth rank correlation.  It remains unpromoted: the optimizer
hit its evaluation limit, maximum full-SE(3) step is 8.37 degrees, and the
legacy upright gate still rejects the intentional tilted pull state.

### Task 7: Use both rails and all four wheels as whole-sequence motion evidence

The v50 review exposed a semantic error in the original feature tracker.  The
four points called `support_point` were ground-contact points at local `z=0`,
not visible wheel centres.  CoTracker was therefore queried on the floor below
the visible wheels.  The subsequent audit compounded the error by requiring
those points to lie in the SAM2 main-body mask and by comparing long-baseline
track banks.  This left only sparse evidence from two front-axle points.

- [x] Add four asset-derived visual wheel centres at local wheel radius while
  retaining the four distinct `z=0` support points for physical contact.
- [x] Track both rail endpoints and all four wheel centres from dense local
  anchors over frames 1-240; keep the current pose only as the query seed.
- [x] Separate absolute pixel evidence from relative local flow so a projected
  query is never relabelled as a measured 2D landmark.
- [x] Compile one local-flow factor per named feature and frame, with the
  nearest reliable bank selected to prevent dense-anchor over-weighting.
- [x] Keep the old absolute feature table as a separate input, permitting an
  isolated flow/no-flow comparison with all other solver inputs fixed.
- [x] Verify full-video flow coverage and render the isolated candidate without
  modifying the accepted pose.

The local tracker now covers all 240 frames.  In the free interval 111-163,
all four wheel centres are directly measured in 49/53 frames and three are
measured in each of the remaining four frames; all four rail endpoints are
available in 49/53 frames and at least two in every frame.  Missing visible
measurements remain fixed local features of the same SE(3) asset and are
propagated by rigid state continuity, never fabricated as pixel observations.
The v53 solve compiled 208 wheel-centre and 206 rail-endpoint flow observations
across all 53 free frames.  Against the current-code no-flow control v54, it preserves the
+246.908-degree winding and zero reversals, changes median rotation by 3.88
degrees, and reduces maximum wheel penetration from 1.99 mm to 0.39 mm.  It
remains isolated because the optimizer hit its evaluation limit and the full
SE(3) step gate is 8.75 degrees against the declared 8-degree limit.

### Task 8: Remove measurement-source and factor-gate jitter without changing the motion

The v53 trajectory has the correct named-face winding, but framewise audit
found two generic sources of high-frequency jitter.  First, overlapping local
CoTracker banks were selected independently by score, causing 193 changes of
measurement source even when the named rigid feature stayed the same.  Second,
rail evidence changed abruptly between paired, unassigned and missing modes at
the visibility boundary.  Acceleration alone then concentrated those conflicts
into adjacent pose steps.

- [x] Select overlapping flow banks with temporal hysteresis per named feature;
  retain the same 414 measured observations and reduce source changes from 193
  to 90.
- [x] Scale line evidence by endpoint reliability and visible line length, and
  ramp line factors at mode boundaries instead of switching them in one frame.
- [x] Add optional translation and SO(3) rotation jerk factors to the generic
  rigid sequence objective; keep their default weights disabled for parity.
- [x] Preserve the locked intervals, +246.908-degree winding, zero reversals,
  support contact and the original non-uniform translation.
- [x] Render the isolated result and leave canonical publication unchanged.

The v58 isolated candidate uses a flow-bank switch penalty of 3, a three-frame
line gate ramp, 0.04 m translation-jerk sigma and 0.06 rad rotation-jerk sigma.
Relative to v53 over frames 111-163, median translation jerk drops from 0.0390
to 0.0231 m/frame^3, maximum translation jerk from 0.1970 to 0.1298, median
rotation jerk from 2.2582 to 1.4873 degrees/frame^3, and maximum rotation jerk
from 13.1432 to 9.4331.  It keeps the required signed turn and zero reversals.
The candidate remains unpromoted because the optimizer still reaches its
evaluation limit and its 8.20-degree maximum SE(3) step is slightly above the
declared 8-degree gate; visual approval is also required.

### Task 9: Re-solve the visible 62-108 turn under the same signed heading topology

The v58 audit proved that frames 62-108 were not a trustworthy lock.  They are
copied exactly from `generic-solve-c661e65678cf`, whose support-plane heading
contains 22 positive and 22 negative steps despite the observed side-to-broad
face transition being purely screen-counterclockwise.  The old attempt applied
six local quaternion repair intervals and a per-frame contact-facing projection,
but had no sequence-level winding constraint.  The current whole-sequence solve
cannot correct it because its free interval starts at frame 111.

**Artifacts:**

- Create: `output/suitcase_rigid_physics_evidence_62_163/`
- Create: `output/suitcase_rigid_sequence_v59_full_turn/`
- Preserve: `samples_known_object/15_suitcase_drag/results/pure_solver_no_audio_no_vlm/object_pose.csv`

- [x] Rebuild the generic rigid evidence manifest with trusted intervals
  `1-61,164-240`; require all evidence gates to pass before solving.
- [x] Run `run_rigid_feature_sequence_candidate.py` with free interval 62-163,
  `--expand-free-interval`, the v58 pose as warm start, the same local-flow
  hysteresis, line gate ramp and jerk factors, and
  `--heading-screen-direction counterclockwise`.
- [x] Verify frames 1-61 and 164-240 remain textually locked; require zero
  signed heading reversals over 62-163 and preserve support penetration and
  bounded translation steps.
- [x] Render all 240 frames and inspect the 62-108 side-to-broad transition,
  then compare frames 111-163 against v58 to ensure the previously approved
  face sequence is retained.
- [x] Keep the candidate isolated unless both hard gates and video review pass;
  do not publish canonical output.

#### v59 diagnostic and bounded v60 correction

The first 62-163 solve established zero reversal but did not recover the turn
timing: only 3.25 degrees occurred over frames 62-108, while the solver used up
to 44.9 degrees of tilt to explain silhouette width and deferred most heading
change until later.  The cause is evidence identity, not the signed-heading
parameterization.  Dense local flow banks were re-seeded from an already-wrong
pose every few frames, while the solver also applied a second exponential
distance decay after the tracker had already encoded distance in reliability.

- [x] Run a frame-61 single-anchor CoTracker diagnostic through frame 111. It
  retains seven of eight rail/wheel measurements at frame 108 and captures the
  two physical rails crossing in image x, providing direct turn evidence.
- [x] Add a parity-default option to disable only the solver's extra distance
  decay for sequence-anchor flow; retain the tracker's visibility and distance
  reliability.
- [x] Rebuild evidence for free interval 62-110 and lock the approved v58 pose
  from frame 111 onward, so the early turn cannot be deferred into the later
  face sequence.
- [x] Solve and render isolated v60 using the frame-61 long-range rail/wheel
  tracks, zero signed reversals and the same support/jerk factors.

The v60 candidate completes a +25.262-degree support-plane turn over frames
62-110 with no negative step beyond floating-point roundoff, and frames 1-61
plus 111-240 are exactly equal to v58.  It compiles 315 long-range flow factors
from eight named rail/wheel identities, has a 7.52-degree maximum rotation step,
0.0658 m maximum translation step and 0.116 mm maximum wheel penetration.  It
remains isolated: the optimizer reaches its evaluation limit, the existing
upright metric peaks at 45.54 degrees during the visibly tilted pull, and the
relative-depth rank gate is negative.  Video review therefore precedes any
promotion.

### Task 10: Correct the audited-feature boundary and visible-depth contract

The v60 failure audit disproved the hypothesis that more optimizer evaluations
would resolve the remaining depth conflict.  A 100-evaluation control reduced
RMS only from 2.047 to 1.884, did not converge, and retained negative depth-rank
correlation.  It also exposed a concrete data-boundary bug: the solver read the
raw feature-track table directly and accepted 401 rows which the typed rigid
evidence had rejected, including tracks seeded from non-trusted anchors.

- [x] Require every absolute rigid feature row consumed by the solver to match
  a usable row in `rigid_feature_track_evidence.csv`.
- [x] Record consumed audited rows and rejected raw rows in trajectory metrics.
- [x] Verify the real 62-110 control removes 401 unaudited rows, reduces named
  observations from 433 to 217, reduces RMS from 2.047 to 1.485, preserves zero
  heading reversals, and leaves the accepted pose untouched.
- [x] Test and reject a nearest-body-point visible-depth approximation: it
  lowers local order violations but retains negative global rank correlation.
- [x] Replace the invalid `mask median depth == root tz` assumption with a
  geometry-aware measurement model.  The production candidate now casts a
  deterministic set of camera rays through the observed body mask, intersects
  them with the oriented rigid-body box under the current SE(3), and compares
  Stage-0 DA3 against the median visible near-surface depth.  The v68 control
  changes the depth-rank correlation from negative to `0.511`; the final v70
  solve reaches `0.658` with four order violations and at least 66 valid rays.
- [x] Add a factor-ledger-derived sparse Jacobian structure before attempting
  another unlocked solve.  The v69 isolation control converges in 32 function
  evaluations instead of exhausting the dense 100-evaluation budget.  The v70
  solve converges in 34 evaluations with final residual RMS `1.578`, zero
  heading reversals, a `7.624`-degree maximum rotation step and a `0.1151 m`
  maximum translation step.
- [x] Move the upright limit out of the solver's implicit suitcase assumption
  and into the geometry capability declaration.  The fixed asset declares
  `support_model.mode=rolling_axle` and a 50-degree pulled-tilt envelope; v70's
  maximum tilt is `44.269` degrees with only `0.053 mm` wheel penetration.
- [x] Render all 240 v70 frames in object-only and read-only-human-overlay
  variants.  The solver reports `human_state_optimized=false`, every pose field
  outside frames 62-110 is exactly equal to the reference, and the canonical
  accepted pose SHA-256 remains
  `b86b0834d8c7b61687f8bef24c5d5f9afdaaddc35262d32a8b1ba4fdc7ea648c`.

The nearest-point approximation is not retained in production code.  v70 now
passes every declared hard gate and remains an isolated candidate pending video
approval; it has not replaced the canonical pose.  The human result is read
only for object-contact evidence and relation rendering and is never optimized.
