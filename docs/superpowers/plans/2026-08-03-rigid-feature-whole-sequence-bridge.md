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
