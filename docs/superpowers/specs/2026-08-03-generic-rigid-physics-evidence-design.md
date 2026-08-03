# Generic Rigid Physics Evidence Design

## Status

Approved on 2026-08-03. This design replaces the rejected strategy of treating
small SO(3) steps as evidence of a physically correct turn.

## Scope and invariants

- Object reconstruction only. GVHMR remains a read-only source of approximate
  human sites and occluder geometry; no human state is optimized.
- The accepted pure pose remains immutable and is retained only as a historical
  raw baseline.
- Corrected `vision_only`, `no_audio`, `no_vlm`, and full runs use one shared
  generic solver. Ablations disable only their declared evidence.
- Core evidence, factor, and solver code must not branch on a case name or frame
  number.
- No accepted pose is overwritten before numerical, provenance, and rendered
  candidate review.

## Observed failure

The existing suitcase candidate does not reconstruct the real three-dimensional
turn. In the clear interval from frame 137 to frame 152:

- DA3 object depth increases from 2.087 m to 2.292 m;
- visible mask area decreases from 23,223 px to 13,257 px;
- stable body height decreases from 205 px to 180 px;
- the candidate depth instead decreases from 4.957 m to 4.646 m;
- the candidate projected height increases from 158.6 px to 160.7 px.

The candidate therefore moves closer while the observation says the object is
moving farther away. Its semantic wide-face yaw first turns in one direction
and then reverses. Quaternion step limits only hide the error.

The evidence path is also broken:

- the candidate executor does not consume `object_ref_depth_m`;
- only horizontal mask bounds are strong production factors;
- paired rail identity is frozen from an unreliable initial pose;
- frames 137--163 retain no usable rail endpoint tracks and only a few body
  tracks, many of which are far inside a contaminated mask;
- MegaPose can switch symmetric pose branches between adjacent keyframes;
- VLM amodal completion is currently report-only;
- no production depth-order factor represents behind, overlap, or in-front-of.

## Chosen architecture

Build a typed, profile-driven rigid evidence layer and compile it into the
shared factor executor. Do not extend the isolated executor with more
case-specific weights.

### 1. Rigid silhouette evidence

For every frame, publish:

- visible mask centroid;
- stable body width and height;
- log area and log aspect ratio;
- robust body contour or quadrilateral;
- visibility and covariance;
- source and artifact hashes.

The compiler must separate the residual roles:

- centroid constrains image-plane translation;
- log scale constrains depth;
- aspect ratio and contour geometry constrain yaw and tilt;
- partial masks cannot supply full-body scale or contour factors.

This prevents yaw from incorrectly compensating for an ignored scale change.

### 2. Relative metric-depth evidence

Publish DA3 depth and confidence as typed measurements. Because the DA3 and
reconstruction gauges currently differ, the first production factor uses
relative log-depth changes anchored by trusted clear frames:

`delta(log(z_pose)) ~= delta(log(depth_DA3))`.

Known asset dimensions plus silhouette scale may later establish an absolute
depth gauge, but that is not required for this repair.

### 3. Rigid feature and correspondence evidence

The descriptor declares eight body corners, four support points, two rails,
and one handle point. Tracking must obey feature visibility:

- do not seed back-facing or occluded 3D points as visible image tracks;
- body corners must remain close to the appropriate mask boundary;
- support points must remain near the lower support boundary;
- rail endpoints must remain on a detected rail line;
- contaminated interior or human tracks are rejected;
- clear-frame anchors propagate both forward and backward.

When identity is ambiguous, publish a candidate feature set instead of assigning
an irreversible left/right identity.

### 4. Temporally persistent face identity

The geometry descriptor declares the handle-side wide face, opposite wide
face, and two narrow faces. A discrete face/correspondence branch is selected
over an interval, not independently per frame.

Evidence includes paired rails, single-rail candidates, body quadrilateral,
handle offset, MegaPose hypotheses, and VLM forced-choice relations. Branch
transition costs prevent symmetric flips without visual evidence.

The selected branch then compiles ordinary residuals for the same generic
continuous SE(3) solver.

### 5. Human-relative depth order

Publish a discrete relation:

- `behind_human`;
- `overlapping_or_occluded`;
- `in_front_of_human`;
- `unclear`.

GVHMR supplies approximate read-only human geometry. VLM may select among these
predefined relations on evidence windows. The relation compiles to a depth
inequality only where the projected human and object overlap. VLM never emits
XYZ, quaternion, or free loss weights.

### 6. Contact and support physics

- Active persistent grasp constrains handle distance and hand/handle relative
  velocity.
- Release immediately removes all grasp factors.
- Support constrains declared wheel points to the plane and prevents
  penetration.
- Tangential motion remains allowed; suitcase caster motion is not assigned a
  false non-holonomic wheel model.
- Temporal factors regularize acceleration but cannot select face identity or
  override conflicting observation evidence.

## Data flow

```text
mask + DA3 + CoTracker + MegaPose + GVHMR + optional VLM
                         |
                         v
              RigidPhysicsEvidenceIR
       silhouette / depth / feature candidates /
       face branch / depth order / contact / support
                         |
                         v
                  EvidenceValidator
                         |
                         v
              Generic Factor Compiler
                         |
                         v
          Generic Sequence Executor + Publisher
```

Evidence validation runs before optimization. Invalid evidence is excluded with
explicit provenance rather than silently converted into a pose prior.

## VLM role

The full run asks forced-choice questions only:

- handle-side wide face, opposite wide face, narrow face, or unclear;
- behind, overlapping/occluded, in front, or unclear;
- visible, partial, occluded, or absent;
- amodal completion candidate A, B, reject both, or unclear.

Every accepted decision records the evidence artifact, selected interval,
confidence, active factor IDs, and whether it changed a branch or gate.
Historical report-only VLM rows do not count as solver provenance.

## Evidence gates before solving

A candidate solve is forbidden unless its manifest reports:

- silhouette scale and DA3 depth trend coverage on clear frames;
- no trusted body corner deep inside a mask boundary;
- rail identity represented as persistent or explicitly ambiguous;
- no use of an untrusted/free pose as a tracking anchor;
- face and depth-order evidence coverage or an explicit underconstrained status;
- complete source hashes and frame intervals.

## Candidate acceptance

Acceptance evaluates physical quantities, not only residual RMS:

- reconstructed depth trend agrees with DA3 and visible scale trend;
- projected width, height, area, and contour follow clear-frame observations;
- signed face identity does not flip without evidence;
- unsupported yaw reversals are rejected;
- wheels remain supported without penetration;
- grasp factors stop at release;
- locked reference intervals remain exact;
- factor ledger contains no case dispatch or human optimization.

For the suitcase evaluation, frame numbers may appear only in evaluation
annotations and review reports, never in evidence, factor, or solver branches.

## Ablation contract

- `historical_raw_baseline`: immutable existing pose, not a fair method
  ablation.
- `vision_only`: corrected shared visual, depth, rigid, support, and temporal
  solver.
- `no_audio`: shared solver with audio factors disabled.
- `no_vlm`: shared solver with VLM relation/branch gates disabled.
- `full`: shared solver with all declared evidence.

This prevents visual and physics fixes from being hidden only in the full run.

## Non-goals

- no human mesh refinement;
- no suitcase-specific optimizer;
- no manually entered solver frame labels;
- no VLM-generated continuous pose;
- no changes to audio event modeling in this repair;
- no promotion of an isolated candidate before explicit review.
