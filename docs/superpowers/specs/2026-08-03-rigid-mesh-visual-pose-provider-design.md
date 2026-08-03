# Generic Rigid-Mesh Visual Pose Provider Design

## Goal

Replace the current bounding-box/principal-axis interpretation of rigid-object masks with an existing-method stack that can preserve mesh geometry, face orientation, and persistent point identity through video. The first validation target is the suitcase sequence, but no suitcase-specific solver or `case_name` branch is allowed.

The accepted object trajectory is not modified during development. Every new result is written as an isolated candidate until its overlay and object-only 3D render are explicitly approved.

## Confirmed root cause

The current pipeline loses the information required to reconstruct a rigid suitcase:

1. `MaskSilhouetteFactor` compares only projected and observed bounding-box edges plus an optional PCA principal axis. It does not compare a rendered mesh silhouette or visible faces.
2. The current CoTracker runner initializes `center`, `left`, `right`, `top`, and `bottom` from mask extrema. Those extrema are not persistent material points on a rotating object.
3. CoTracker is restarted independently every 32 frames. A new query set is therefore created from the potentially occluded mask at each boundary. The 128-frame boundary overlaps the observed failure interval.
4. The exported pseudo-mesh tracks do not contain valid mesh-surface coordinates for a rigid cuboid. Most local coordinates are zero, so they cannot constrain full SE(3) pose.

SAM2 mask quality and generic temporal smoothing cannot recover information discarded at these boundaries.

## Existing methods selected

### SAM2 plus point-propagated prompting

Use the published SAM-PT pattern instead of treating SAM2 and CoTracker as independent preprocessors:

- initialize positive points inside a clean visible object mask;
- initialize negative points immediately outside the object and on common occluders;
- propagate the same point identities with CoTracker;
- pass visible propagated points back to SAM2 as prompts;
- retain explicit per-point visibility and do not turn invisible points into mask evidence.

This layer outputs visible-object masks and visibility evidence. It does not infer 6D pose.

### Persistent CoTracker3 tracks

Use the official full-sequence offline API or the official stateful online API. Do not split the video into independent 32-frame tracking problems.

Query points are sampled as a stable grid inside a clean keyframe mask, with additional points on declared geometry features such as the two handle rails. Point IDs remain stable for the complete tracking interval. Reinitialization is allowed only as an explicit new tracking attempt after a hard confidence failure; it never silently changes point identity inside one attempt.

This layer outputs 2D tracks, visibility, confidence, query-frame identity, and provenance. Mask extrema are retained only as non-correspondence diagnostics.

### MegaPose RGB keyframe pose hypotheses

Use MegaPose's RGB model because the project already has:

- RGB frames;
- camera intrinsics;
- an Articraft mesh;
- an object ROI from SAM2;
- no consistently reliable metric depth during the failing interval.

MegaPose runs on visible keyframes and produces multiple mesh-aware SE(3) hypotheses. It is an external generic pose provider, not a new optimizer inside the AudioHOI solver.

FoundationPose and SAM-6D remain optional future providers for runs with verified RGB-D. DA3 pseudo-depth is not treated as equivalent to measured depth without separate validation.

## Data flow

```text
RGB frames + first clean mask
        |
        v
SAM2 <-> persistent CoTracker3
        |
        +--> visible masks + point tracks + visibility
        |
        v
MegaPose RGB on reliable keyframes
        |
        v
mesh-aware SE(3) hypotheses + render evidence
        |
        v
typed pose measurements / hypothesis ledger
        |
        v
existing GenericSequenceExecutor
support + contact + temporal + observation reliability
        |
        v
isolated candidate pose/render
```

## Interfaces and artifacts

### Persistent track artifact

`rigid_point_tracks.csv` contains:

- `frame`, `time`;
- `track_id`, `query_frame`;
- `x`, `y`, `visible`, `confidence`;
- `semantic_feature_id` when a point belongs to a declared feature;
- `source`, `attempt_id`.

It does not claim a 3D local coordinate unless that coordinate was assigned from a rendered mesh under a recorded pose hypothesis.

### Keyframe pose artifact

`rigid_pose_hypotheses.jsonl` contains:

- frame and hypothesis ID;
- translation and quaternion;
- MegaPose score/model identifier;
- mesh hash, camera hash, and RGB-frame hash;
- source ROI/mask measurement IDs;
- rendered overlay and silhouette evidence paths;
- accepted/rejected status and reason.

### Solver boundary

The generic solver consumes typed external pose hypotheses as ordinary measurements. It must not import MegaPose internals, inspect `case_name`, or create suitcase-specific weights. Existing contact, support, penetration, temporal velocity, and temporal acceleration factors remain generic.

VLM may:

- classify visibility or occlusion;
- approve or reject amodal mask completion;
- reject an obviously wrong visible face or physically impossible candidate;
- arbitrate close candidates using the same evidence package.

VLM may not output pose coordinates, quaternion values, or free-form solver weights.

## Failure and fallback behavior

- Missing MegaPose environment: record a blocked external-provider attempt; do not silently fall back to bbox/PCA as an equivalent 6D measurement.
- Low mask confidence: suspend mask-derived updates and rely on tracked visible points plus sequence/contact priors.
- Low track visibility: end the tracking interval explicitly and wait for a reliable reinitialization keyframe.
- Ambiguous MegaPose hypotheses: retain multiple candidates and defer selection; do not average incompatible rotations.
- Occluded interval: propagate the last supported rigid state through generic contact/temporal factors, then reconnect to a post-occlusion MegaPose keyframe.

## Validation without changing accepted output

No new repository test suite is added for this change, following the project boundary. Validation uses isolated diagnostic artifacts:

1. demonstrate that track IDs remain unchanged across frames 127-129 and other former chunk boundaries;
2. report per-frame visible-track count and reinitialization events;
3. render MegaPose hypotheses on visible keyframes before, inside, and after the 118-163 interval;
4. verify mesh rigidity, handle/body proportions, ground support, and broad-face orientation in object-only and overlay renders;
5. compare the isolated candidate against the current accepted trajectory without publishing it;
6. require explicit visual approval before promotion.

## Implementation order

1. Preserve the accepted pose and archive the rejected full-sequence candidate as blocked evidence.
2. Replace independent 32-frame CoTracker restarts with persistent official tracking and stable query IDs.
3. Add SAM-PT-style point-prompt feedback while keeping SAM2 masks as visible, not automatically amodal, evidence.
4. Register MegaPose RGB as a generic external rigid-mesh pose provider.
5. Generate keyframe pose hypotheses and typed provenance for suitcase without running publication.
6. Feed approved hypotheses into one isolated generic sequence attempt.
7. Render and review the candidate before any accepted-output change.

## Non-goals

- No suitcase-specific solver, yaw schedule, interval pose patch, or hand-authored pose.
- No optimization of the human body; GVHMR skeleton remains observation-only support for object reconstruction and HOI visualization.
- No change to loss definitions, canonical thresholds, downstream human refinement, ablation evaluation, or final-result publication in this work unit.
- No remote push.
