# Mug Fixed-Rigid Occlusion Design

## Objective

Migrate the canonical mug sequence to the same `GenericSequenceExecutor` used by the other production cases without adding a mug-specific solver or object-name branch.

The mug is a known fixed rigid asset. Its handle does not rotate relative to the body. The observed axial “phase” is the mug's physical rotation about the approximately symmetric body axis and must be represented once in the root rotation, not as an independent component state.

## Scope and invariants

- Object reconstruction only.
- GVHMR hand sites are read-only measurements used to constrain the object.
- No human state, mesh, contact refinement, handoff, or downstream human pipeline is optimized or published.
- Core solver, factor, state, and geometry code must not branch on `case_name` or `mug`.
- The production path must not read solved/final pose or historical phase artifacts.
- Do not change loss families, global hard-gate thresholds, or optimization algorithm.
- Do not add test files. Verification uses compilation, existing diagnostics, isolated solves, factor provenance, quantitative trajectory checks, and full-sequence renders.

## Diagnosed production gap

The current production preparation reads:

```text
observation_seed/body_pose.csv
observation_seed/axial_phase.csv
```

It constructs a nine-dimensional state:

```text
translation(3) + root quaternion(4) + scale(1) + handle phase(1)
```

The same phase seed is then reused as the target of `PeriodicPhasePriorFactor`. No handle reprojection factor enters the continuous solve. This creates two problems:

1. root axial twist and phase encode the same observable orientation;
2. the geometry provider applies phase to the handle feature as if it could rotate relative to the body, although the asset URDF declares `body_to_handle` as fixed.

This is a state-contract and measurement-consumption defect, not a weight-tuning defect.

The current sequence has 240 frames. The old thin-part detector exposes 159 handle centers and 81 missing centers, but those rows flicker inside true occlusion. The dense Qwen visibility artifact marks 60 semantic hidden frames in intervals 27–42, 46–47, 62, 64, 66–88, and 98–114. Applying semantic visibility before numerical factor construction leaves 131 reliable handle-point rows and 109 frames without a handle point. Contact evidence is active for all 240 frames. Occlusion handling is therefore a primary requirement, not a missing-data edge case.

## Selected architecture

### 1. Fixed rigid state contract

The production state is:

```text
root translation(3) + root quaternion(4) + scale(1)
```

There is no independent handle phase DOF. The descriptor declares the body, handle, rim, bottom, contact locations, axial symmetry axis, and fixed root-local feature coordinates.

The former phase is composed into the root quaternion during initialization. Publication exports one physical object pose. Compatibility columns may be decoded from root rotation for legacy renderers, but they are derived outputs and never solver inputs.

### 2. Geometry-capability initializer

Add a generic fixed-rigid initializer capability for an approximately axial body with an off-axis semantic feature. It consumes:

- body center and body bounding box;
- metric depth;
- visible off-axis feature center;
- asset-declared symmetry axis and feature point;
- optional fixed object/contact correspondences.

It generates a finite set of root-pose hypotheses. Body observations determine translation, scale, and tilt. The visible off-axis feature resolves rotation about the symmetry axis. The initializer records all input measurement IDs, hypothesis scores, visibility decisions, and selected state in its ledger.

This capability is geometry-driven and reusable for bottles with a fixed spout, kettles with a fixed handle, or other approximately axial rigid assets. It does not import mug components or inspect the case name.

### 3. Typed production factors

The single generic solve consumes:

- `PointReprojectionFactor` for object/body center;
- `PointReprojectionFactor` for the fixed off-axis feature when visible and reliable;
- `MetricDepthFactor` for object depth;
- `ContactDistanceFactor` from read-only hand sites to fixed object contact features;
- `ContactRelativeVelocityFactor` during persistent grasp;
- `ContactTwistGaugeFactor` when two independent contact sites are available;
- `TemporalVelocityFactor` and `TemporalAccelerationFactor`;
- `StaticFreezeFactor` only in inferred supported-static intervals.

`PeriodicPhasePriorFactor` is removed from the accepted production path because it currently targets a derived seed rather than an independent observation.

### 4. Handle occlusion state

Handle visibility is frame-local evidence:

```text
visible and reliable
    -> enable fixed-feature reprojection

occluded / absent / unreliable
    -> disable feature reprojection
    -> never write a zero target
    -> never interpolate a fake visible measurement
```

During a persistent grasp with the handle occluded, object state is propagated by the rigid relationship between:

- the previous accepted root pose;
- the asset-declared fixed handle/contact geometry;
- current read-only hand-site position and velocity;
- temporal velocity and acceleration constraints.

The state is `occluded_hold`, not a visible handle observation. Contact distance prevents the fixed handle from leaving the hand; contact relative velocity suppresses hand/object slip; temporal factors regularize any remaining monocular gauge. When the handle becomes visible again, reprojection is reactivated and corrects drift through the same continuous solve.

If hand contact provenance becomes unreliable during an occluded interval, the solver must lower the predefined contact activation tier and preserve multiple finite root hypotheses or block publication. It must not invent a phase, ask VLM for a quaternion, or silently keep a stale handle pixel target.

### 5. VLM role

VLM may classify only discrete evidence states:

- handle visible / partial / occluded by hand / absent / unclear;
- hand contacts handle / body / unclear;
- candidate has obvious floating, penetration, or wrong-side handle placement.

VLM may downweight a predefined visual or contact factor tier and may reject a candidate. A visible label enables an existing pixel measurement but does not raise that pixel detector's confidence; an occluded label removes the feature reprojection and changes persistent contact to `occluded_hold + keep_previous`. VLM cannot output pose, phase, continuous weights, or override hard geometry metrics.

### 6. Metric-asset scale responsibility

Body silhouette and metric depth estimate one sequence-level asset scale. For a known metric rigid asset this scale is constant and is not an object-motion degree of freedom. The axial initializer must read scale bounds from `StateSpec`, compute a robust sequence estimate from body bbox/depth, and publish it as an unobservable fixed DOF for the subsequent trajectory solve. Contact factors cannot change scale to manufacture a zero hand gap.

### 7. Publication and provenance

Every isolated attempt writes state, residuals, factor ledger, hard metrics, visibility/contact gates, and status. `AcceptedObjectOutputPublisher` remains the only writer of canonical `object_pose.csv`.

The accepted rows must record:

```text
source=generic_sequence_executor
case_dispatch_used=false
human_state_optimized=false
```

Any automatic hard-gate failure remains in publication provenance even if the user later authorizes visual promotion.

## Verification and acceptance

No new test files are added. The implementation is accepted only after all of the following evidence is produced:

1. modified Python modules compile in the `audiohoi` environment;
2. a fresh mug preparation reports no read of baseline/final pose or historical phase;
3. the selected production state has eight values per frame and no independent phase DOF;
4. core solver/factor/state/geometry diffs contain no mug or case-name branch;
5. visible-handle frames execute fixed-feature reprojection, while all 81 currently missing frames have no fabricated pixel residual;
6. occluded-contact intervals execute contact distance, relative velocity, and temporal factors with complete input provenance;
7. the isolated 240-frame solve produces finite states, normalized quaternions, bounded scale, and no structural handle/body separation;
8. quantitative reports include handle reprojection on visible frames, contact gap and relative velocity in occluded-hold frames, translation/rotation velocity, acceleration, and reappearance correction;
9. full 240-frame object-only and read-only-skeleton relationship renders are produced in overlay, camera-3D, and side/depth views;
10. the user reviews the final videos before canonical publication.

## Explicit non-goals

- No independently articulated mug handle.
- No mug-named continuous solver or least-squares optimizer.
- No restoration of solved pose/phase seeds as production inputs.
- No human pose refinement.
- No threshold relaxation to obtain promotion.
- No open-ended self-repair loop or new downstream orchestration.
