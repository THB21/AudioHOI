# VLM and Audio Evidence Gain Design

## Objective

Demonstrate a visible and causally attributable improvement from VLM and audio
in the object-only reconstruction mainline.  The initial demonstration uses the
suitcase sequence because it contains both a rigid-orientation ambiguity and an
irregular rolling-motion schedule.  The implementation must remain generic:
core solver code may branch on typed state, geometry capabilities and evidence
availability, but not on a case name.

The intended research claim is:

> VLM resolves discrete visibility, face-identity and orientation-topology
> ambiguity; audio resolves motion interval and transition timing ambiguity;
> one geometry-aware sequence solver computes the continuous object trajectory.

The goal is a clearly visible difference between `full`, `no_vlm`, `no_audio`
and `vision_only`, without adding noise to an ablation or manually editing its
pose.  A modality may help only by contributing a real, hashed artifact that is
consumed by a declared factor or gate.

## Scope and Boundaries

- Object reconstruction only.
- GVHMR is read-only evidence for object contact, relative motion and relation
  rendering.  Human state is never optimized.
- VLM does not output camera-space or world-space translation, rotation,
  quaternion, joint state or a free loss weight.
- Audio does not output an object position or orientation.
- Manual statements such as the suitcase face identity, turn direction or
  occlusion interval are evaluation annotations only.  They are not pipeline
  inputs.
- Core code may not contain `suitcase`, a suitcase case name, or a frame-number
  exception.  Asset configuration may declare semantic faces, rails, wheels,
  grasp features and rolling support.
- No new pytest files are required.  Verification uses real Qwen artifacts,
  the real audio track, factor ledgers, numerical assertions and rendered
  videos.
- Existing canonical outputs remain unchanged until explicit video approval.

## Selected Architecture

The selected design is interaction-state-conditioned factor activation.
Machine observations first generate a finite set of geometrically valid
hypotheses.  VLM and audio produce typed discrete evidence.  The
`FactorCompiler` converts that evidence into active masks and predefined weight
tiers for one `GenericSequenceExecutor`.

```text
SAM2 / CoTracker / DA3 / read-only GVHMR / audio
                         |
                         v
       Measurements + uncertainty intervals
                         |
          +--------------+--------------+
          |                             |
          v                             v
  VLM SemanticRelationIR          AudioEventIR
  face / facing / turn /          motion / silence /
  visibility / grasp              onset / offset / tug
          |                             |
          +--------------+--------------+
                         v
               InteractionStateIR
                         |
                         v
                FactorCompiler
                         |
                         v
              GenericSequenceExecutor
                         |
                         v
       hard gates -> candidate selector -> publisher
```

VLM and audio never select an object-specific solver.  Their only effect is to
activate generic factor kinds over typed frame intervals.

## Asset Semantic Contract

The asset descriptor declares semantic features in local coordinates.  The
names below are roles rather than suitcase-specific solver branches:

- `face:grasp_side_wide`
- `face:opposite_wide`
- `face:side_left`
- `face:side_right`
- `feature:grasp_handle`
- `feature:parallel_rail_left`
- `feature:parallel_rail_right`
- `feature:rolling_support_points`
- `support_model.mode = rolling_axle`

Other rigid assets can provide equivalent role mappings.  The VLM prompt is
generated from the available roles and does not embed a hard-coded category
vocabulary in core code.

## Automatic VLM Query Triggering

VLM queries are created from machine uncertainty, not manually supplied frame
numbers.  A frame contributes to an uncertainty interval when one or more of
the following signals exceed a declared threshold:

1. visible mask area or mask completeness drops sharply;
2. tracked rail or support-feature coverage drops;
3. broad-face and side-face hypotheses have similar hard geometry scores;
4. neighboring heading hypotheses disagree in winding direction;
5. the estimated human occluder overlaps a large fraction of the object mask;
6. the hand-handle relation becomes inconsistent or disappears.

The trigger groups adjacent uncertain frames into intervals.  Each interval
queries its start, peak-uncertainty and end frames, together with the closest
clear frame before and after the interval.  This produces a bounded number of
queries and lets the VLM reason from visible bracketing evidence.

Each evidence package contains:

- the original temporal RGB strip;
- the visible SAM2 mask, without pretending it is amodal truth;
- CoTracker rail and rolling-support tracks;
- the read-only human occlusion layer and hand site;
- anonymized projections of the finite orientation hypotheses;
- hard metric summaries that do not reveal an accepted or legacy pose.

## VLM Output Contract

Every query is forced choice.  The production labels are:

- `visible_face`: grasp-side-wide, opposite-wide, side-left, side-right,
  unclear;
- `facing_relation`: grasp-side-toward-human, away-from-human, side-on,
  unclear;
- `turn_direction_screen`: counterclockwise, clockwise, stationary, unclear;
- `visibility`: visible, partial, human-occluded, absent, unclear;
- `grasp_state`: active, released, unclear.

An optional amodal mask completion remains a 2D measurement operation.  It is
stored separately from semantic relations, must preserve every visible mask
pixel, and may not be interpreted as a continuous 3D pose.

`unclear`, malformed, low-confidence or temporally contradictory outputs do not
activate a semantic factor.  At most two VLM query rounds are allowed for one
uncertainty interval.

## VLM Factor Mapping

The compiler maps accepted semantic evidence to generic factors:

- face identity -> `FaceVisibilityInequalityFactor`;
- face-to-human relation -> `FacingRelationFactor`;
- screen turn direction -> `HeadingTopologyFactor`;
- human occlusion -> visual-factor attenuation plus rigid/contact propagation;
- persistent grasp -> contact distance, relative velocity and local-anchor
  constancy;
- release -> removal of persistent-grasp factors.

The VLM chooses labels, not weights.  Each factor kind has fixed, documented
`low`, `medium` and `high` tiers selected by confidence and cross-modal
agreement.

## Audio Event and Envelope Contract

The existing suitcase artifact contains 21 isolated peaks.  Peaks alone cannot
represent continuous rolling, silence or irregular speed.  Stage 0 therefore
adds a sustained-motion envelope and interval-level events:

- `sustained_motion`;
- `silence`;
- `motion_onset`;
- `motion_offset`;
- `short_tug`;
- `seam_click`;
- `unknown`.

Each event records start, peak and end time, confidence, SNR, energy, band
profile, source-audio hash and extractor version.  Audio does not assume that
amplitude is linearly proportional to object speed.

## Audio Factor Mapping

- sustained rolling activates tangential-motion preservation and weakens
  static freeze;
- silence strengthens static freeze only when visual motion does not clearly
  contradict it;
- onset and offset create bounded transition windows and relax acceleration
  smoothing near the boundary;
- a short tug preserves a local velocity or acceleration peak;
- seam clicks are timing evidence only and cannot directly move the object.

If high-confidence audio and high-confidence visual motion conflict, neither
modality silently overrides the other.  The factor is withheld, the conflict is
recorded and the solver falls back to visual/depth/support/temporal factors.

## Candidate Generation and Selection

All variants use the same hypothesis count, initializers, solver, bounds and
hard gates.  Candidate selection proceeds in this order:

1. reject NaN, invalid schema, rigid-part separation, scale change, excessive
   penetration, support failure or motion-step failure;
2. among hard-pass candidates, reject semantic face, facing, visibility and
   heading-topology contradictions when effective VLM evidence exists;
3. evaluate audio motion/silence/transition consistency when effective audio
   evidence exists;
4. rank remaining candidates by mask, rail/support reprojection,
   visible-surface depth, contact, support and temporal metrics;
5. atomically publish only the selected candidate.

Hard geometry always outranks a VLM preference.  VLM and audio may break a
genuine ambiguity but may not rescue a physically invalid candidate.

## Ablation Matrix

| Variant | VLM Semantic IR | Audio Interval IR | Expected visible behavior |
| --- | --- | --- | --- |
| `full` | enabled | enabled | correct orientation topology and irregular motion timing |
| `no_vlm` | unavailable | enabled | geometric face/heading ambiguity can remain during occlusion |
| `no_audio` | enabled | unavailable | spatial branch can be correct while short tug and stops are over-smoothed |
| `vision_only` | unavailable | unavailable | both spatial ambiguity and timing loss can remain |

`no_vlm` must not call or consume the VLM provider.  `no_audio` must not
generate or consume AudioEventIR.  The run manifest must reflect actual
behavior, not merely a directory name.

## Provenance

Every VLM record contains:

- query ID, frame window and trigger scores;
- evidence package paths and SHA-256 values;
- prompt schema, choices, provider, model and version;
- raw response hash, normalized labels, confidence and validation result;
- every consuming factor ID, interval, weight tier and active mask.

Every audio record contains:

- source-audio SHA-256 and extractor version;
- interval, type, confidence, SNR, energy and band profile;
- every consuming factor ID, interval, weight tier and active mask.

Expected production artifacts are:

- `semantic_relations.jsonl`;
- `audio_events.jsonl`;
- `interaction_timeline.jsonl`;
- `vlm_query_triggers.csv`;
- `vlm_results.jsonl`;
- `factor_activation_ledger.csv`;
- `candidate_selection.json`;
- per-attempt states, residuals, hard metrics and status;
- one object pose and object render set for each ablation.

## Evaluation

Manual annotations are held out from all pipeline inputs.  For the suitcase
demonstration they include visible-face identity, face-to-human relation,
screen turn direction, visibility, grasp/release state, motion intervals and
start/stop boundaries.

VLM metrics:

- visible-face accuracy;
- grasp-side-wide-to-human relation accuracy;
- heading-topology accuracy;
- wrong-face frame count and longest wrong-face run;
- occlusion reattachment error;
- number of incorrect hypotheses blocked by effective VLM evidence.

Audio metrics:

- motion interval F1;
- onset and offset timing error;
- displacement during complete stops;
- short-tug speed-peak preservation;
- rolling-envelope/tangential-motion interval agreement;
- non-event acceleration spikes.

Geometry and video metrics remain modality-independent:

- mask and visible feature overlay;
- rail and rolling-support reprojection error;
- visible-surface depth consistency;
- support gap and penetration;
- rigid consistency;
- translation and rotation step limits;
- trajectory acceleration and jerk.

## Acceptance Criteria

The initial suitcase demonstration is accepted only when:

1. `full` passes every hard geometry and physics gate;
2. every effective VLM/audio factor is traceable to a real hashed artifact;
3. `full` reduces wrong face or heading-topology frames by at least 50% relative
   to `no_vlm` on the held-out annotations;
4. `full` reduces complete-stop drift by at least 50% relative to `no_audio` and
   improves onset/offset timing;
5. the full-vs-ablation differences are visible in synchronized overlay and 3D
   videos without manually editing an ablation pose;
6. the same executor, candidate budget, bounds and hard-gate policy are used in
   every variant;
7. core solver/factor/state/geometry code contains no case-name branch;
8. `human_state_optimized` is false;
9. canonical outputs are not replaced until explicit video approval.

If VLM or audio does not meet these criteria, the result is reported as a
negative or inconclusive modality result.  The implementation must not amplify
the difference by corrupting an ablation.

## Implementation Scope

The implementation is intentionally limited to five production changes:

1. extend typed semantic and audio interval schemas;
2. add uncertainty-triggered, profile-driven VLM evidence generation;
3. add sustained audio envelope extraction;
4. compile semantic/audio evidence into generic factor activation;
5. run and render the four suitcase variants with complete provenance.

Promotion of the isolated v70 trajectory into the canonical mainline and work
on volleyball or table-tennis cases are separate decisions after this evidence
path is verified.
