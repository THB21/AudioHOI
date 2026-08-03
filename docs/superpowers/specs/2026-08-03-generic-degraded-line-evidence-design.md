# Generic Degraded Line Evidence Design

## Scope

Recover rigid-object orientation through partial visibility without object-name
dispatch, manual frame poses, or contact factors after release. The immediate
failure is a rigid asset whose two declared parallel feature lines become one
visible line before full occlusion.

## Evidence hierarchy

1. When two bounded parallel lines are visible, retain the existing paired-line
   observation and fixed feature correspondence.
2. When only one sufficiently long line remains, publish one unassigned axis-line
   observation. The solver compares it with every descriptor-declared candidate
   line and uses the lowest axis-line residual; it does not guess left versus
   right identity.
3. When no line is reliable, use the existing binary-mask principal axis only as
   a weak orientation-continuity observation. It cannot establish face identity.
4. Independently reject every tracked object feature that lies outside the
   current object mask. Grasp features receive no exemption.

## Interaction-state boundary

The line and mask observations are visual measurements and remain independent of
contact state. At a confirmed release, grasp reprojection, grasp-facing, and hand
co-motion factors become inactive. Visual line, mask, support, penetration, and
temporal factors remain eligible according to their own reliability.

## Single-line contract

The observation artifact may contain either two rows with
`line_observation_mode=paired` or one row with
`line_observation_mode=unassigned_axis`. A single row records endpoints,
confidence, visibility, source mask, and the descriptor-declared candidate
feature set. The solver evaluates point-to-line and direction residuals against
each candidate projected line and selects the minimum-cost candidate per
residual evaluation.

## Failure handling

Single-line evidence is rejected when its span is below the configured minimum,
its direction is inconsistent with the feature-region geometry, or its mask
support is absent. The fallback must never create a line by interpolating a
historical pose. Missing evidence falls through to mask principal axis and
temporal/support factors.

## Acceptance

- Frames outside the free interval remain exactly locked.
- No grasp-related factor is active after the inferred release.
- No mask-incompatible named feature is consumed.
- The 145-to-146 transition does not switch to a different yaw branch.
- Support penetration stays below 1 cm and the accepted pose artifact remains
  byte-identical until explicit approval.
- Validation uses real factor ledgers, trajectory statistics, and rendered
  overlays; no new pytest suite is added.

