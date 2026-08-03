# Rigid Feature Whole-Sequence Bridge Design

## Goal

Replace the rejected per-frame suitcase PnP bridge with a generic known-rigid-asset sequence solve. The solve consumes named asset features over the whole video while preserving two user-approved trajectory segments byte-for-byte:

- frames 0–124 from `generic-solve-c661e65678cf/state.csv` (Annotation 1 reference),
- frames 164–end from `generic-solve-e1b1a07dff8d/state.csv` (Annotation 2 reference).

Only frames 125–163 may change in the first candidate. The accepted `object_pose.csv` remains immutable until explicit approval.

## Whole-video feature contract

Every frame may receive observations for the same asset-declared rigid feature identities:

- eight body cuboid corners,
- four wheel/support points,
- two handle rails,
- one handle/grasp point.

The feature identities come from the geometry descriptor, not from a suitcase-specific solver branch. Clear frames such as 113, 125 and 165 seed or validate these identities, but tracking and factor compilation cover the complete sequence.

SAM2 provides visible-region membership and contour evidence. CoTracker propagates named feature projections forward and backward. A track is usable only when its forward/backward identity is cycle-consistent, remains compatible with the visible mask or declared occlusion, and agrees with the projected asset neighborhood. A per-frame minimum-area rectangle is never treated as the physical cuboid under occlusion.

## State and factors

The state is one rigid root SE(3) per frame. One sequence solve uses:

- point reprojection for body corners, wheel points and the grasp point,
- axis-line reprojection for the two rails,
- one-sided visible silhouette containment rather than equality to a partial-mask bounding box,
- persistent hand-to-handle contact and relative-velocity factors,
- support and penetration factors for all four wheel points,
- temporal velocity and acceleration factors over SE(3),
- hard reference locks outside frames 125–163.

Occluded measurements are downweighted or disabled; they do not generate a new independent pose. VLM may classify visibility and approve an amodal-mask candidate, but may not output or overwrite SE(3).

## Candidate and rejection rules

The first implementation remains an isolated candidate. It must write a feature ledger, per-factor residuals and a pose CSV. It is rejected if either locked segment differs numerically from its reference, if any wheel penetrates the support plane, or if the free interval contains an unexplained one-frame pose jump. Rendering is required before promotion.

No object-specific optimizer, `case_name` branch, human-state optimization, accepted-pose overwrite, ablation change or downstream human pipeline work is included.
