# HOI Interaction Evaluation Method (human-object layer)

Companion to `docs/final_result_evaluation_method_en.md` (Yixin's object-centric final
evaluator). That layer scores the OBJECT (SE3 schema, overlay, anchor drift, jumps). This
layer scores the INTERACTION between the reconstructed human and the object — the part the
object evaluator marks as missing ("more detailed HOI interaction evaluation and full
human-object penetration evaluation").

Implementation: `scripts/shared/evaluation/compute_hoi_interaction_metrics.py`
Cross-case table: `samples_known_object/hoi_interaction_evaluation/hoi_summary.csv`
SOTA grounding: `docs/research_hoi_eval_metrics_sota.md` (verified citations; the metric
names below follow the field conventions found there).

## Evaluated reconstruction

- Human: GVHMR SMPL-X (camera frame) + HaMeR fingers stitched in + **body-side contact
  refinement** (`refine_body_pose_contact.py`) — params precedence contact_refine >
  stitched > raw, same as the renderers.
- Human contact geometry: per-part skeleton point clouds (hands = wrist + 15 finger joints
  + 5 fingertips; feet = ankle/foot/toes/heel), not just part centers.
- Object: per-frame surface geometry from the trajectory CSV via
  `scripts/shared/human_ball/contact/object_geometry.py`:
  - `sphere` (balls, mug proxy r=0.048) — center + radius;
  - `capsule` (stick: SE3 pose columns, local **x** axis, L=1.86 m, r=0.018 from URDF) —
    differentiable distance to the rotated axis segment.
  - Mesh-SDF support is the queued upgrade (EasyHOI-style; needed for chair and exact mug).
- Expected contacts: trajectory contact columns (`contact_frame`, `audio_contact_frame`,
  `human_contact_state`) ∪ audio contact records (`target_entity`, falling back to
  `stable_entity` for sustained grasps / `keep_grasp` states).

## Metric battery

**A. Penetration (GT-free plausibility; PSI/ObMan/CHOIS conventions)**
- `pen_frame_ratio` / `non_collision_ratio` — frames with any part point deeper than
  3 mm (skeleton-to-skin slack) inside the surface. Saturates near 1.0 → co-report depth.
- `pen_depth_mean_mm`, `pen_depth_max_mm` — over penetrating (frame, point) pairs.
  Reference bar: InterCap's GT capture shows ~7 mm mean penetration at contact.

**B. Contact (must be co-reported with A — OMOMO/CHOIR gaming pair)**
- `contact_frame_ratio` — closest part point within surface ± 15 mm.
- `contact_gap_mm` — mean |closest distance − radius| over expected-contact frames
  (CHOIR "H-O dist": exposes floating hands that penetration alone rewards).
- `part_correct_ratio` — at expected-contact frames with a named part, is the
  geometrically closest part the named one? (HOI-PAGE part-level accuracy precedent;
  labels are our audio→VLM records, analogous to HOI-PAGE's LLM part pairs.)

**C. Temporal / physics**
- `object_jerk` — mean ‖Δ³T‖² (co-report with overlay/2D fit; over-smoothing caveat).
- `grasp_stability_mm` — std of closest distance over contact runs (held object should
  keep constant hand-surface distance).
- `mdev_star_mm` — **GT-free adaptation of ARCTIC MDev**: during predicted contact runs,
  mean ‖δpart − δobject‖ per frame — does the object move WITH the hand? (mug/stick hold).

**Tier-3 (novel, audio-aware — ours)**
- `contact_ratio_audio_windows` — fraction of audio events with actual geometric contact
  within ±2 frames. This is the audio-visual consistency number no prior work reports.
- `accel_at_events` vs `accel_in_flight` — an acceleration spike AT an audio event is
  physics (impact); a spike elsewhere is an artifact. Complements Yixin's jump count,
  which cannot distinguish the two.

**Planned next (from the research pass):** vertex-level SDF penetration (mesh objects),
foot-skate ratio + ground penetrate/float (GMD/PhysDiff forms) for chair/football,
intersection volume for grasps, rendered-silhouette mIoU alignment with Yixin's overlay
proxy, and the 2AFC user study protocol (PHOSA form) full-vs-no-audio.

## Current five-case reading (2026-07-08 first pass)

| case | pen ratio | pen max mm | contact gap mm | part correct | C_audio | note |
|---|---|---|---|---|---|---|
| basketball | 0.000 | 0 | 20.4 | 0.97 | 1.00 | after body refine (69/69 penetrations cleared) |
| football | 0.000 | 0 | 144.0 | 0.86 | 0.45 | gap driven by kick-frame timing + foot pose mismatch — the real diagnostic her floating-proxy could not localize |
| mug | 0.000 | 0 | 15.8 | 1.00 | 0.57 | sphere proxy (r 4.8 cm); handle grasp needs mesh SDF |
| stick | 0.033 | 7.4 | 54.2 | 0.88 | 0.50 | capsule; 61/80 penetrations cleared, residual ≤7 mm |
| chair | 0.229 | 8.9 | 3.7 | 0.67 | 1.00 | true mesh SDF (baked URDF); AFTER differentiable-SDF Stage C (86/91 hand/foot penetrations cleared, 0.401→0.229); residual = torso leaning on backrest (hands/feet-only mask → PROX body sets queued) |

The `mesh` geometry (`object_geometry.MeshGeometry`, trimesh signed distance on the
`--keep-origin` baked URDF GLB) is metrics-only; Stage C still needs a differentiable
surface (sphere/capsule today, SDF grid next).

Contrast with the object-centric proxies: her summary reports Penetration Rate ≈ 1.0 for
basketball/football/mug from `contact_depth_offset_m` sign — the direct geometric
measurement shows zero actual human-object interpenetration after body-side refinement.
The proxy measures depth-offset bookkeeping, not interaction quality; these columns should
replace it in the final report.
