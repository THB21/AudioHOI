# Results Layout

This folder is currently centered on the shared-camera basketball line.

## Stable groups

- `tracking/`
  - `ball_trajectory.csv`
  - `cotracker_center_trajectory.csv`
  - `cotracker_points.csv`
- `events/`
  - `audio_events.csv`
  - `visual_events.csv`
  - `audio_visual_alignment.csv`
- `depth/`
  - `object_depth.csv` (DA3 metric object depth, scaled to GVHMR)
  - `depth_alignment.json`
- `pose6d_sharedcam/`
  - `ball_pose6d_sharedcam_trajectory.csv` (sphere, size-based depth)
  - `ball_pose6d_sharedcam_reprojection_comparison.csv`
- `pose6d_sharedcam_depthv3/`
  - `ball_pose6d_sharedcam_trajectory.csv` (object-agnostic, DA3 depth)
- `pose6d_sharedcam_contactphase*/`
  - `ball_pose6d_sharedcam_contactphase_trajectory.csv` (e.g. `_depthv3`)
  - `ball_pose6d_sharedcam_contactphase_summary.txt`
- `hands/`
  - `hand_mano_params.pkl`, `stitched_smplx_params.pkl` (HaMeR fingers stitched into the body)
- `joint/`
  - shared human-ball inspection tables
- `renders/`
  - `full_scene_3d/` (`overlay.mp4` + `world.mp4`: body + hands + object)
  - `pose6d_sharedcam_direct/`
  - `pose6d_sharedcam_contactphase_direct/`
- `segmentation/`
  - `masks/`
  - `sam2_jpg_frames/`
- `diagnostics/`
  - auxiliary plots kept from earlier inspections

## Intentionally removed from the active tree

- the old `lifting/` branch
- `lifted_scene_*` render outputs

## Notes

- `pose6d_sharedcam` is the sphere baseline (size-based depth).
- `pose6d_sharedcam_depthv3` is the object-agnostic lift using DA3 metric depth.
- `pose6d_sharedcam_contactphase*` is the contact-aware refinement (anchors + smoothness, no gravity).
- See `../../../method_losses.md` for the loss functions and the generalized energy.
