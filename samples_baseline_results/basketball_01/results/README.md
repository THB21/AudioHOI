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
- `pose6d_sharedcam/`
  - `ball_pose6d_sharedcam_trajectory.csv`
  - `ball_pose6d_sharedcam_reprojection_comparison.csv`
- `pose6d_sharedcam_contactphase/`
  - `ball_pose6d_sharedcam_contactphase_trajectory.csv`
  - `ball_pose6d_sharedcam_contactphase_summary.txt`
- `joint/`
  - shared human-ball inspection tables
- `renders/`
  - `pose6d_sharedcam/`
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

- `pose6d_sharedcam` is the baseline.
- `pose6d_sharedcam_contactphase` is the contact-aware refinement branch.
