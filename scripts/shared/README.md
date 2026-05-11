# Shared Scripts

This folder contains the shared post-segmentation steps used by both:

- `samples/basketball_01`
- `samples_known_object/*`

## Current scripts

### `run_cotracker_basketball.py`

Shared tracking step.

Inputs:

- `results/segmentation/masks/`
- `frames/`

Outputs:

- `results/tracking/ball_trajectory.csv`
- `results/tracking/cotracker_center_trajectory.csv`
- `results/tracking/cotracker_points.csv`

Notes:

- The current main `ball_trajectory.csv` is derived from SAM2 mask circle fitting.
- CoTracker outputs are kept as auxiliary tracking signals.

### `align_basketball_events.py`

Shared event extraction and audio-visual alignment step.

Inputs:

- `results/tracking/ball_trajectory.csv`
- `audio.wav`

Outputs:

- `results/events/audio_events.csv`
- `results/events/visual_events.csv`
- `results/events/audio_visual_alignment.csv`

### `run_basketball_3d_lifting.py`

Minimal monocular 3D lifting baseline for the basketball sample.

Inputs:

- `results/tracking/ball_trajectory.csv`
- `results/events/visual_events.csv`

Outputs:

- `results/lifting/ball_3d_lifted_trajectory.csv`
- `results/lifting/ball_3d_reprojection_comparison.csv`
- `results/lifting/ball_3d_reprojection_comparison.png`

### `evaluate_reprojection.py`

Evaluation helper for checking whether the lifted 3D trajectory projects back to the observed 2D track.

### `render_lifted_scene.py`

True XYZ-based lifted-scene visualization.

Inputs:

- `results/lifting/ball_3d_lifted_trajectory.csv`

Views:

- `world`
- `camera`

Outputs:

- `results/renders/lifted_scene_world*.mp4`
- `results/renders/lifted_scene_camera*.mp4`

## Notes

- Old pseudo-3D render experiments are no longer part of the active shared pipeline.
- The active basketball 3D path is:

```text
mask -> ball_trajectory.csv -> visual/audio events -> 3D lifting -> lifted scene render
```
