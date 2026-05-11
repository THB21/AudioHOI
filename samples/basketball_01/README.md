# Basketball sample baseline

Folder contract:

```text
samples/basketball_01/
  video.mp4
  frames/
  audio.wav
  results/
```

Prepare sample and frames:

```bash
conda run -n audiohoi python -m scripts.manual_init.prepare_basketball_sample
```

If `ffmpeg` is available, the script also creates `audio.wav`. Otherwise run this inside `samples/basketball_01` after installing ffmpeg:

```bash
ffmpeg -i video.mp4 -vn -ac 1 -ar 16000 audio.wav
```

Run SAM2 segmentation:

```bash
conda run -n audiohoi python -m scripts.manual_init.run_sam2_basketball
```

Run CoTracker point tracking:

```bash
conda run -n audiohoi python -m scripts.shared.run_cotracker_basketball
```

Run audio peak detection and audio-visual alignment:

```bash
conda run -n audiohoi python -m scripts.shared.align_basketball_events
```

Run the 3D lifting baseline:

```bash
conda run -n audiohoi python -m scripts.shared.run_basketball_3d_lifting
```

Render the lifted 3D scene:

```bash
conda run -n audiohoi python -m scripts.shared.render_lifted_scene --view world --out lifted_scene_world.mp4
conda run -n audiohoi python -m scripts.shared.render_lifted_scene --view camera --out lifted_scene_camera.mp4
```

This current branch focuses on:

- explicit `u/v/r -> X/Y/Z` lifting
- reprojection comparison
- true XYZ-based scene rendering

Current outputs:

- `results/segmentation/masks/%05d_mask.png`: SAM2 basketball masks.
- `results/tracking/cotracker_points.csv`: CoTracker center/left/right/top/bottom tracks.
- `results/tracking/ball_trajectory.csv`: main basketball trajectory from SAM2 mask circle fitting.
- `results/events/visual_events.csv`: visual bounce/contact candidates.
- `results/events/audio_events.csv`: impact candidates from audio onset peaks.
- `results/events/audio_visual_alignment.csv`: nearest-neighbor audio/visual event alignment table.
- `results/lifting/ball_3d_lifted_trajectory.csv`: lifted XYZ trajectory.
- `results/lifting/ball_3d_lifted_plot.png`: static 3D lifted trajectory plot.
- `results/lifting/ball_3d_lifted_components.png`: raw-vs-fit XYZ diagnostic plot.
- `results/lifting/ball_3d_reprojection_comparison.png`: observed vs reprojected 2D comparison.
- `results/renders/lifted_scene_world*.mp4`: world-view 3D render.
- `results/renders/lifted_scene_camera*.mp4`: camera-view 3D render.

Folder details for `results/` are documented in:

- `results/README.md`
