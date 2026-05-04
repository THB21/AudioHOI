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
conda run -n audiohoi python -m scripts.prepare_basketball_sample
```

If `ffmpeg` is available, the script also creates `audio.wav`. Otherwise run this inside `samples/basketball_01` after installing ffmpeg:

```bash
ffmpeg -i video.mp4 -vn -ac 1 -ar 16000 audio.wav
```

Run SAM2 segmentation:

```bash
conda run -n audiohoi python -m scripts.run_sam2_basketball
```

Run CoTracker point tracking:

```bash
conda run -n audiohoi python -m scripts.run_cotracker_basketball
```

Run audio peak detection and audio-visual alignment:

```bash
conda run -n audiohoi python -m scripts.align_basketball_events
```

Current outputs:

- `results/masks/%05d_mask.png`: SAM2 basketball masks.
- `results/cotracker_points.csv`: CoTracker center/left/right/top/bottom tracks.
- `results/ball_trajectory.csv`: CoTracker basketball center trajectory.
- `results/visual_events.csv`: visual bounce/contact candidates from local maxima in y-position.
- `results/audio_events.csv`: impact candidates from audio onset peaks.
- `results/audio_visual_alignment.csv`: nearest-neighbor audio/visual event alignment table.
