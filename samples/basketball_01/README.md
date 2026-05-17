# Basketball sample shared-camera workflow

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

If `ffmpeg` is available, the script also creates `audio.wav`. Otherwise run:

```bash
ffmpeg -i video.mp4 -vn -ac 1 -ar 16000 audio.wav
```

Run SAM2 segmentation:

```bash
conda run -n audiohoi python -m scripts.manual_init.run_sam2_basketball
```

Run CoTracker point tracking:

```bash
conda run -n audiohoi python -m scripts.shared.tracking.run_cotracker_basketball
```

Run audio peak detection and audio-visual alignment:

```bash
conda run -n audiohoi python -m scripts.shared.events.align_basketball_events
```

Run the shared-camera ball baseline:

```bash
conda run -n audiohoi python -m scripts.shared.sharedcam.run_basketball_pose6d_sharedcam
```

Render the shared-camera baseline with human:

```bash
conda run -n bodyrender python scripts/shared/sharedcam/render_pose6d_sharedcam_direct.py   --sample-dir samples/basketball_01   --ball-csv samples/basketball_01/results/pose6d_sharedcam/ball_pose6d_sharedcam_trajectory.csv   --render-tag pose6d_sharedcam_direct   --with-human
```

Run the contact-phase calibration branch:

```bash
conda run -n bodyrender python scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration.py
```

Render the contact-phase branch with human:

```bash
conda run -n bodyrender python scripts/shared/sharedcam/render_pose6d_sharedcam_direct.py   --sample-dir samples/basketball_01   --ball-csv samples/basketball_01/results/pose6d_sharedcam_contactphase/ball_pose6d_sharedcam_contactphase_trajectory.csv   --render-tag pose6d_sharedcam_contactphase_direct   --with-human
```

Current focus:

- `pose6d_sharedcam`: shared-camera basketball baseline
- `pose6d_sharedcam_contactphase`: contact-aware refinement on top of the shared-camera baseline

Folder details for `results/` are documented in:

- `results/README.md`
