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
conda run -n gvhmr python scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp_generic.py \
  --sample-dir samples/basketball_01 \
  --body-model-root scripts/third-party/GVHMR/inputs/checkpoints/body_models
```

Render the contact-phase branch with human:

```bash
conda run -n bodyrender python scripts/shared/sharedcam/render_pose6d_sharedcam_direct.py   --sample-dir samples/basketball_01   --ball-csv samples/basketball_01/results/pose6d_sharedcam_contactphase/ball_pose6d_sharedcam_contactphase_trajectory.csv   --render-tag pose6d_sharedcam_contactphase_direct   --with-human
```

## Object-agnostic depth + full 3D scene (current)

These steps use the `gvhmr` env (and `hamer` for hands), per the top-level `CLAUDE.md`.
The math is in `../../method_losses.md`.

```bash
# Metric depth from Depth Anything 3, scaled per-frame to the GVHMR body
conda run -n gvhmr python -m scripts.shared.depth.run_depth_anything_v3 --sample-dir samples/basketball_01

# Lift the object with DA3 depth instead of the sphere size cue (object-agnostic)
conda run -n gvhmr python scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py \
  --sample-dir samples/basketball_01 --depth-source depthv3

# Contact-phase refine on the DA3 trajectory (anchors + smoothness, no gravity)
conda run -n gvhmr python scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp_generic.py \
  --sample-dir samples/basketball_01 \
  --body-model-root scripts/third-party/GVHMR/inputs/checkpoints/body_models \
  --ball-trajectory-csv samples/basketball_01/results/pose6d_sharedcam_depthv3/ball_pose6d_sharedcam_trajectory.csv \
  --support-geometry-json samples/basketball_01/results/pose6d_sharedcam_depthv3/support_geometry.json \
  --out-subdir pose6d_sharedcam_contactphase_depthv3

# Unified 3D scene: body + HaMeR hands + object (HOI-paper style). overlay.mp4 + world.mp4
conda run -n gvhmr python scripts/shared/human_ball/render_full_scene_3d.py \
  --sample-dir samples/basketball_01 --mode both
```

Current focus:

- `pose6d_sharedcam`: shared-camera basketball baseline (sphere, size-based depth)
- `pose6d_sharedcam_depthv3`: object-agnostic lift using DA3 metric depth
- `pose6d_sharedcam_contactphase*`: contact-aware refinement on top (e.g. `_depthv3`)
- `renders/full_scene_3d/`: unified body + hands + object scene (overlay + world)

Folder details for `results/` are documented in:

- `results/README.md`
