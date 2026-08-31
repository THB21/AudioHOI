# Generate the nine AudioHOI scenes

This repository contains one script for rendering the nine final 4D
human-object interaction scenes. Every scene is rendered as a clean world view
with the SMPL-X body, animated HaMeR hands, the reconstructed object, a ground
plane, and an orbiting camera.

For every sequence, the script creates

- a VLM render
- a VLM plus audio render
- a labelled side-by-side comparison

The renderer does not create camera overlays, diagnostic side views, contact
markers, coloured hands, or an audio HUD.

## The nine scenes

1. Basketball
2. Football
3. Mug
4. Chair
5. Stick
6. Back-view basketball
7. Volleyball
8. Ping-pong
9. Suitcase

## Generate all scenes

Run this command from the repository root

```bash
/home/hebestreit/miniforge3/envs/gvhmr/bin/python \
  scripts/shared/evaluation/run_nine_visual_vlm_audio_ablation.py \
  --mode both \
  --world-only
```

The command renders both variants and rebuilds every side-by-side comparison.
The full run may take several minutes.

## Output

All videos are written to

```text
deliverables/nine_case_visual_vlm_audio_ablation/world_results/
```

The output structure is

```text
world_results/
├── basketball/
│   ├── vlm/world.mp4
│   ├── vlm_audio/world.mp4
│   └── comparison/world_vlm_vs_vlm_audio.mp4
├── football/
├── mug/
├── chair/
├── stick/
├── back_view_basketball/
├── volleyball/
├── pingpong/
├── suitcase/
└── world_manifest.json
```

`world_manifest.json` lists the final video path for every scene and variant.

## Generate selected scenes

Use a comma-separated case list to render only selected scenes

```bash
/home/hebestreit/miniforge3/envs/gvhmr/bin/python \
  scripts/shared/evaluation/run_nine_visual_vlm_audio_ablation.py \
  --mode both \
  --world-only \
  --cases basketball,mug,chair
```

Available case names are

```text
basketball
football
mug
chair
stick
back_view_basketball
volleyball
pingpong
suitcase
```

## Rebuild comparisons without rerendering

Use `--resume` when all individual videos already exist and only the comparison
videos and manifest need to be refreshed

```bash
/home/hebestreit/miniforge3/envs/gvhmr/bin/python \
  scripts/shared/evaluation/run_nine_visual_vlm_audio_ablation.py \
  --mode both \
  --world-only \
  --resume
```

## Current controlled inputs

- Basketball uses the same accepted Audio-VLM object trajectory in both arms.
- Chair uses the corrected final chair pose in both arms.
- Mug uses the reconstructed mug mesh and its 6DoF rotation.
- Ping-pong includes the ball and the hand-bound paddle.
- Suitcase uses the extended-handle mesh and the audio-guided grasp trajectory
  in the audio arm.
- All scenes use the available stitched HaMeR hand parameters.

The exact trajectory, mesh, and HaMeR mappings are defined in
`scripts/shared/evaluation/nine_modality_ablation.py`.

## Local requirements

The current run expects

- the `gvhmr` Conda environment at
  `/home/hebestreit/miniforge3/envs/gvhmr`
- FFmpeg from the same environment
- the repository assets and reconstructed sample results

All nine sample directories and controlled trajectories are stored under
`samples_known_object`, so the renderer no longer depends on temporary
workstation paths.
