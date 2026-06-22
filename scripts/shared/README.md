# Shared Scripts

This shared pipeline is currently organized around the shared-camera basketball branch
and its contact-aware refinement.

## Active layout

```text
scripts/shared/
├── tracking/
│   └── run_cotracker_basketball.py
├── events/
│   └── align_basketball_events.py
├── depth/
│   └── run_depth_anything_v3.py            # DA3 metric depth, scaled to GVHMR
├── sharedcam/
│   ├── run_basketball_pose6d_sharedcam.py  # --depth-source sphere|depthv3
│   └── render_pose6d_sharedcam_direct.py
└── human_ball/
    ├── build_human_ball_shared_scene.py
    ├── render_human_ball_fullbody_scene.py
    ├── render_smplx_pyrender_overlay.py
    ├── render_full_scene_3d.py             # body + hands + object scene
    ├── hands/                              # HaMeR run + stitch into body
    └── contact/
        └── run_human_ball_contact_phase_calibration_anchorinterp_generic.py
```

## Main line

### `depth/run_depth_anything_v3.py`

Per-frame metric object depth from Depth Anything 3, scaled to the GVHMR body. Output:
`results/depth/object_depth.csv` (+ `depth_alignment.json`). Object-agnostic.

### `sharedcam/run_basketball_pose6d_sharedcam.py`

Shared-camera ball baseline in the GVHMR full-image camera frame. `--depth-source sphere`
(legacy size cue) or `--depth-source depthv3` (DA3 depth, object-agnostic).

Outputs:

- `results/pose6d_sharedcam/ball_pose6d_sharedcam_trajectory.csv` (sphere)
- `results/pose6d_sharedcam_depthv3/ball_pose6d_sharedcam_trajectory.csv` (DA3)

### `human_ball/render_full_scene_3d.py`

Body + HaMeR hands + object in one scene; `overlay.mp4` (semi-transparent over the video)
and `world.mp4` (orbiting camera, HOI-paper style) under `results/renders/full_scene_3d/`.

### `sharedcam/render_pose6d_sharedcam_direct.py`

Direct renderer for the shared-camera branch.

Outputs live under:

- `results/renders/pose6d_sharedcam_direct/`
- `results/renders/pose6d_sharedcam_contactphase_direct/`

## Contact-aware branch

### `human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp_generic.py`

Contact-phase calibration on top of the shared-camera baseline: pins object depth to the
contacting body part at contact frames and solves the rest with a smoothness regularizer
(no gravity). Use `--ball-trajectory-csv` to refine the DA3 trajectory.

Outputs (e.g. `--out-subdir pose6d_sharedcam_contactphase_depthv3`):

- `.../ball_pose6d_sharedcam_contactphase_trajectory.csv`
- `.../ball_pose6d_sharedcam_contactphase_summary.txt`

## Support scripts

### `tracking/run_cotracker_basketball.py`

Tracking support for `results/tracking/ball_trajectory.csv`.

### `events/align_basketball_events.py`

Audio-visual event extraction used by the contact-aware branch.
