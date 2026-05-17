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
├── sharedcam/
│   ├── run_basketball_pose6d_sharedcam.py
│   └── render_pose6d_sharedcam_direct.py
└── human_ball/
    ├── build_human_ball_shared_scene.py
    ├── render_human_ball_fullbody_scene.py
    └── contact/
        └── run_human_ball_contact_phase_calibration.py
```

## Main line

### `sharedcam/run_basketball_pose6d_sharedcam.py`

Shared-camera ball baseline in the GVHMR full-image camera frame.

Outputs:

- `results/pose6d_sharedcam/ball_pose6d_sharedcam_trajectory.csv`
- `results/pose6d_sharedcam/ball_pose6d_sharedcam_reprojection_comparison.csv`

### `sharedcam/render_pose6d_sharedcam_direct.py`

Direct renderer for the shared-camera branch.

Outputs live under:

- `results/renders/pose6d_sharedcam_direct/`
- `results/renders/pose6d_sharedcam_contactphase_direct/`

## Contact-aware branch

### `human_ball/contact/run_human_ball_contact_phase_calibration.py`

Contact-phase calibration on top of the shared-camera baseline.

Outputs:

- `results/pose6d_sharedcam_contactphase/ball_pose6d_sharedcam_contactphase_trajectory.csv`
- `results/pose6d_sharedcam_contactphase/ball_pose6d_sharedcam_contactphase_summary.txt`

## Support scripts

### `tracking/run_cotracker_basketball.py`

Tracking support for `results/tracking/ball_trajectory.csv`.

### `events/align_basketball_events.py`

Audio-visual event extraction used by the contact-aware branch.
