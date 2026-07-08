# Per-source analysis — football_10

Each input source analysed separately. ✓/✗ = looks correct; verdict is data-driven.

| source | present | artifact | correct? | verdict | influence on positioning |
|---|---|---|---|---|---|
| SAM2 mask | yes | `mask.mp4 + mask.png` | ✓ | lost 0/242 frames, max center jump 79px, radius CV 0.07 | drives the 2D centre observation (R_center) and the depth-sampling region |
| CoTracker | yes | `cotracker.mp4` | ✓ | 242 frames tracked | tracks sparse object points (constrains R_kp / rotation) |
| DA3 depth | yes | `depth.png` | ✗ | z range 6.44-7.08m, conf 0.07, neg-slope frames 23/242 | metric depth prior (R_depth); per-frame affine to GVHMR body |
| Audio events | yes | `audio.png` | ✓ | 8 audio onsets, 11 visual events, mean A-V offset +0.5 frames (11 aligned) | times/gates the contact residual (R_contact) |
| GVHMR body | yes | `(reuse) renders/full_scene_3d/overlay.mp4` | ✓ | 242 frames, transl accel max 0.048 (jitter) | metric human anchor: hands stitch to it AND DA3 depth is affine-fit to its joints |
| HaMeR hands | yes | `hamer.mp4` | ✓ | detect L 242/242 R 242/242, median fingertip jitter 4.1px/frame | fingertip/palm positions; used for hand-object contact & grasp |
| Object 3D traj | yes | `see diagnostics/energy/*` | ✓ | 242 frames; E_2d RMS 0.00px | the solved object pose (what every other source constrains) |