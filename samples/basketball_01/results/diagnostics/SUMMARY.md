# Source / loss diagnostics — basketball_01

- trajectory: `samples/basketball_01/results/pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv` (kind=ball)
- masks: yes | depth: yes

## Artifacts

- mask: mask_overlay.mp4 (192/192 frames with mask)
- reproj: reproj_overlay.mp4 (green=2D obs, orange=projected 3D)
- contact: contact_overlay.mp4 (24 human-contact frames)
- curves: loss_curves.png (E_2d, E_contact, depth/DA3, E_smooth)

## Source correctness (fill in after viewing)

| source | looks correct? | how it influences positioning |
|---|---|---|
| SAM2 mask |  |  |
| CoTracker |  |  |
| DA3 depth |  |  |
| 2D reproj (E_2d) |  |  |
| contact (E_contact) |  |  |
| floor/support |  |  |
| temporal smooth |  |  |
| audio events |  |  |

## Verified findings (from overlays + curves, 2026-06-29)

3D animation: `results/renders/full_scene_3d/world.mp4` (paper-style, body+ball+ground).

| source | correct? | influence on positioning |
|---|---|---|
| SAM2 mask | Y | clean track 192/192; fitted circle sits on ball — drives the 2D center obs |
| CoTracker | Y (avail) | `tracking/cotracker_points.csv` present; trajectory uses mask center as primary |
| DA3 depth | partial | tracks trend but ~0.1–0.3 m higher & noisier than solved tz; weak metric anchor |
| 2D reproj (E_2d) | Y | `residual_px ≡ 0` — center reprojection solved exactly; NOT a discriminating term |
| contact (E_contact) | Y | gap→0 at the 24 human-contact frames; oscillates ±0.4 m between bounces |
| floor/support | Y | `floor_v≈668` stable; bottom_proj tracks ball underside |
| temporal smooth | watch | clear |tz acc| spikes, some at contacts → candidates for Step-3 LLM jump fixing |
| audio events | Y | `audio_contact_frame` flags align with bounce frames |
