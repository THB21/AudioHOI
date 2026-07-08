# Source / loss diagnostics — football_10

- trajectory: `samples/football_10/results/pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv` (kind=ball)
- masks: yes | depth: yes

## Artifacts

- mask: mask_overlay.mp4 (242/242 frames with mask)
- reproj: reproj_overlay.mp4 (green=2D obs, orange=projected 3D)
- contact: contact_overlay.mp4 (7 human-contact frames)
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

3D animation: `results/renders/full_scene_3d/world.mp4`.

| source | correct? | influence on positioning |
|---|---|---|
| SAM2 mask | Y | backfilled this session; tracks juggled ball 242/242, fitted circle on ball |
| DA3 depth | Y | present; metric depth anchor |
| 2D reproj (E_2d) | Y | `residual_px ≡ 0` — solved exactly; not discriminating |
| contact (E_contact) | partial | only 7 foot-contact frames; `contact_depth_gap` negative in free flight (ball above foot) |
| audio events | Y | `audio_contact_frame` present (kick/juggle touches) |
| CoTracker | N/A | not generated for football; mask center is the 2D obs |
