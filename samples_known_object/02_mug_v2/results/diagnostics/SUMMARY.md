# Source / loss diagnostics — 02_mug_v2

- trajectory: `samples_known_object/02_mug_v2/results/mug_oriented_pose.csv` (kind=mug6d)
- masks: yes | depth: yes

## Artifacts

- mask: mask_overlay.mp4 (192/192 frames with mask)
- curves: loss_curves.png (6DOF pose components + jump markers + DA3 depth)
- reproj: SKIP (mug pose CSV has no precomputed reprojection residual)
- contact: SKIP (no contact_frame column in mug pose CSV)

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

3D animation: existing proper mesh scene `results/renders/final/world.mp4` (render_full_scene_3d is sphere-proxy/ball-only, so reuse the mug-mesh render).

| source | correct? | influence on positioning |
|---|---|---|
| SAM2 mask | Y | track 192/192; fitted circle on mug body |
| DA3 depth | weak | pose `tz` (1.90–2.05 m) and DA3 depth (2.0–2.11 m) disagree in shape → depth is an unreliable mug constraint |
| 6DOF pose tx/ty/tz | Y | smooth lift-to-mouth-and-back; matches the sip-then-place prompt |
| yaw | **JUMP** | flips ~π (≈3 rad) at f≈23 and back at f≈112 — handle-phase ambiguity, a hard discrete jump (Step-3 target) |
| pitch/roll | not estimated | constant 0 — mug solver leaves them unsolved |
| reproj (E_2d) | N/A | mug pose CSV has no precomputed reprojection residual |
| contact (E_contact) | N/A | no contact_frame column in mug schema |
