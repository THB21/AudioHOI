# Energy decomposition — basketball_01

- trajectory: `results/pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv` (kind=ball)
- frames: 192 | contact frames: 24 | depth: yes

E_total = Σ wᵢ·Rᵢ  (weights = documented defaults, method_losses.md §3-4)

| term | meaning | weight | RMS | max | weighted RMS | verdict |
|---|---|---|---|---|---|---|
| R_center | center reprojection ‖proj−obs‖ | 1.0 | 0 px | 0 | 0 | OK — reprojection solved exactly (≈0) |
| R_contact | contact depth gap (active@contact) | 1.0 | 0 m | 0 | 0 | gap→0 at contacts (RMS 0.000 m over 24 frames) |
| R_support | floor reproj |bottom−floor| | 0.05 | 6.887 px | 10.31 | 0.3444 | floor offset RMS 6.9px at contacts |
| R_reg | ‖accel‖ translation | 0.1 | 0.06436  | 0.2085 | 0.006436 | smooth (max 0.208) |
| R_depth | |tz − DA3 metric depth| | 1.0 | 0.2653 m | 0.4831 | 0.2653 | weak/large — |tz-DA3| RMS 0.27 m |

**Σ weighted-RMS ≈ 0.6161** (rough scalar energy level; for cross-frame trend see `energy_terms.png`).

Figures: `energy_terms.png` (per-term over time), `trajectory_3d.png` (3D markers).