# Energy decomposition — football_10

- trajectory: `results/pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv` (kind=ball)
- frames: 242 | contact frames: 7 | depth: yes

E_total = Σ wᵢ·Rᵢ  (weights = documented defaults, method_losses.md §3-4)

| term | meaning | weight | RMS | max | weighted RMS | verdict |
|---|---|---|---|---|---|---|
| R_center | center reprojection ‖proj−obs‖ | 1.0 | 0 px | 0 | 0 | OK — reprojection solved exactly (≈0) |
| R_contact | contact depth gap (active@contact) | 1.0 | 0 m | 0 | 0 | gap→0 at contacts (RMS 0.000 m over 7 frames) |
| R_support | floor reproj |bottom−floor| | 0.05 | 18.2 px | 30.98 | 0.9099 | floor offset RMS 18.2px at contacts |
| R_reg | ‖accel‖ translation | 0.1 | 0.07342  | 0.5232 | 0.007342 | smoothness max 0.523 — spikes flag jumps |
| R_depth | |tz − DA3 metric depth| | 1.0 | 2.368 m | 4.662 | 2.368 | weak/large — |tz-DA3| RMS 2.37 m |

**Σ weighted-RMS ≈ 3.285** (rough scalar energy level; for cross-frame trend see `energy_terms.png`).

Figures: `energy_terms.png` (per-term over time), `trajectory_3d.png` (3D markers).