# Energy decomposition — 02_mug_v2

- trajectory: `results/mug_oriented_pose.csv` (kind=mug6d)
- frames: 192 | contact frames: 0 | depth: yes

E_total = Σ wᵢ·Rᵢ  (weights = documented defaults, method_losses.md §3-4)

| term | meaning | weight | RMS | max | weighted RMS | verdict |
|---|---|---|---|---|---|---|
| R_reg(trans) | ‖accel‖ translation | 0.1 | 0.002465  | 0.007867 | 0.0002465 | smooth (max 0.00787) |
| R_reg(rot) | ‖accel‖ rotation | 0.1 | 0.04192  | 0.1766 | 0.004192 | smooth (max 0.177) |
| R_depth | |tz − DA3 metric depth| | 1.0 | 0.09401 m | 0.1252 | 0.09401 | tight (RMS 0.09 m) |

**Σ weighted-RMS ≈ 0.09845** (rough scalar energy level; for cross-frame trend see `energy_terms.png`).

Figures: `energy_terms.png` (per-term over time), `trajectory_3d.png` (3D markers).