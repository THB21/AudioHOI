# SAM 3D Objects-based Radius Estimation

## Overview

This pipeline estimates the 3D object radius from single-image SAM 3D Objects reconstruction, replacing the hard-coded `radius_m = 0.12` with per-frame data-driven estimates.

**Pipeline:**
```
SAM 2 masks + video frames
    ↓
SAM 3D Objects (3D reconstruction)
    ↓
Sphere fitting → per-frame radius_m
    ↓
sharedcam / contactphase (use estimated radius)
```

## Installation

1. **SAM 3D Objects** is already cloned to `third-party/sam-3d-objects/`

2. **Install dependencies** (if needed):
```bash
cd third-party/sam-3d-objects
pip install -e .
```

3. **Download checkpoints** (automated by SAM 3D Objects on first run)

## Usage

### Step 1: Estimate per-frame radius

```bash
python scripts/shared/sam3d_radius_estimation/estimate_radius_from_sam3d.py \
  --sample-dir samples/basketball_01 \
  --max-frames 10  # optional: test on subset first
```

Output: `results/object_observations/radius_estimates.csv`

This CSV contains:
- `frame`: frame number
- `estimated_radius_m`: radius in meters (or empty if estimation failed)
- `center_x`, `center_y`, `center_z`: 3D object center in camera coordinates

### Step 2: Historical sharedcam usage

The old fixed-radius sharedcam script has been removed from the active tree. This folder is kept only as an archived SAM3D radius-estimation reference.

### Step 3: Use in contactphase (optional)

```bash
python scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp.py \
  --sample-dir samples/basketball_01 \
  --radius-estimates-csv results/object_observations/radius_estimates.csv
```

## Sphere Fitting Details

The `estimate_radius_from_sam3d.py` script:

1. Runs SAM 3D Objects inference on each frame's image + SAM 2 mask
2. Extracts 3D point cloud from reconstruction (either mesh vertices or Gaussian splat means)
3. **Fits sphere** to the point cloud using iterative least-squares:
   ```
   center ← argmin_c { Σ_i (||p_i - c|| - r_mean)^2 }
   radius ← mean(||p_i - center||)
   ```
4. Exports CSV with estimated radius

### Robustness Notes

- **Partial occlusions**: SAM 3D handles moderate occlusion; extreme occlusion may give poor estimates
- **Noise**: Uses mean radius across all reconstructed points (natural smoothing)
- **Frame skipping**: If a frame fails, that row in CSV is left empty; sharedcam will use default radius

## Comparison with Fixed Radius

**Before (fixed):**
- `radius_m = 0.12` (hard-coded)
- Same initialization across all frames
- May be suboptimal if true object size varies or is miscalibrated

**After (data-driven):**
- Per-frame `radius_m` from SAM 3D Objects
- Adapts to actual reconstructed geometry
- Still allows override with `--ball-radius-m` if estimates are bad

## Troubleshooting

### "SAM 3D Objects config not found"
- Checkpoints auto-download on first run
- Or manually download from [Meta SAM 3D](https://ai.meta.com/sam3d/)

### Radius estimates are wildly off
- Check mask quality in `results/segmentation/masks/`
- SAM 3D performs best with clean, well-separated objects
- Try `--max-frames 10` to visually inspect first 10 frames

### Performance is slow
- SAM 3D inference is per-frame, so can be slow for long videos
- Consider running on GPU: ensure CUDA is available
- Can parallelize across frames if needed

## Future Extensions

1. **Robustness to occlusion**: Filter outlier points before sphere fitting
2. **Multi-object support**: Extend to simultaneous radius estimation for multiple objects
3. **Shape adaptation**: For non-spherical objects, estimate bounding box or other shape cues
4. **Temporal smoothing**: Smooth per-frame radius estimates across time for stability
