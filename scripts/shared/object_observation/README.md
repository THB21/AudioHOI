# Generic Object Observation Layer

This module turns segmentation and tracking outputs into a generic per-frame
object observation table.

The goal is to separate:

- `2D object observations`
- `object-specific priors`
- `3D trajectory optimization`

So later:

- basketball can consume `enclosing_radius_px`
- mug can consume `bbox / area / silhouette extent`
- hammer can consume `major_axis_px / aspect ratio / tracked endpoints`

## Current inputs

For a sample directory such as `samples/basketball_01`, the script can read:

- `results/segmentation/masks/*.png`
- `results/tracking/cotracker_center_trajectory.csv`
- `results/tracking/cotracker_points.csv`

## Current outputs

It writes:

- `results/object_observations/object_observations.csv`
- `results/object_observations/object_observations_summary.txt`

## Main features

Per-frame generic observations include:

- fused object center
- mask center
- tracking center
- mask area
- bounding box
- enclosing-circle radius
- major / minor axis
- circularity
- tracked anchor points and visibility
- simple confidence scores

## Example

```bash
python scripts/shared/object_observation/build_object_observations.py \
  --sample-dir samples/basketball_01 \
  --object-name basketball
```
