# Basketball 3D Lifting Flow

This note describes the current **basketball 3D lifting baseline** only.

## 1. Goal

Recover a minimal 3D ball-center trajectory from a monocular basketball video:

```text
video -> mask / 2D trajectory -> UVZ estimation -> 3D lifting -> scene rendering
```

The final target is a per-frame 3D state:

```text
(X, Y, Z)
```

where:

- `X`: lateral position
- `Y`: height above the ground
- `Z`: forward depth from the camera

---

## 2. Current input files

The current pipeline mainly depends on the following files.

### 2.1 `results/segmentation/masks/*.png`

Frame-wise basketball masks produced by SAM2.

Used for:

- locating the ball in the image
- extracting the image-space center and apparent radius

### 2.2 `results/tracking/ball_trajectory.csv`

The current main trajectory comes from **mask circle fitting**, not from the CoTracker center track.

Main fields:

- `frame`
- `time`
- `ball_center_x`
- `ball_center_y`
- `radius`
- `source = sam2_mask_circle`

Here:

- `ball_center_x -> u`
- `ball_center_y -> v`
- `radius -> r`

### 2.3 `results/events/visual_events.csv`

Contains visually detected bounce/contact frames.

Used for:

- estimating the floor-contact timing
- constraining the later fit in the `Y` direction

---

## 3. Step 1: extract image-space geometry from the mask

Script:

- [scripts/shared/run_cotracker_basketball.py](/mnt/hdd/AudioHOI/scripts/shared/run_cotracker_basketball.py)

The current main trajectory is not taken from the raw mask centroid. Instead, each mask is fitted with a circle:

```text
mask -> fitted circle -> (u, v, r)
```

This gives, for each frame:

- `u`: horizontal image coordinate of the ball center
- `v`: vertical image coordinate of the ball center
- `r`: apparent radius in image space

The purpose is to feed the lifting stage with a more self-consistent ball geometry.

---

## 4. Step 2: UVZ estimation logic

Script:

- [scripts/shared/run_basketball_3d_lifting.py](/mnt/hdd/AudioHOI/scripts/shared/run_basketball_3d_lifting.py)

### 4.1 Assumptions

The current baseline uses the following minimal assumptions:

1. The real basketball radius is fixed:

```text
R = 0.12 m
```

2. The camera is approximated as a pinhole camera.

3. The focal length is approximated as:

```text
f_x = f_y = 0.9 * image_width
```

For a 1280-pixel-wide video:

```text
f ≈ 1152 px
```

4. The principal point is approximated at the image center:

```text
c_x = image_width / 2
c_y = image_height / 2
```

### 4.2 Estimating Z

Since the real ball radius is assumed known and fixed, and the apparent radius can be measured from the image, the current depth estimate is:

```text
Z_raw = f * R / r
```

Interpretation:

- larger `r` -> the ball is closer -> smaller `Z`
- smaller `r` -> the ball is farther -> larger `Z`

This is the current **monocular size-based depth estimation** step.

### 4.3 Estimating X

Once `Z` is available, the horizontal image coordinate is back-projected to 3D:

```text
X_raw = (u - c_x) * Z_raw / f_x
```

Interpretation:

- ball on the left in the image -> negative `X`
- ball on the right in the image -> positive `X`

### 4.4 Estimating Y

`Y` is not obtained by directly projecting `v`. Instead, it is estimated from the ball-bottom displacement relative to the floor.

First compute the image-space ball bottom:

```text
bottom_v = v + r
```

Then estimate a floor image coordinate from the visual contact frames:

```text
floor_v = median(bottom_v at contact frames)
```

Then convert the bottom-to-floor pixel offset into height:

```text
Y_raw = R + max(0, floor_v - bottom_v) * Z_raw / f_y
```

Interpretation:

- at ground contact, the ball-center height should be close to the ball radius `R`
- the farther the ball bottom is above the floor, the larger the ball height

So the current `Y` is a:

```text
ground-anchored height estimate
```

rather than a direct projection of image-space `v`.

---

## 5. Step 3: 3D trajectory reconstruction and fitting

The script does not use `X_raw, Y_raw, Z_raw` as the final result directly. It performs a second fitting stage.

### 5.1 Why fitting is needed

The raw per-frame estimates are affected by:

- mask jitter
- radius estimation error
- local noise around contact frames

So the trajectory is regularized into a smoother and more motion-consistent 3D path.

### 5.2 Fitting strategy

The trajectory is split into segments using bounce/contact frames as boundaries.

#### X fitting

Within each segment:

```text
X -> linear fit
```

#### Z fitting

Within each segment:

```text
Z -> linear fit
```

#### Y fitting

Within each segment:

```text
Y -> quadratic fit
```

while enforcing:

```text
Y = R
```

at contact frames.

This means the ball-center height must equal the ball radius when the ball touches the floor.

So the current `Y` fitting is essentially:

```text
contact-constrained parabolic fitting
```

---

## 6. Current output files

### 6.1 `results/lifting/ball_3d_lifted_trajectory.csv`

This is the main output of the current 3D lifting stage.

Main fields:

- `u, v, radius_px`
- `X_raw, Y_raw, Z_raw`
- `X, Y, Z`

Here:

- `raw` means direct per-frame estimates from the formulas
- `X, Y, Z` means the final fitted and smoothed trajectory

### 6.2 `results/ball_3d_lifted_plot.png`

A standard 3D trajectory plot.

### 6.3 `results/ball_3d_lifted_components.png`

Shows the raw-vs-fit comparison for `X/Y/Z`.

### 6.4 `results/lifting/ball_3d_reprojection_comparison.csv/png`

Projects the final 3D trajectory back into 2D and compares it with the original `u/v`.

Used for:

- checking whether the lifting is self-consistent
- evaluating whether the current 3D baseline is at least approximately reasonable

### 6.5 `results/renders/lifted_scene*.mp4/png`

3D scene renders driven by `ball_3d_lifted_trajectory.csv`.

---

## 7. Current logic of `render_lifted_scene.py`

Script:

- [scripts/shared/render_lifted_scene.py](/mnt/hdd/AudioHOI/scripts/shared/render_lifted_scene.py)

This script now consumes only:

- `results/lifting/ball_3d_lifted_trajectory.csv`

It no longer depends on the earlier pseudo-3D UV tricks.

It currently supports two views:

### 7.1 `world` view

Renders `X/Y/Z` directly as a standalone 3D scene:

- floor plane
- full historical trajectory
- current ball position
- shadow point on the floor

### 7.2 `camera` view

Places the same `X/Y/Z` trajectory in a fixed synthetic camera and projects it to 2D:

- fixed camera height, pitch, and focal length
- a more source-video-like 3D scene view

This is:

```text
XYZ-based camera render
```

not the earlier pseudo-3D drawing based directly on image-space quantities.

---

## 8. Current end-to-end summary

The current basketball baseline can be compressed into these six steps.

### Step A

```text
video -> SAM2 masks
```

### Step B

```text
mask -> fitted circle -> (u, v, r)
```

### Step C

```text
r -> Z_raw
u + Z_raw -> X_raw
bottom_v + floor_v + Z_raw -> Y_raw
```

### Step D

```text
bounce/contact frames -> segment boundaries
```

### Step E

```text
X_raw, Y_raw, Z_raw -> fitted X, Y, Z
```

### Step F

```text
ball_3d_lifted_trajectory.csv -> 3D plots / lifted scene render
```

---

## 9. What this method is

This pipeline is a:

```text
monocular 3D lifting baseline
```

It is not:

- full 4D reconstruction
- multi-view geometric reconstruction
- learned end-to-end 3D recovery

Its role is:

```text
to lift a 2D mask trajectory into a minimal but explicit 3D ball trajectory baseline that can be analyzed, rendered, and improved later
```

---

## 10. Current limitations

1. `Z` depends strongly on the apparent radius `r`, so it is sensitive to mask error.
2. `Y` depends strongly on the ground/contact assumption.
3. The focal length `f` is currently approximate rather than calibrated.
4. The result is a baseline, not precise physical ground truth.

Still, this pipeline has already moved the project from:

```text
pseudo-3D visualization
```

to:

```text
explicit UVZ estimation + geometric 3D lifting + 3D scene rendering
```
