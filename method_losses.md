# Method & Loss Functions

A mathematical overview of the optimisation problems in the AudioHOI pipeline:
how the object is lifted to 3D, how Depth Anything 3 depth is scaled, and how the
contact phase refines it. Everything lives in the shared GVHMR full-image camera
frame so the human, hands and object are metrically consistent.

## 1. Camera and notation

Pinhole camera with per-frame intrinsics $K_t = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ (from GVHMR's `K_fullimg`).

A 3D point $\mathbf{P}=(X,Y,Z)$ in the camera frame projects to pixel

$$u = f_x \frac{X}{Z} + c_x, \qquad v = f_y \frac{Y}{Z} + c_y .$$

Back-projection of an observed pixel $(u,v)$ at a known depth $Z$:

$$X = \frac{(u-c_x)\,Z}{f_x}, \qquad Y = \frac{(v-c_y)\,Z}{f_y}.$$

OpenCV convention: $x$ right, $y$ down, $z$ forward.

## 2. Object depth — Depth Anything 3 scaled to GVHMR

`scripts/shared/depth/run_depth_anything_v3.py`

DA3 returns a per-frame depth map $D_t(u,v)$ that is only correct up to an unknown
per-frame scale and offset. We tie it to true metric scale using the GVHMR body,
which is already metric in the same camera.

Let $\mathbf{J}_{t,k}$ be SMPL-X body joint $k\in\{0,\dots,21\}$ at frame $t$ (camera
frame), $Z_{t,k}$ its depth, and $\mathbf{p}_{t,k}$ its projection. Sample DA3 there,
$d_{t,k} = D_t(\mathbf{p}_{t,k})$, and fit a **per-frame** affine

$$(a_t, b_t) = \arg\min_{a,b} \sum_{k \in \mathcal{I}_t} \big( a\, d_{t,k} + b - Z_{t,k} \big)^2 ,$$

where $\mathcal{I}_t$ is the inlier set from iterative MAD rejection (drop points with
residual $> 3\cdot 1.4826\cdot \mathrm{MAD}$, refit). Per-frame (not global) because
DA3's scale drifts frame to frame; a single global $(a,b)$ flattens the motion.

The curves $a_t,b_t$ are linearly interpolated over frames where the fit failed and
smoothed in time (Gaussian, $\sigma=2$ frames). The object depth is then

$$\hat Z_t = a_t \cdot \operatorname{median}_{\,p \in \text{mask}_t} D_t(p) + b_t ,$$

using the segmentation mask if present, otherwise a small disk around the tracked
centre. (The median commutes with the affine, so we scale the scalar median rather
than the whole map.)

## 3. Object 3D lifting — shared-camera baseline

`scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py`

Estimate the object translation $\mathbf{t}_t=(X_t,Y_t,Z_t)$. The optimisation
variables are per-frame $X_t,Y_t$ plus a piecewise-linear depth: for segment $s$
(split at floor-contact frames) $Z_t = a_s \tau + b_s$ with $\tau$ the time since the
segment start. The sphere model projects a radius $R$ as $r_\text{px}=f_x R/Z$.

The state is fit with `scipy.least_squares` under a robust `soft_l1` loss
$\rho(e)=2(\sqrt{1+e^2}-1)$ applied to the stacked, weighted residuals below.

**Always on (shape independent):**

- Centre reprojection: $w_c\,(u_\text{pred}-u_\text{obs})$, $w_c\,(v_\text{pred}-v_\text{obs})$
- Floor contact at contact frames: $w_\text{contact}\,(v^\text{bottom}_\text{pred}-v_\text{floor})/20$ (half weight on the nearby support window)
- Ground penalty: $0.30\,\max(0,\,0.35 - Z_t)$ (keep the ball in front of the camera)
- Temporal smoothness of $X,Y$: $w_t\,(\mathbf{p}_{t+1}-2\mathbf{p}_t+\mathbf{p}_{t-1})$, $\;\mathbf{p}=(X,Y)$
- Depth smoothness within a segment: $w_{zt}\,(Z_{t+1}-2Z_t+Z_{t-1})$
- Segment continuity at the join: $w_{zb}\,(Z_\text{next}-Z_\text{end})$ and slope match $w_{zs}\,(a_{s+1}-a_s)$
- Small endpoint velocities

**`--depth-source sphere` (legacy, needs a known radius):**

- Mask chamfer: $w_m \cdot \mathrm{chamfer}\big(\text{circle}(u,v,r_\text{px}),\,\text{contour}_\text{obs}\big)$
- Size: $w_s\,(2 r_\text{px}-\text{bbox}_w)$, similarly for height/mean, and an area term

This is where depth comes from in the old baseline: the size residuals pin $r_\text{px}$,
and $Z=f_x R/r_\text{px}$.

**`--depth-source depthv3` (object-agnostic):** the mask/size residuals are switched
off and replaced by the DA3 metric-depth term

$$w_d\,\big(Z_t - \hat Z_t\big),$$

so depth comes from Section 2 and nothing assumes a sphere.

Full objective: $\;\displaystyle \min_{\{X_t,Y_t\},\{a_s,b_s\}} \sum_i \rho\!\big(r_i^2\big)$ over all residuals $r_i$ above.

## 4. Contact-phase refinement

`scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp_generic.py`

A depth-only pass that fixes the 2D observation $(u_\text{obs},v_\text{obs})$ and refines
$Z_t$. Contact frames are **anchors**: $Z_t$ is pinned to the depth of the contacting
body part (hand/foot). Non-anchor (free) frames are solved for. $X_t,Y_t$ follow from
back-projection at the current $Z_t$.

Residuals (robust `soft_l1`):

- Stay near the prior: $w_\text{ref}\,(Z_t - Z_t^\text{ref})$ on free frames
- Temporal smoothness: $w_\text{temp}\,(Z_{t+1}-2Z_t+Z_{t-1})$ on non-anchor interiors
- Ballistic prior during free-flight triplets (constant horizontal velocity, gravity on the vertical):
  $$w_{xz}\,(X_{t+1}-2X_t+X_{t-1}),\quad w_{xz}\,(Z_{t+1}-2Z_t+Z_{t-1}),\quad w_{y}\,\big((Y_{t+1}-2Y_t+Y_{t-1}) - g\,\Delta t^2\big),$$
  where the second difference of position approximates acceleration and $g\,\Delta t^2$ is the expected drop under gravity.

Defaults: $w_\text{ref}=0.7,\ w_\text{temp}=5.0,\ w_{xz}=1.25,\ w_{y}=1.5,\ g=9.81\,\mathrm{m/s^2}$.

## 5. Rendering transforms (not a loss)

`scripts/shared/human_ball/render_full_scene_3d.py`

- Overlay path: camera-frame meshes via $\mathrm{diag}(1,1,-1)$ (OpenCV → OpenGL), image flipped vertically.
- World path: OpenCV → y-up world via $\mathrm{diag}(1,-1,1)$. Both are reflections ($\det=-1$) and invert triangle winding, so faces are reversed (`faces[:, ::-1]`) to keep outward normals.
- The mesh is composited over the input video at opacity $\alpha$ for the comparison overlay; the world view is opaque with a single directional shadow.
