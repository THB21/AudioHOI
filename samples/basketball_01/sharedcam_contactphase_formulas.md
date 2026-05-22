# Sharedcam and Contactphase Mathematical Formulation

This note summarizes the current optimization objectives used in the basketball shared-camera pipeline.

## 1. Notation

For frame $t$:

- Ball translation in camera coordinates:
  $$T_t = (X_t, Y_t, Z_t)$$
- Projected ball center:
  $$(u_t, v_t)$$
- Projected ball radius in pixels:
  $$r_t$$
- Observed center / radius from tracking:
  $$(u_t^{obs}, v_t^{obs}, r_t^{obs})$$
- Shared floor line in image space:
  $$floor_v$$
- Projected ball-bottom image coordinate:
  $$bottom_v(t)$$

The basketball radius is fixed:

$$
R = 0.12 \text{ m}
$$

The projection model is:

$$
u_t = f_x \frac{X_t}{Z_t} + c_x
$$

$$
v_t = f_y \frac{Y_t}{Z_t} + c_y
$$

$$
r_t = f_x \frac{R}{Z_t}
$$

$$
bottom_v(t) = v_t + r_t
$$

---

## 2. Sharedcam baseline

Code:
- `scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py`

The current shared-camera baseline optimizes:

$$
E_{sharedcam}
= \lambda_{mask} E_{mask}
+ \lambda_{center} E_{center}
+ \lambda_{size} E_{size}
+ \lambda_{contact} E_{contact}
+ \lambda_{temp} E_{temp}^{xyz}
+ \lambda_{ztemp} E_{temp}^{z}
+ \lambda_{zb} E_{z-boundary}
+ \lambda_{zs} E_{z-slope}
+ E_{aux}
$$

### 2.1 Mask / contour term

Projected circle contour should stay close to the observed mask contour:

$$
E_{mask} = \sum_t d_{chamfer}(C_t^{proj}, C_t^{obs})
$$

Current weight:

$$
\lambda_{mask} = 0.018
$$

### 2.2 Center reprojection term

$$
E_{center} = \sum_t \left[(u_t-u_t^{obs})^2 + (v_t-v_t^{obs})^2\right]
$$

Current weight:

$$
\lambda_{center} = 0.04
$$

### 2.3 Size term

The projected diameter / area should match the observed mask extent:

$$
E_{size} = \sum_t \Big[(d_t-w_t)^2 + (d_t-h_t)^2 + (d_t-s_t)^2 + \alpha (a_t-a_t^{obs})^2\Big]
$$

where $d_t = 2r_t$.

Current weight:

$$
\lambda_{size} = 0.02
$$

### 2.4 Contact / floor term

At contact frames, the projected ball bottom should stay close to the shared floor line:

$$
E_{contact} = \sum_{t \in C} (bottom_v(t)-floor_v)^2
$$

Current weight:

$$
\lambda_{contact} = 10.0
$$

### 2.5 Temporal smoothness in 3D

Using second-order finite differences on translation:

$$
E_{temp}^{xyz} = \sum_t \|T_{t+1} - 2T_t + T_{t-1}\|^2
$$

Current weight:

$$
\lambda_{temp} = 0.08
$$

### 2.6 Extra temporal smoothness on depth

$$
E_{temp}^{z} = \sum_t (Z_{t+1} - 2Z_t + Z_{t-1})^2
$$

Current weight:

$$
\lambda_{ztemp} = 0.22
$$

### 2.7 Segment boundary continuity

Depth continuity between neighboring fitted segments:

$$
E_{z-boundary} = \sum_k (Z_{start(k+1)} - Z_{end(k)})^2
$$

Current weight:

$$
\lambda_{zb} = 3.5
$$

### 2.8 Segment slope continuity

$$
E_{z-slope} = \sum_k (a_{k+1} - a_k)^2
$$

Current weight:

$$
\lambda_{zs} = 0.35
$$

### 2.9 Auxiliary terms

The current implementation also includes several small regularizers:

- minimum depth penalty
- segment slope regularization
- endpoint stabilization

with hard-coded coefficients:

$$
0.30, \; 0.12, \; 0.15
$$

---

## 3. Contactphase refinement

Code:
- `scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration.py`

This stage does not fully re-optimize the ball trajectory. Instead, it keeps the observed 2D center fixed and locally refines the depth trajectory around contact frames.

The optimized variable is mainly:

$$
Z_t
$$

and then $(X_t, Y_t)$ are reconstructed from $(u_t^{obs}, v_t^{obs}, Z_t)$.

The objective is:

$$
E_{contactphase}
= w_{size} E_{size}
+ w_{fw} E_{floor-window}
+ w_{fc} E_{floor-contact}
+ w_{anchor} E_{anchor}
+ w_{temp} E_{temp}

\text{subject to } Z_t = Z_{hand}(t),\; t \in C
$$

### 3.1 Size term

Keep projected radius consistent with the observed radius:

$$
E_{size} = \sum_t (r_t-r_t^{obs})^2
$$

Current weight:

$$
w_{size} = 0.12
$$

### 3.2 Floor-window term

Within a small window around contact frames:

$$
E_{floor-window} = \sum_t w_t^{window} (bottom_v(t)-floor_v(t))^2
$$

Current weight:

$$
w_{fw} = 0.45
$$

Current window parameters:

$$
radius = 3 \text{ frames}, \quad \sigma = 1.35
$$

### 3.3 Exact-contact floor term

At exact contact frames:

$$
E_{floor-contact} = \sum_{t \in C} (bottom_v(t)-floor_v(t))^2
$$

Current weight:

$$
w_{fc} = 1.35
$$

### 3.4 Hard hand-Z constraint at contact frames

We first construct a coarse palm proxy from SMPL-X joints:

$$
P_t^{palm} = \frac{1}{5}\left(P_t^{wrist} + P_t^{index1} + P_t^{middle1} + P_t^{ring1} + P_t^{pinky1}\right)
$$

Then for each frame we choose the active hand (left or right) as the one closer to the current ball estimate.

At exact contact frames, the ball depth is enforced directly to equal the active hand depth:

$$
Z_t = Z_{hand}(t), \quad t \in C
$$

This is now treated as a hard constraint rather than a soft penalty term. In the implementation, the optimized depth vector is overwritten at contact frames before evaluating the rest of the objective.

So this stage no longer uses a weighted hand-contact loss; instead, it fixes contact-frame depth and only optimizes the remaining frames around it.

### 3.5 Anchor term

Keep the refined depth close to the global reference trajectory outside contact frames:

$$
E_{anchor} = \sum_{t \notin C} (Z_t - Z_t^{ref})^2
$$

Current weight:

$$
w_{anchor} = 0.70
$$

### 3.6 Temporal term

Second-order smoothness on the refined depth trajectory:

$$
E_{temp} = \sum_t (Z_{t+1} - 2Z_t + Z_{t-1})^2
$$

Current weight:

$$
w_{temp} = 1.35
$$

---

## 4. High-level interpretation

### Sharedcam

The shared-camera baseline puts the ball into the same camera geometry as the GVHMR human result and optimizes a stable ball trajectory using reprojection, size, floor contact, and temporal smoothness.

### Contactphase

The contactphase stage now enforces hand depth as a hard rule at contact frames, then refines the remaining trajectory around that fixed contact depth while still preserving floor consistency and temporal smoothness.
