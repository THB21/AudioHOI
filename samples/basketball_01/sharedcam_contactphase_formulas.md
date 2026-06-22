# Sharedcam and Contactphase Mathematical Formulation

This note summarizes the optimization objectives for the basketball shared-camera pipeline (sphere baseline).

Pipeline update:
SAM2 masks + CoTracker -> generic object observations -> shared-camera baseline -> contactphase anchor interpolation.

> **Newer than this note:** depth can now come from Depth Anything 3 (`--depth-source depthv3`)
> instead of the size term §2.3, and the contact phase lives in
> `run_human_ball_contact_phase_calibration_anchorinterp_generic.py` with **no gravity/ballistic
> term** (anchors + reference + smoothness only). The generalized, object-agnostic energy is in
> `method_losses.md`.

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
- Generic object observation vector (from SAM2 masks + CoTracker):
  $$\mathbf{o}_t = [u_t, v_t, r_t^{obs}, a_t, w_t, h_t, \rho_t, c_t]$$
  where $a_t$ is mask area, $w_t, h_t$ are the mask/bbox width and height, $\rho_t$ is the aspect ratio, and $c_t$ is a contour compactness/circularity cue. For basketball, we mainly use $u_t, v_t, r_t^{obs}$. For mug/hammer, other shape cues can be emphasized.
- Shared support geometry in image space:
  $$S = (support\_type, floor_v, source, confidence)$$
- Shared floor/support line in image space:
  $$floor_v$$
- Projected ball-bottom image coordinate:
  $$bottom_v(t)$$

For the current basketball case, the default radius is:

$$
R = 0.12 \text{ m}
$$

The current `sharedcam` code also exposes an optional backend for per-frame radius estimates, but the mainline branch documented here still uses the default fixed radius unless those estimates are explicitly supplied. This should be read as a basketball-specific instantiation, not as the universal object model for the whole framework.

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
- `scripts/shared/sharedcam/support_geometry.py`

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

### 2.0 Shared support geometry and floor-aware initialization

Before optimization, `sharedcam` first estimates a scene-level support definition

$$
S = (support\_type, floor_v, source, confidence)
$$

from ball-contact event frames in image space. In the current implementation:

- `support_type = floor`
- `source = ball_contact_events`
- `floor_v = \operatorname{median}` of observed ball-bottom values on contact frames

This support geometry is written to:

```text
results/pose6d_sharedcam/support_geometry.json
```

The initialization is then **floor-aware**, but the framework-level view should be written more generally as

$$
T_t^{init} \leftarrow \operatorname{InitFromObservation}(\mathbf{o}_t, K, \Pi, S),
$$

where:

- $\mathbf{o}_t$ is the generic object observation vector,
- $K$ is the camera intrinsic matrix,
- $\Pi$ is the object prior package (geometry / scale assumptions),
- $S$ is the shared support geometry.

For the **current basketball instantiation**, the object prior is sphere-like with known radius $R$, so the implementation uses:

$$
Z_t^{init} = \frac{f_x R}{r_t^{obs}}
$$

and

$$
r_t^{init} = \frac{f_x R}{Z_t^{init}}.
$$

Instead of initializing the vertical coordinate directly from the observed image center, the current code uses the shared support line:

$$
v_t^{center,init} = floor_v - r_t^{init}
$$

$$
X_t^{init} = \frac{(u_t^{obs}-c_x)Z_t^{init}}{f_x}
$$

$$
Y_t^{init} = \frac{(v_t^{center,init}-c_y)Z_t^{init}}{f_y}
$$

So the **current basketball-specific** `sharedcam` logic is:

```text
radius -> depth z
u + z -> x
floor + z + radius -> y
```

For non-spherical objects, the initialization should still be viewed as observation-driven, but with a different prior package $\Pi$ rather than a fixed sphere radius.

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

The current implementation also adds a weaker version of the same residual on audio-aligned support frames:

$$
E_{contact}^{weak} = \sum_{t \in C_{weak}} \frac{1}{2}(bottom_v(t)-floor_v)^2
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

#### 2.9.1 Minimum depth penalty

A soft lower bound is applied when the depth becomes too small:

$$
E_{z-min} = \sum_t \max(0, 0.35 - Z_t)
$$

with coefficient:

$$
0.30
$$

#### 2.9.2 Segment slope regularization

If each fitted depth segment is written as

$$
Z_t = a_k \tau_t + b_k, \qquad t \in segment\ k,
$$

then the implementation also penalizes the raw slope magnitude:

$$
E_{seg-slope-mag} = \sum_k a_k
$$

with coefficient:

$$
0.12
$$

#### 2.9.3 Endpoint stabilization

The beginning and end of the sequence are weakly stabilized by penalizing the first and last inter-frame jumps:

$$
E_{endpoints} = \|T_2 - T_1\|^2 + \|T_T - T_{T-1}\|^2
$$

with coefficient:

$$
0.15
$$

---

## 3. Contactphase anchor interpolation

Code:
- `scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp.py`

This is the current main refinement branch.

The stage keeps the observed 2D ball center fixed and optimizes only the depth trajectory:

$$
Z_1, Z_2, \dots, Z_T
$$

Then $(X_t, Y_t)$ are reconstructed from $(u_t^{obs}, v_t^{obs}, Z_t)$:

$$
X_t = \frac{(u_t^{obs}-c_x)Z_t}{f_x},
\qquad
Y_t = \frac{(v_t^{obs}-c_y)Z_t}{f_y}
$$

This stage **reads** the shared support definition from:

```text
results/pose6d_sharedcam/support_geometry.json
```

and copies `floor_v` / support metadata into its outputs, but it does **not** re-estimate the floor inside the `handball` module.

### 3.1 Right-palm contact anchor

For the current basketball sequence, the active hand is fixed to the right hand.

We define a coarse right-palm proxy from SMPL-X joints:

$$
P_t^{right\text{-}palm}
= \frac{1}{5}\left(
P_t^{21}
+ P_t^{40}
+ P_t^{43}
+ P_t^{46}
+ P_t^{49}
\right)
$$

and use its depth as:

$$
Z_{hand}(t) = P_{t,z}^{right\text{-}palm}
$$

Let $C$ be the set of hand-contact event frames. At those frames, depth is treated as an exact anchor:

$$
Z_t = Z_{hand}(t), \qquad t \in C
$$

### 3.2 Global depth-layer correction

The sharedcam baseline depth is denoted by $Z_t^{base}$.

We first estimate a single robust global depth bias from hand-contact events:

$$
\Delta Z = \operatorname{median}_{t \in C}\left(Z_{hand}(t) - Z_t^{base}\right)
$$

Then we form the globally shifted reference trajectory:

$$
Z_t^{ref} = Z_t^{base} + \Delta Z
$$

### 3.3 Anchor interpolation objective

Between hand-event anchors, the trajectory shape is not driven by local hand windows. Instead, it is determined by:

1. a weak prior toward the globally shifted reference, and
2. a strong second-difference smoothness prior.

The objective is:

$$
E_{anchorinterp} = w_{ref} E_{ref} + w_{temp} E_{temp}
$$

subject to

$$
Z_t = Z_{hand}(t), \qquad t \in C
$$

where

$$
E_{ref} = \sum_t (Z_t - Z_t^{ref})^2
$$

and

$$
E_{temp} = \sum_{t=2}^{T-1} (Z_{t+1} - 2Z_t + Z_{t-1})^2
$$

Current weights:

$$
w_{ref} = 0.70,
\qquad
w_{temp} = 5.00
$$

### 3.4 Interpretation

This formulation explicitly separates:

- **anchor values**: hand-contact event depths $Z_{hand}(t)$
- **trajectory shape**: smooth interpolation between anchors

So hand events only determine exact contact-frame depth values; they do not drag neighboring frames through symmetric local windows.

---

## 4. High-level interpretation

### Sharedcam

The shared-camera baseline puts the ball into the same camera geometry as the GVHMR human result, defines a scene-level support geometry, initializes the ball height from that support, and then optimizes a stable ball trajectory using reprojection, size, floor contact, and temporal smoothness.

### Contactphase anchor interpolation

The current contactphase stage reads the support geometry produced by `sharedcam`, applies a global human-relative depth-layer correction, pins hand-contact event frames to the right-palm depth, and finally solves the depths between anchors by smooth interpolation.
