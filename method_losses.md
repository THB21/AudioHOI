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
body part (auto-selected, e.g. hand/foot). Non-anchor (free) frames are solved for, and
$X_t,Y_t$ follow from back-projection at the current $Z_t$.

Residuals (robust `soft_l1`):

- Stay near the prior: $w_\text{ref}\,(Z_t - Z_t^\text{ref})$ on free frames
- Smoothness regularizer: $w_\text{temp}\,(Z_{t+1}-2Z_t+Z_{t-1})$ on non-anchor interiors

**Audio terms (implemented).** Each onset in `results/events/audio_events.csv` carries a
confidence $s_t\in[0,1]$ (the `audio_score` column, or a `prominence`-normalized fallback),
spread over $\pm2$ frames with a triangular falloff into a per-frame $\text{aud}_t$. It enters
the depth solve in three ways:

- **New contact anchors (timing):** a strong onset frame ($s_t\ge\tau$, default $\tau=0.5$) is
  promoted to a hard anchor pinned to the contacting part's depth, even where visual contact
  detection missed it — audio gives the exact moment the object touches.
- **Soft audio contact pull:** $w_\text{audio}\,\text{aud}_t\,(Z_t - Z_t^\text{part})$ on free
  frames near an onset, nudging depth to the contacting part with confidence $\propto s_t$.
- **Audio-gated acceleration relaxation:** the smoothness term is locally scaled by
  $(1-\gamma\,\text{aud}_t)$ so a real bounce/placement velocity kink at the impact instant is
  not smoothed away.

Defaults: $w_\text{ref}=0.7,\ w_\text{temp}=5.0,\ w_\text{audio}=3.0,\ \gamma=0.8$. With no
`audio_events.csv` present, $\text{aud}_t\equiv0$ and all three terms vanish — the solve is
identical to the contacts-only version (verified byte-identical on basketball).

**Audio semantics (implemented, `scripts/shared/events/audio_semantics.py`).** Rather than
one global $\gamma$ and a blanket promotion rule, each onset is classified **zero-shot** from
generic acoustic features (attack sharpness, decay time, spectral brightness, recomputed from
`audio.wav` with `scipy` only) into `impact / bounce / placement / scrape / sustained`, and each
type sets its own physics: the relaxation $\gamma_t$ (full for an impact, almost none for a
sustained hold), the pull weight, and crucially a **contact target** (`part` vs `support`). Only
body-part contacts are promoted to anchors, so a periodic floor **bounce** (a dribble) is no
longer falsely pinned to the hand — it gets the kink relaxation but no body-part anchor. (On
basketball this drops 15 spurious hand anchors to 0; on a football kick the impacts promote.)

There is deliberately **no gravity / ballistic prior** here. That kind of physics
assumption is object- and scenario-specific (a thrown ball, but not a swung hammer or a
sliding drawer), so it's been removed. The only prior is the acceleration-smoothness
regularizer of Section 6; everything else comes from the data (contacts, observations, audio).

## 5. Generalized zero-shot interaction energy (target formulation)

Sections 2–4 are the current object-side baseline. The direction we're consolidating
toward is a single, object-agnostic energy that works **zero-shot on anything** — no
per-object or per-category training, no baked-in physics. Every cue comes from a
foundation model that is itself zero-shot, and every term is geometric/observational.
The human (GVHMR + HaMeR) is fixed and metric; we optimise the object pose
$\mathbf{T}_t=(\mathbf{R}_t,\mathbf{t}_t)$ against it.

$$\mathbf{T}_{1:N}^\star=\arg\min \sum_t\Big[ w_\text{mask}R_\text{mask} + w_\text{kp}R_\text{kp} + w_\text{depth}R_\text{depth} + w_\text{center}R_\text{center} + w_\text{contact}R_\text{contact} + w_\text{support}R_\text{support} \Big] \;+\; w_\text{reg}R_\text{reg}$$

Data terms (each from a zero-shot source):

- $R_\text{mask}$ — object geometry silhouette vs. SAM2/Grounding-DINO mask (chamfer/IoU). Replaces the sphere size/radius cue.
- $R_\text{kp}$ — CoTracker 2D tracks vs. reprojected surface points (also constrains rotation).
- $R_\text{depth}$ — object metric depth $Z_t-\hat Z_t$ from DA3 (Section 2).
- $R_\text{center}$ — projected centroid vs. observed centroid/bbox.
- $R_\text{contact}$ — at contact frames, object surface meets the human contact part or support plane; **gated and time-stamped by the audio–visual events** (audio onsets give the exact impact frame, video gives the candidate, alignment activates the term). The contact *point* itself comes from `scripts/shared/events/extract_contact_points.py`: at each audio-confirmed onset it places a surface point on the object (geometric: along the centre→nearest-part ray, exact for a sphere; or VLM: a Qwen3-VL crop query for the precise contact pixel + object part). Reprojecting that known surface point to the observed contact pixel is the residual that constrains object translation and — for non-spherical objects — rotation.
- $R_\text{support}$ — object rests on / rebounds off the estimated ground plane.

**The only prior is one smoothness regularizer** (Section 6) — no gravity, no
constant-velocity, no parabola. The motion is carried by the data; the regularizer just
stops it jumping.

## 6. The smoothness regularizer

A light penalty on **acceleration / jerk** (the *change* of motion), not on speed — so
genuinely fast-but-smooth motion (a kick, a throw) is allowed, while frame-to-frame
jitter and teleport-like jumps are suppressed:

$$R_\text{reg}=\sum_t \big\| \mathbf{t}_{t+1}-2\mathbf{t}_t+\mathbf{t}_{t-1} \big\|^2 \quad(\text{and analogously for } \mathbf{R}_t).$$

Keep $w_\text{reg}$ small and the loss robust (`soft_l1`) so sharp but real velocity
changes at contacts survive; relax it locally at audio/visual contact frames, where a kink
is physically expected. This local audio relaxation is implemented in the depth-only
contact-phase solver (Section 4, the $\gamma$ term).

## 7. VLM agentic check & real-time scene adjustment

On top of the optimisation sits an agentic feedback layer. A vision-language model
inspects the generated scene (rendered overlay + world view) against the input video and
the textual/event description, and judges plausibility that the energy alone can't —
interpenetration, wrong/missing contact, implausible placement or pose, object on the
wrong side, etc.

Its findings drive a **real-time adjustment of the scene generation**: re-weight or
toggle terms (e.g. strengthen $R_\text{contact}$ or $R_\text{depth}$), re-time/re-assign
contacts, re-pick the contact part, or re-run the relevant stage — then re-render and
re-check, in a loop until the VLM signs off. This keeps the system zero-shot (the VLM is
a general critic, not a trained-per-object module) while catching the failures a
geometric loss is blind to.

## 8. Rendering transforms (not a loss)

`scripts/shared/human_ball/render_full_scene_3d.py`

- Overlay path: camera-frame meshes via $\mathrm{diag}(1,1,-1)$ (OpenCV → OpenGL), image flipped vertically.
- World path: OpenCV → y-up world via $\mathrm{diag}(1,-1,1)$. Both are reflections ($\det=-1$) and invert triangle winding, so faces are reversed (`faces[:, ::-1]`) to keep outward normals.
- The mesh is composited over the input video at opacity $\alpha$ for the comparison overlay; the world view is opaque with a single directional shadow.
