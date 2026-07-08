# Why DA3 depth fails — root cause (instrumented)

Method: `scripts/shared/depth/diagnose_da3_fit.py` re-runs DA3, samples depth at the projected
SMPL-X body joints, and measures the conditioning of the per-frame affine `Z = a·D + b`.

## Measured per video (stride-24 frames)

| metric | basketball (works) | football (fails) | mug (fails) |
|---|---|---|---|
| joint TRUE depth spread (m) | 0.46 | 0.57 | **0.29** |
| corr(DA3, trueZ) on body | **+0.55** | +0.48 | **−0.41** |
| affine slope a | +0.52 | +0.23 | **−0.13** |
| DA3-at-joints spread (stability) | mostly ~0.7 | **spikes to 10–22** | ~1.07 |
| object DA3 in joint range | 7/8 | **7/11 (4 extrapolate, up to 1.8×)** | 4/8 |
| resulting object Z (m) | 3.75–4.12 ✓ | **6.5–11.7 ✗** | 2.01–2.08 (flat ✗) |

## Root causes

1. **Tiny metric baseline (all videos).** The body joints span only ~0.3–0.6 m in *true* depth —
   a person is nearly fronto-parallel. Fitting scale+offset on a 0.3 m baseline is inherently
   ill-conditioned: small DA3 noise produces large slope error. Basketball survives only because
   its DA3 ordering on the body is decent (corr +0.55).

2. **DA3 not monotonic on the body → mug.** On the mug clip DA3 depth is *anti-correlated* with
   true depth on the body (corr **−0.41**), so the fitted slope is **negative** (−0.13). Depth
   then moves the object the wrong way; since the mug barely moves in depth (Z range 0.07 m) the
   absolute error is small but the *shape* is wrong — depth is anti-informative. This is why
   `R_depth` disagreed with the solved `tz`.

3. **Background-contaminated joint samples → football.** On several frames a projected joint
   lands on the far grass/horizon, injecting a huge DA3 value (DA3-at-joints spread spikes to
   **10–22** vs ~0.8 normally). That outlier flattens the slope to ~0.015, and the robust MAD
   rejection doesn't remove it. Outdoor distant scenes are also DA3's weak regime.

4. **Object extrapolation → football.** The ball's DA3 value lands *outside* the body-joint DA3
   range on 4/11 frames (up to 1.8× the joint span beyond it). The affine then extrapolates,
   amplifying error → object Z jumps to 7.3–11.7 m.

## Implications / fixes (no code yet)

- **Widen the baseline**: add depth-diverse anchors beyond the torso — both feet vs head, the
  estimated ground plane, or any static scene points — so the fit isn't on a 0.3 m slab.
- **Reject background joints**: sample a small patch around each joint and drop joints whose DA3
  is far from the body median (kills the football horizon spikes).
- **Forbid extrapolation**: clamp the object DA3 value to the joint range, or **gate `R_depth`**
  by `corr` and extrapolation distance — drop/down-weight depth when corr<~0.3 or the object is
  out of range. The energy decomposition already shows depth is the loose term; this makes the
  down-weighting principled and automatic.
- **mug**: with corr −0.41, depth should simply be dropped for this clip; rely on 2D + the
  object's own geometry.
