# Audio Loss v2 — from depth-anchors to an audio-gated contact-implicit schedule

Status: design (2026-07-08). Extends the KEPT audio-loss iterations logged in
`docs/loop_plan.md` §9 and memory `audio-loss-method`. Motivated by Tom's ask:
"reason about a better audio loss," and coupled to focus #2 (minimize mesh↔object
contact distance during contact without the body entering the object).

## 1. What the current audio loss actually does — and its ceiling

Today the audio signal enters the contact-phase solver as three things (all KEPT/verified):

1. **Sub-frame depth anchors** — at each detected event time `t_e` (fractional),
   pin object *depth* `Z(t_e)` to the contacting body part (surface-gap corrected so
   the center sits one radius off the joint).
2. **Inverse-confidence coupling** — anchor pull weight ×(1+κ(1−depth_conf)); trust
   audio more where the visual depth is unreliable (occlusion).
3. **Impulse budget** — the smoothness regularizer becomes a *hinge* with a per-frame
   ceiling `b_t = κ·A^{5/6}` (Hertz Δv∝A^{5/6}); `b=0` at silence, so silence forbids
   depth kinks ("silence as evidence").

**The structural limitation:** all three treat audio as a modifier on a *position/depth*
regularizer. The event's *timing* and its *energy ceiling* are used; the event's
**kinematic content** is not. Specifically:

- The impulse budget is **one-sided and permissive at the event**: it *allows* a kink of
  up to `b_t` but never *requires* one. A trajectory that glides straight through a loud
  impact with zero velocity change pays no audio penalty. Physically an impact *is* a
  velocity discontinuity — the loss should demand it.
- The event **type** (impact / bounce / catch / placement / scrape) is decoded by the
  classifier and used only to pick which body part to anchor to. It never shapes the
  *post-event motion*, even though each type implies a distinct velocity signature.
- Everything is a **point anchor at an instant**. Sustained contact (a mug set down and
  resting, a person sitting on a chair, a broom scraping) produces a *band* of audio
  energy over an interval, not an impulse — so it yields **zero promotable anchors**
  (the chair blocker in memory). Point anchors cannot express "in contact for 40 frames."

## 2. The reframing: audio as a contact-state schedule, not a set of anchors

The memory already names the correct framing — "audio as a mode oracle for
contact-implicit optimization (Posà, IJRR'14)." The current loss only implements the
*anchor* half of that idea. Audio Loss v2 completes it: decode audio into a **per-frame
contact-mode posterior** and let the mode select which physics residual is active.

```
audio  ──►  m(t) ∈ {FREE, IMPACT, SUSTAINED}  with soft confidences p_free, p_imp, p_sus
                     │
     ┌───────────────┼─────────────────────────────┐
     ▼               ▼                               ▼
  L_free           L_jump                          L_rest
 (ballistic)   (velocity-jump at event,        (zero relative normal velocity
               sign+magnitude from type/A)      over the interval + surface support)
```

Per frame the total audio-physics residual is the confidence-blended sum
`p_free·L_free + p_imp·L_jump + p_sus·L_rest`. This subsumes the current anchors as the
special case where `L_jump`/`L_rest` degenerate to a depth pin, but adds prescriptive
velocity structure and, crucially, an *interval* mode.

### 2.1 Decoding m(t) from audio

- **FREE**: audio energy at/below noise floor in the object band → object touches nothing.
- **IMPACT**: sharp broadband onset (existing detector), short decay. Confidence from onset
  sharpness × SNR. Already have this — it is the current "event."
- **SUSTAINED**: energy held above baseline across ≥K frames with *no* single sharp onset —
  friction noise (scrape/slide) or a low-frequency load transient (sit/place-and-rest).
  Detect via HPSS percussive-vs-harmonic + a boxcar energy envelope over the object band
  (the HPSS pass is already researched in `loop_plan §5.5`). The interval `[t0,t1]` is the
  above-baseline run. **This is the piece that unblocks the chair.**

## 3. The three residuals

### 3.1 L_free — ballistic (already exists, opt-in)

On FREE frames: penalize XZ acceleration, pull Y acceleration toward `g`. This is the
existing `--audio-flight-physics` / `--w-phys-*` term; v2 just gates it by `p_free(t)`
instead of a hard contact-state boolean, so a soft/uncertain classifier degrades
gracefully instead of switching physics on/off at a threshold.

### 3.2 L_jump — prescriptive velocity discontinuity (NEW)

At an IMPACT event, decompose object velocity into normal `v_n` and tangential `v_t`
w.r.t. the contact surface (support-plane normal for a bounce; body-part surface normal
for a catch). The event type prescribes the post-event normal velocity:

| type       | post-event normal velocity        | tangential |
|------------|-----------------------------------|------------|
| bounce     | `v_n⁺ = −e·v_n⁻` (restitution e)  | preserved  |
| catch/hold | `v_n⁺ ≈ 0`, object joins the part | → part's   |
| placement  | `v_n⁺ ≈ 0`, rests on support      | → 0        |

The residual has two coupled parts:
- **presence**: `L_jump = ‖v_n⁺ − f_type(v_n⁻, e)‖²` — *require* the reversal/arrest of the
  right sign and magnitude at `t_e`. This is what today's budget omits.
- **magnitude tie**: `|v_n⁻|` (impact speed) is tied to event energy via Hertz,
  `|Δv_n| = |v_n⁺−v_n⁻| ∝ A^{5/6}` — reusing the constant already fit for the budget.

Away from events the existing silence budget (`b_t≈0`) keeps `Δ²Z` small, so the kink is
forced to live *at* the event and nowhere else. Net effect: two-sided instead of a
permissive ceiling.

### 3.3 L_rest — sustained-contact interval (NEW, unblocks chair)

Over `[t0,t1]` (SUSTAINED): enforce **zero relative normal velocity** between object and
support, `‖(v_obj − v_support)·n̂‖² → 0` at every frame in the interval, plus a soft
surface-support term `|d_surface − band| → 0` (object surface rests at contact distance
on the support). Unlike a point anchor this is a *first-class interval constraint*: the
object is glued to the support for the whole span, which is exactly the semantics of
sitting/resting/scraping. Tangential motion is left free (a broom slides; a chair does not,
but that falls out of the support geometry, not the audio term).

## 4. Restitution `e` as a novel audio-derived physical parameter

For a bouncing object (basketball), consecutive bounce audio energies decay geometrically;
the ratio of successive impact energies estimates the coefficient of restitution
`e = (A_{k+1}/A_k)^{β}` (β from the Hertz energy↔velocity mapping). This gives a **measured,
per-material e** that:

- sets the magnitude in `L_jump` (§3.2) instead of a hand-tuned constant, and
- constrains the ballistic arc *between* two bounces to be consistent with the measured
  entry/exit speeds — a boundary-value constraint on the free segment, not just its ends.

No monocular-HOI reconstruction paper derives a restitution coefficient from the audio
track; this is a concrete novel-metric candidate for the paper's audio-physics section
(complements the existing audio-contact-timing-error and known-size-error metrics).

## 5. Coupling to focus #2 — mesh contact without body penetration

Focus #2 (minimize mesh↔object contact distance during contact, no body inside the object)
lives in the **body-side** Stage C solver (`refine_body_pose_contact.py`,
`--object-type mesh_sdf`), while the audio loss lives in the **object-side** contact-phase
solver. They currently use *different* contact truths (Stage C: VLM/geometry labels;
audio loss: audio events). v2 unifies them on **one audio-gated contact schedule** `m(t)`:

- On `p_imp`/`p_sus` frames Stage C activates its **two-sided mesh-SDF contact term**:
  - *non-penetration* hinge `Σ_pts max(0, −d(x) + ε)²` over the part point cloud
    (`d` = differentiable SDF, negative inside) — every hand/foot point must clear the
    surface (+flesh margin). Pushes the body out.
  - *touch* hinge on the **closest** point only, `max(0, d_min − band)²` — pull the nearest
    part point onto the surface. Minimizes the gap.
  Together these *minimize contact distance while forbidding penetration* — exactly the
  ask — and the SDF makes it surface-exact for the mug/chair/stick meshes instead of a
  center-distance proxy.
- On `p_free` frames only the non-penetration hinge is active — never fake a touch when
  audio says the object is airborne. This is the safeguard against the "pull-to-surface
  invents contact" failure mode (the existing `--pull-range-m` guard becomes audio-gated).

So the *same* `m(t)` that tells the object-side solver "reverse velocity here" tells the
body-side solver "close the gap to the mesh here (and only here)." One audio truth, two
solvers, consistent contact.

## 6. Implementation order (smallest verifiable steps)

1. **`m(t)` decoder** — add SUSTAINED detection (HPSS + boxcar envelope) to the audio
   stage; emit `contact_mode` + confidences per frame alongside the existing events CSV.
   Verify on basketball (all IMPACT), chair (one long SUSTAINED), mug (IMPACT at set-down).
2. **L_jump presence term** in the contact-phase solver, behind `--audio-jump` (default
   off first). Re-measure basketball/football timing error + jerk; expect timing to hold
   and the impact kinks to sharpen (co-report jerk, per the ablation rule).
3. **Restitution estimator** — `e` from bounce energy ratios on basketball; feed into
   L_jump magnitude; sanity-check `e≈0.75–0.85` for a basketball.
4. **L_rest interval term** — behind `--audio-sustained`; run the chair, confirm it yields
   a usable contact interval where point-anchors gave zero, and that Stage C penetration
   drops further than the differentiable-SDF-only 0.401→0.229.
5. **Unify Stage C on `m(t)`** — audio-gate the mesh-SDF touch hinge; re-run mug/chair/stick
   with real meshes; report contact_gap and penetration before/after.

Each step is one flag, default-off, verified with render frames + metrics before it flips
on — same discipline as §9. Steps 1–2 are the paper's core "prescriptive audio" claim;
4 is the generality claim (sustained contact, not just impacts).
