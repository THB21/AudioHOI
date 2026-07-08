# AudioHOI — Pipeline Overview, Loss Formulation & Improvement Log

> **ACTIVE (2026-07-08, overnight autonomous run):** Yixin's ask = complete **human modeling
> + object interaction results for the five cases** (basketball, football, mug, chair, stick
> in `samples_known_object/`) + HOI-level evaluation design. Campaign plan in **§10**;
> per-iteration verification continues in §9. Her new work (generic SE3 mainline, stick
> line-object, 3-layer final evaluator) synced from `origin/vlm-gated-mainline` into this
> tree (uncommitted).
>
> Superseded-but-still-open items from 07-06 restart: verify grasp-rotation inheritance;
> bake real mug/chair URDFs → GLB; PHOSA baseline; commits (~145+ files).

Working document for the paper push (3D-vision venue). Contribution: **audio as a
first-class constraint in monocular 4D HOI reconstruction** — perfectly combined with
visual evidence inside one contact-centric optimization, with VLM reasoning selecting
*what* is in contact and an LLM correction pass auditing the result.

Maintained iteratively: every improvement step is appended to §9 with visual + metric
verification before it is kept.

---

## 1. Information sources → what each contributes

| Source | Extractor | Information | Enters optimization as |
|---|---|---|---|
| **Audio track** | `src/audio` detect→features→classify | contact **timing** (ms-precise), event **type** (impact/placement/scrape/sustained), **intensity** (onset energy), rhythm of repeated contacts | hard anchors (promoted onsets), soft depth pull `E_audio`, local relaxation of smoothness (γ), event-type-specific weights |
| **Video: object 2D** | SAM2 masks (centroid/bbox/contour). CoTracker dropped — masks already give the per-frame observation; sparse points added nothing beyond them | per-frame 2D center (u,v), apparent radius, mask | fixed ray constraint (u,v back-projection), `E_center`, `E_mask`, `E_size` (sphere branch) |
| **Video: metric depth** | Depth Anything 3, per-frame affine-aligned to GVHMR body | object depth prior + per-frame confidence | `E_depth` (confidence-weighted), init for Z |
| **Video: human** | GVHMR (SMPL-X, world-grounded) + HaMeR (fingers) | body/hand joints per frame in camera space | contact anchor **values** (part depth), penetration geometry, floor estimate |
| **Contact candidates** | proximity probes (foot/hand ↔ object, object ↔ floor) | candidate contact frames/intervals + side | anchor **frames**, contact states gating flight/physics |
| **VLM reasoning** (Qwen-VL on event windows) | `src/audio/vlm_pipeline` | **which body part** contacts, event relevance, interaction verb, impact strength | overrides fuse's nearest-part attribution (geometry-gated) → anchor **part selection**; relevance gate kills spurious onsets |
| **LLM prior/correction** | (Stage −1 prior + Stage 6.5 audit on teammate branch; ours: §7) | symbolic expectations (which parts CAN touch, support type); post-hoc audit labels | discrete only: gates/flags, re-run recipes — never raw pose |

Flow: sources → per-frame evidence tables (CSV contracts) → **Stage A** object lifting →
**Stage B** contact-phase depth refinement (audio lives here) → **Stage C** body-side
contact refinement → renders → **LLM correction audit** (§7) → optional re-run with
corrected flags.

## 2. Current optimization, mathematically

Notation: object center \(T_t=(X_t,Y_t,Z_t)\), observed pixel \((u_t,v_t)\), intrinsics
\(K_t\), body-part depth \(p_t\), anchor set \(\mathcal A\), audio support \(a_t\in[0,1]\),
per-event relaxation \(\gamma_t\), radius \(r\) (proxy), part point cloud \(\mathcal P^b_t\)
(fingertips/toes included).

### Stage A — lifting (`run_basketball_pose6d_sharedcam.py --depth-source depthv3`)
\[
E_A=\lambda_c E_{center}+\lambda_d\sum_t c_t\,(Z_t-Z^{DA3}_t)^2
+\lambda_s\sum_t\|\ddot T_t\|^2+\lambda_{seg}E_{lin\text-Z}+\lambda_{fl}E_{floor}
+\lambda_e(\|\dot T_1\|^2+\|\dot T_N\|^2)
\]
- \(c_t\): DA3 confidence = affine-fit correlation × extrapolation penalty (≈0 when the
  object is far from the body — the known begin/end weakness).
- Per-segment linear Z between floor contacts; endpoint velocity damping 0.15.

### Stage B — contact-phase depth refinement (audio is here)
Only \(Z_t\) is free; \((u_t,v_t)\) fixed; \(X,Y\) follow by back-projection.
\[
E_B=\underbrace{\sum_{t\notin\mathcal A} w_{ref}\,(Z_t-Z^{ref}_t)^2}_{\text{anchor-segment reference}}
+\underbrace{\sum_{t\notin\mathcal A}\big[w_{temp}(1-\gamma_t a_t)\big]\,(\Delta^2 Z_t)^2}_{\text{audio-relaxed smoothness}}
+\underbrace{\sum_{t\notin\mathcal A} w_{aud}\,s_t\,a_t\,(Z_t-p_t)^2}_{\text{audio-gated contact pull}}
+\underbrace{w_{pen}\sum_{t\notin\mathcal A}\sum_{b}\big[\max(0,\,r-\|C_t-J^b_t\|)\big]^2}_{\text{object-out-of-human hinge}}
\]
subject to hard anchors \(Z_t=p_t\ \forall t\in\mathcal A\), where
\(\mathcal A=\{\text{visual contact events}\}\cup\{\text{VLM-promoted audio onsets}\}\)
and \(p_t\) is the depth of the **VLM-chosen** part. Post-solve: boundary-constant clamp
outside \([\min\mathcal A,\max\mathcal A]\); ray-sliding penetration cleanup on clamped tails.
Optional flight-phase physics (off by default): \(w_{xz}\|\ddot X,\ddot Z\|^2+w_y(\ddot Y-g\,dt^2)^2\)
on triplets with no contact state.

Weights: \(w_{ref}=0.7,\ w_{temp}=5,\ w_{aud}=3,\ \gamma\in[0.2,1]\) by event type,
\(w_{pen}=2\); robust soft-L1 wrapper.

### Stage C — body-side contact refinement (`refine_body_pose_contact.py`)
Free variables: masked body-pose deltas \(\delta\theta\) on the involved chain only.
\[
E_C=w_{pen}\sum_{t,b,k}\big[\max(0,\,r{+}\epsilon-\|P^{b,k}_t-C_t\|)\big]^2
+w_{tch}\sum_{t\in\mathcal T}\big(\min_k\|P^{b,k}_t-C_t\|-(r{+}\epsilon_{tch})\big)^2
+w_{pr}\|\delta\theta\|^2+w_{tm}\|\Delta\delta\theta\|^2
\]
\(\mathcal T\): contact-state intervals with gap < 4 cm. Hinge live on all frames
(touch/temporal coupling must not push clean frames in). \(w=(500,50,1,4)\), ≤5° deltas.

### Where VLM/LLM sit
- **VLM (continuous-adjacent but discrete output)**: selects anchor parts + event relevance.
  Never emits coordinates. Geometry-gated (part must be ≤140 px from object in a ±3 frame
  window) so hallucinations can't steer the optimization.
- **LLM (discrete)**: audits tables/renders post-hoc; output = failure labels + bounded
  re-run recipes (§7). Never touches pose numbers.

## 3. The audio loss today — honest assessment

What audio currently contributes: (1) anchor **timing** (promoted onsets), (2) soft pull
toward the contacting part near onsets, (3) smoothness relaxation so impact velocity kinks
survive, (4) event taxonomy → per-type γ and weights, (5) dual-modal records (silent visual
contacts kept). fps is auto-detected (30 vs 24 bug fixed — timing was 20% off before).

Known weaknesses (= improvement surface):
- **W1: onset→frame quantization.** Audio is 16 kHz; we round contact time to the nearest
  frame. Sub-frame timing is available for free and could place the anchor *between*
  frames (fractional anchor / two-frame split weights).
- **W2: amplitude unused for dynamics.** Onset energy only gates confidence; it plausibly
  encodes impact momentum change |Δv| — could scale the allowed velocity kink (large hit
  → allow large Δv; soft touch → keep smooth).
- **W3: no propagation-delay model.** Sound arrives ~3 ms/m late; at 6 m and 30 fps that
  is ~0.1 frame — small but free to correct once W1 exists.
- **W4: occlusion blindness.** When the object is invisible (behind body/motion blur), the
  2D ray is wrong/missing, DA3 conf ≈ 0 — currently the reference just interpolates. Audio
  says exactly WHEN contact happens during occlusion; nothing currently strengthens the
  audio terms when visual confidence collapses (inverse-confidence coupling).
- **W5: no scrape/roll model.** Sustained/friction events only lower γ; continuous contact
  could constrain the trajectory to the support surface for the whole interval.
- **W6: spectral content unused** beyond the 5-way taxonomy (material/part hints exist in
  the features we already compute).

## 3.5 Method critique — is our audio loss *correct*? (07-06)

Stepping back from incremental terms: what is the mathematically right way to use audio,
and where does our formulation deviate?

**The correct generative view.** An impact k is a continuous EVENT TIME τ_k with an
audio measurement `t^a_k ~ N(τ_k + ‖X‖/c, σ_a,k²)` (σ_a ≈ ms). Contact at τ_k means the
3D **surface gap vanishes**: `φ(X(τ_k), body) = 0`, and the object velocity may jump only
at {τ_k}. The principled objective treats contact times as first-class variables:

\[
E_{audio} = \sum_k \Big[\tfrac{(τ_k - t^a_k + \|X\|/c)^2}{σ_{a,k}^2}
 + w_{gap}\,φ\big(X(τ_k)\big)^2\Big]
 + \sum_{t\ \text{silent}} w_{temp}\,ρ(\Delta^2 X_t)
\qquad \text{(velocity resets only at } τ_k\text{)}
\]

**Deviation audit of our implementation:**

| Aspect | Correct form | What we do | Verdict / fix |
|---|---|---|---|
| Contact time | continuous τ_k, σ_a-weighted | hard anchor at a frame; NEW: value interpolated at fractional time (iter #1) | half-fixed. Full fix = soft anchor weight 1/σ_a² per event (cheap; queued) |
| Contact geometry | surface gap φ=0 (\|C−P\|=r) | depth equality Z_C = Z_P — **systematically biased by up to one radius** along the ray | WRONG-ish. Fix: anchor value = ray-root of \|C(z)−P\|=r nearest current z (same quadratic as `resolve_ray_penetration`) — "audio pins the surface touch, not center-to-joint depth" |
| Anchor value source | vision (part depth) with ITS uncertainty | hard pin → GVHMR part-depth bias propagates with infinite weight; left/right part flips become hard errors | soften: anchor weight should combine σ_a AND part-depth confidence (VLM conf × pose conf) |
| Silence | **silence is a measurement**: no impact ⇒ no velocity discontinuity, no new hard contact | silent frames just get default smoothness (quadratic — penalizes but permits kinks everywhere a little) | the impulse-budget hinge (in eval now) IS the silence loss: budget b_t=0 at silence forbids kinks one-sidedly; b_t ∝ A^{5/6} at onsets. Unifies relaxation + silence evidence |
| Between events | inter-onset intervals constrain flight (ballistics, restitution T_{n+1}=e·T_n) | nothing (plain smoothness) | audio-delimited flight physics (in eval now) |
| Audio as data vs gate | timing is a *measurement* entering residuals | audio only gates/selects/relaxes; all numeric values come from vision | partly by design (audio can't measure depth) — but τ_k residuals and the silence hinge make audio a first-class data term |
| Propagation delay | t_contact = t_audio − ‖X‖/c | ignored (~3 ms at 6 m ≈ 0.1 frame @30fps) | implement when σ_a weighting lands (free, self-consistent) |
| What audio can't do | say WHICH part / WHICH object | VLM chooses part (geometry-gated); relevance gate rejects off-object sounds | correct division of labor; keep |

**Bottom line:** our formulation is a frame-quantized, hard-anchored special case of the
correct event-time objective. The two real *correctness* gaps (not just refinements) are
(1) the center-depth-vs-surface-gap bias (≤ one radius, systematic) and (2) silence not
acting as evidence. Both have cheap fixes that are now queued/in evaluation. The rest
(σ_a weights, delay correction, restitution) are precision upgrades, not sign errors.

## 4. Where visual input is not enough (audio must carry)

1. **Contact instant**: at impact the object is fastest → max motion blur, max occlusion by
   the striking limb; mask/track quality is *worst exactly when the constraint matters most*.
   Audio timing is unaffected.
2. **Depth at contact**: DA3 collapses when object and body overlap (the +4.25 m global-shift
   failure came from this); audio-anchored part depth is what rescued football.
3. **Out-of-frame / full occlusion intervals**: only smoothness + audio events bound the
   trajectory.
4. **Sub-frame events**: a 30 fps camera cannot see a 5 ms bounce reversal; the mic can.
5. **Contact vs proximity ambiguity**: visually "touching" (2D overlap) ≠ contact; the
   impact sound disambiguates true contact from passing in front.

These scenarios are the paper's argument; each improvement step should include at least
one verification frame from an occlusion/impact moment, not only easy frames.

## 5. SOTA concepts to import (research pass done — audio side)

**Novelty check (verified, ~50 sources):** no prior work uses *passive* impact audio as
residual terms inside a per-clip least-squares 3D trajectory optimization for monocular
HOI. Closest lines: inverse audio physics (DiffImpact CoRL'21 PMLR v164; DiffSound
SIGGRAPH'24 arXiv:2409.13486 — recover impact timing/force/location from sound, no 3D
trajectory), acoustic human pose (PoseKernelLifter CVPR'22 arXiv:2112.00216 — audio
time-of-arrival resolves monocular scale), robotics contact mics (ManiWAV, Hearing Touch —
instrumented), sports bounce timing (Gossard et al. arXiv:2409.11760 — ms-accurate).
Safe claim: *"to our knowledge, first to use passive interaction audio as a temporal
contact constraint in a monocular 4D HOI reconstruction objective."*

Physics the audio carries (with citations in the research report):
- Modal model `s(t)=Σ a_i e^{-d_i t} sin(2π f_i t)`: contact time (sharp), material/damping,
  contact location (relative modal amplitudes) are single-mic recoverable; absolute force
  is not (gain/distance/efficiency unknown) — only **relative** amplitude within a clip.
- Hertz: contact duration `τ_c ∝ v^{-1/5}`, peak force `F ∝ v^{6/5}`, impulse `J=(1+e)mv`
  → **relative velocity change per impact `Δv ∝ A^{5/6}`**.
- Restitution from inter-onset intervals: `T_{n+1}=e·T_n` (Aguiar & Laudares 2003) —
  audio alone constrains free-flight *between* bounces. Unused in any CV paper.
- Propagation delay `Δt=‖X−mic‖/c` (2.9 ms/m): nobody corrects it; we can, self-consistently.
- Onset σ: spectral-flux ~5-8 ms; hard impacts sub-ms → per-event `1/σ_a²` weights.

**Import ranking** (impact × implementability) → iteration queue:
1. ~~Fractional-frame anchors~~ (**done, iter #1**) + per-event σ_a confidence weighting
2. Amplitude-modulated impulse budget: relaxation scale ∝ `A^{5/6}` (replaces binary γ gate)
3. Event-type residual templates: sustained/scrape events get interval-long surface-contact
   (`φ(X_t)=0` over the segment) instead of point anchors — key for mug/drawer/broom
4. Inter-onset ballistic/restitution residuals between repeated same-surface bounces (novel)
5. Propagation-delay self-correction (solve → shift anchors by −‖X‖/c → re-solve once)

Framing gem: contact-implicit trajectory optimization (Posa et al. IJRR'14) treats contact
scheduling as the hard combinatorial part — **audio is a mode oracle** that fixes the
contact schedule, reducing the MPCC to our mode-fixed least squares. Use in method section.

Evaluation gap (own-contribution angle): 3D HOI datasets (BEHAVE, InterCap, ARCTIC, HOI4D)
have no audio; AV datasets (Greatest Hits, EPIC-Sounds, RealImpact) have no 3D GT. Greatest
Hits (46.6k annotated impacts) can evaluate our onset/contact-timing accuracy; RealImpact
can calibrate amplitude↔force claims.

## 5.5 Audio EXTRACTION upgrades (research pass 2, 07-06 — processing side)

Verified findings (~60 sources): field standard for impact audio = log-mel at **10 ms/5 ms**
(EPIC-Sounds; finer temporal resolution measurably helps transients); fine-tuned encoders
beat handcrafted on benchmarks but frozen probes don't reliably (HEAR; ManiWAV even found
scratch-trained > AudioSet-pretrained for contact audio) — our handcrafted stack is
defensible, cite EPIC-Sounds' 49%-human-confidence material caveat. Decay rate τ is THE
material feature (Klatzky–Pai, near shape-invariant). Doppler/envelope speed: principled
negative for single-mic HOI (Cevher'09 verbatim). Sound-to-object attribution at contact
level (hand vs floor vs object): **no published work** — open. Audio-as-depth-residual on a
tracked object in monocular video: **open niche** (DCASE AV fusion currently makes distance
WORSE; impulsive sources are DRR-favorable). Closest 2025-26: Hearing Hands (forward:
trajectory→sound), PAVAS (video→audio w/ physics consistency — its APCC metric is adaptable
to evaluate us: does recovered impact velocity predict sound-energy ordering?).

**Implementation queue (extraction side, no training):**
1. HPSS-percussive onset detection + 5 ms-hop mel (hours; timing-safe robustness to speech/music)
2. Per-onset physical descriptor: decay τ, spectral centroid, ringing f0, band energy →
   coarse material + relative impact intensity + size-consistency cue in one routine
3. **Onset↔kinematic-discontinuity attribution** (novel): score each tracked candidate
   (object/hands/feet) by |velocity discontinuity| at the onset — turns audio from "when"
   into "who/where"; resolves multi-contact ambiguity; free from existing tracks
4. **Per-onset DRR relative depth cue** (novel, citable error budget ~octave/20-30%):
   direct-window vs tail energy ordering across bounces constrains tz ordering independent of DA3
5. Synchformer per-clip AV-offset gate (±0.2 s zero-shot) + AudioSep as onset VERIFICATION
   only (late integration; masking smears transients — never for timing)

## 6. Loss comparison with other HOI papers (research pass done)

Every optimization-based method converges on the same 5-block vocabulary; **none uses audio**:

| Method | 2D evidence | Contact | Non-penetration | Temporal | Priors / notes |
|---|---|---|---|---|---|
| PHOSA (ECCV'20) | occl.-aware silhouette L2 + edge chamfer | centroid + **hand-labeled part-pair** attraction | penetration local distance field (eq. unpublished, absent from code) | — (per image) | per-category scale Gaussian; 8 templates only (no ball/chair/hammer/mug); weights from code: sil 10, inter 20, part 50, scale 100 |
| CHORE (ECCV'22) | learned pixel-aligned UDFs + silhouette | part-aware mutual chamfer of predicted contact sets (ε=0.08) | collision 3² | — (per frame) | BEHAVE-trained, fixed Kinect intrinsics; human pinned at z₀=2.2 m |
| VisTracker (CVPR'23) | CHORE fields ×**per-frame visibility v_i** | same | same | `‖x_{i}−2x_{i+1}+x_{i+2}‖²` human λ=25, object λ_ao=225 | occlusion infilling via transformer; 6-8 h/1500 frames |
| HOLD (CVPR'24) | volume rendering RGB+segm | fingertip↔object min-dist | SDF | — | template-free implicit; ~10 h/seq A100; SfM-able objects only |
| InterTrack (3DV'25) | chamfer to per-frame diffusion point clouds | thresholded contact-set attraction | — | accel on O,R,T,s | **non-metric — eval used GT translation** (we are metric via GVHMR/DA3) |
| CARI4D (arXiv:2512.11988) | mask 0.002 + j2d 0.03 | contact-gated joint↔object dist λ=200 | ReLU(−Φ_SMPL)² λ=2 | accel λ=600/1000 | A100 80GB, 45 min/300 frames |
| WildHOI (MM'24) | keypoint λ_J=0.01 | contact λ=1 | — | — | 2D flow prior from Internet sports video (incl. basketball) |
| **Ours** | fixed 2D ray + center reproj + DA3 conf-weighted depth | **audio-timed anchors** (VLM part) + audio-gated pull + touch/penetration both directions | hinge, object↔body point clouds both sides | accel, **audio-relaxed at impacts** | + sub-frame audio timing, inverse-confidence coupling; runs on 8 GB |

What audio adds that none of them have: (i) ms-precise contact *timing* (they all rely on
learned/geometric contact detection that fails under occlusion/blur at exactly the contact
instant); (ii) contact *confirmation* (2D proximity ≠ touch); (iii) constraint continuity
through full occlusion; (iv) their "physics" = contact+penetration+floor only — never
event dynamics.

**HOI-PAGE (ICML'26, arXiv:2506.07209)**: zero-shot *generation* (text+meshes → 4D HOI), not
reconstruction — position as related-work SOTA of the generation branch, adopt its metric
*definitions* (non-collision vertex ratio, contact-frame ratio, smoothness) computed on our
data; head-to-head numbers are meaningless across the generation/reconstruction divide.

**Comparison baselines (decided):**
1. **Per-frame depth-lift** (flag-off mode of our Layer 5: DA3 back-projection, no smoothness,
   no audio, no contact) — the TAPVid-3D-sanctioned "Type I" baseline; doubles as ablation.
2. **PHOSA** (canonical wild-footage baseline; needs legacy env + authoring 4 templates —
   the manual labor itself demonstrates our template-free advantage). Fallback: InterTrack
   (modern env, but non-metric output needs alignment; 8 GB smoke test required).

**Metric protocol** (per clip): center reprojection px + projection IoU · penetration
depth/non-collision ratio · contact rate & gap at expected-contact frames · jerk
`Σ‖Δ³x‖²` · **audio-contact timing error (ms, scored on held-out events — novel metric)** ·
**known-size error** (recovered vs spec diameter: basketball 24 cm, football 22 cm — nobody
reports this; hard to game) · A/B user study.

## 6.5 Object rotation (6DoF) — plan (investigated 07-06)

Current pipeline is translation-only (exact for balls, wrong for mug/hammer/chair).
Teammate's machinery: mug = Euler+1-DOF handle-phase keyframe fit, driven largely by
MANUAL rim/bottom ellipse annotations + hardcoded per-sample frames (not portable as-is;
her handle-phase optimizer + VLM visibility gating IS reusable); chair = per-frame
rotvec+t+2 hinge angles fit to semantic 2D line segments + DA3 depth (her most automatic
piece — reusable given a segment-model CSV); `small_se3` = bounded per-contact-frame SE(3).
SOTA check: FoundationPose needs depth (DA3 works, CARI4D-style) but 8 GB requires config
surgery; MegaPose = best RGB-only tracking; 4DHOISolver/EasyHOI/PHOSA fit rotation by
silhouette + contact optimization on commodity GPUs — that's our lane.

**Implementation order (all optimization-based, 8 GB-safe):**
1. **Grasp-rigidity rotation inheritance** (highest value/effort): during contact phases,
   solve one constant `T_wrist→object` per grasp segment; inherit `R_obj = R_wrist·R_off`
   from HaMeR. Gives hammer swing + mug tipping for free. Principle: *regularize, don't
   estimate, unobservable DoF* (hammer axial roll, hidden-handle mug roll).
2. **Hammer**: mask principal axis (in-plane) + elongation ratio (out-of-plane) + chair-style
   bounded rotvec/t solve over a 2-segment model; axial roll from wrist during grasp.
3. **Mug**: port teammate's keyframe fitter, mask-only observation path, drop manual
   annotations/hardcoded frames; keep handle-phase + VLM visibility.
4. **Chair**: reuse her chair solver with a segment CSV (rotation ≈ about gravity).
5. Next sprint: FoundationPose/RGBTrack with DA3 depth as a learned-tracker baseline.

**Custom meshes** (07-06): procedural metric GLBs estimated from prompts/images in
`assets/object_meshes/` (`build_object_meshes.py`) — basketball r=.121, football r=.110
w/ pentagon patches, mug cylinder+torus handle, hammer handle+head, folding chair.
Rendered with `--object-scale 1.0`. Placeholder until SAM-3D-class generation (32 GB box).

## 7. LLM correction (end of pipeline) — design

Input: trajectory/contact CSVs + summary stats + K sampled render frames (begin/contact/end).
Audits (discrete labels): depth outliers (|Δtz| spikes), boundary drift, contact-window
mismatch vs audio events, penetration counts, anchor-part inconsistency (left/right flip),
static-object drift. Output: `llm_correction_report.md` + a bounded **re-run recipe**
(e.g. "re-run solver with --w-temp 8 on frames 200-242", "demote anchor at f105 — audio
support 0.04"). Never numeric pose edits. Verify by seeding a known error and checking the
audit catches it.

## 8. Experiment plan (basketball, football, hammer, chair)

| Sample | Data state | Missing | Object mesh |
|---|---|---|---|
| basketball_01 | complete, current best | — | textured sphere proxy (exact) |
| football_10 | complete, current best | — | sphere proxy (exact) |
| hammer (video_sample/3_hammer_video.mp4) | raw video only | sample dir, masks, tracking, GVHMR, HaMeR, DA3, events, records, solver, renders | need rigid mesh: box/cylinder proxy or SAM-3D-style export; hammer = rigid 6DoF, not sphere |
| chair (video_sample/5_chair_video.mp4) | raw video only | same as hammer | box/URDF-style proxy (teammate branch has chair URDF config) |

Comparisons (§ task 16): (i) **audio ablation** — full method vs all audio terms off
(anchors from visual only, no pull/relax); metrics: reprojection px, tz range sanity,
penetration count, jerk, contact-gap at events. (ii) two external baselines chosen by the
research pass (candidates: PHOSA-style optimization, CHORE/VisTracker if runnable, else
geometric DA3-only lift as "no-contact no-audio" reference + HOI-PAGE as qualitative
positioning). Honesty rule: document what is and isn't comparable.

### 8.1 Ablation results (07-06, current method state)

| sample | variant | tz range (m) | jerk | contact gap (cm) | audio err (ms) | pen frames |
|---|---|---|---|---|---|---|
| football | **full (audio)** | 5.94–6.72 | 0.0206 | 0.12 | **24.0** | 2 |
| football | no audio | 5.94–6.37 | 0.0074 | 0.00 | 30.7 | 1 |
| football | depth-lift baseline | **2.17**–7.10 | 0.936 | — | (22.7*) | 11 |
| basketball | **full (audio)** | 3.49–3.82 | 0.0059 | 0.08 | **118** | 2 |
| basketball | no audio | 3.49–3.77 | 0.0070 | 0.00 | 224 | 1 |
| basketball | depth-lift baseline | 3.51–3.97 | 0.0200 | — | (45.7*) | 11 |

Honest reading: (i) the depth-lift baseline collapses on football (tz→2.17 m, jerk 45×,
11 penetrating frames) — contact+smoothness is load-bearing; (ii) vs no-audio, the wins
are **audio-contact timing** (24 vs 31 ms football, 118 vs 224 ms basketball) and the
**occluded-window depth** (no-audio never reaches the body plane at football f105–120 —
tz_max 6.37 vs the visually-verified ~6.5 m); (iii) no-audio shows LOWER raw jerk because
audio deliberately relaxes smoothness at impacts — real kinks are physics, which is why
jerk must be co-reported with timing error; (iv) *the baseline's timing numbers are
jitter-gamed (a kink near every onset by chance) — the kink-detector metric needs a
held-out-events version before the paper table; (v) contact gap 0.00 for no-audio is
self-referential (its only expected-contact frames are its own hard anchors).

## 9. Iteration log (append-only; every step verified before "kept")

| # | Date | Change | Verification (frames + metrics) | Verdict |
|---|---|---|---|---|
| 0 | 07-06 | Baseline = tonight's state (boundary fix, VLM part anchors, fps fix, penetration both sides, body refine) | football edges 5.94–6.72 m flat; basketball hand rides ball; grids in scratchpad | KEPT (user-approved) |
| 1 | 07-06 | **W1 sub-frame audio anchors**: anchor depth = part depth interpolated at the audio-refined fractional contact time (`--audio-subframe-anchors`, default on) | anchors shift mean 0.55 cm vs frame-grid values (= the quantization error audio removes); no side effects | KEPT |
| 2 | 07-06 | **W4 audio-visual inverse-confidence coupling**: audio pull ×(1+κ(1−depth_conf)), κ=2 default | football occluded window f105–120 (depth_conf≈0.00–0.02): ball moves from blindly interpolated 6.25 m to body plane ~6.5 m — original frames f109/f112 show the ball AT the hip, so the audio-coupled depth is right. Basketball regression clean (mean 0.8 cm, range 3.49–3.82 m). | KEPT |
| 3 | 07-06 | **Metrics + ablation infrastructure**: `scripts/shared/evaluation/compute_hoi_metrics.py` (jerk, contact rate/gap, penetration, audio-timing error) + audio-off ablation runs | table in §8.1; audio improves timing error on both samples; occlusion-window win confirmed; identified metric flaw (kink detector jitter-gameable → needs held-out events) | KEPT |
| 4 | 07-06 | **LLM correction audit** (`scripts/shared/evaluation/run_llm_correction_audit.py`): 6 deterministic table audits + VLM render check + bounded re-run recipes (never pose edits) | both samples WARN with real findings (football: depth outlier at strike f53–54, strong unanchored audio events f57/f205, blind-interp ranges); seeded ±3 m corruption at 5 frames → detected exactly, verdict FAIL | KEPT |
| 5 | 07-06 | **Impulse budget = silence-as-evidence** (`--impulse-mode budget`, now default): smoothness becomes one-sided hinge `max(0,\|Δ²Z\|−b_t)`, `b_t = κ·A_t^{5/6}` (Hertz), b=0 at silence — unifies impact relaxation + silence evidence (§3.5 gap #2) | football jerk 0.0206→**0.0108** at identical timing/contact/pen; basketball audio-timing error 118→**44 ms**; occlusion window unchanged. Audio-flight-physics evaluated too: neutral once budget on → stays opt-in | KEPT |
| 6 | 07-06 | **Surface-gap anchors** (`--anchor-geometry surface_gap`, now default; §3.5 gap #1): anchor depth = ray-root of \|C(z)−P\|=r (surfaces touch) instead of z=part_z (centers aligned, ≤1-radius bias); closest-approach fallback when the ray can't reach touch | part-inside-object anchors 2→**0** (football), 1→**0** (basketball); mean \|dist3D−r\| at anchors 1.64→1.48 cm basketball; football residual ~10 cm gaps are ray-unreachable (2D/pose disagreement — the anchor takes the best achievable depth). `contact_gap_cm` metric "worsens" by construction (it measured the old center-depth semantics) → metric to be redefined as \|dist3D−r\| | KEPT |
| 7 | 07-08 | **Object-geometry abstraction** (`contact/object_geometry.py`): body-refine + metrics now take sphere OR capsule (SE3 pose cols → rotated axis segment; stick local **x**, L=1.86, r=0.018 from URDF — axis verified by reprojecting endpoints against her line_correspondence, <15 px). Stage C gains `--object-type/--object-length-m/--object-radius-m`; anchor sides `left+right` → both-hand parts | stick body refine vs Yixin's SE3: 80 hand penetrations found, 61 cleared, deltas ~1°, residual ≤7.4 mm; basketball/football re-refine after iter-6 trajectories: 69/69 and 2/2 cleared | KEPT |
| 8 | 07-08 | **URDF-frame rendering fixes**: `bake_urdf_to_glb --keep-origin` (recentring broke floor-origin URDFs) + `render_full_scene_3d --keep-mesh-origin`; renderer transform verified identical to her `generic_urdf_scene` (`V @ R.T + t`) | stick + chair full scenes: her SE3 pose drives the real URDF mesh over the video — chair upright & seated on its floor spot (was half-buried), stick tracks the staff through both palms | KEPT |
| 9 | 07-08 | **HOI interaction metric layer** (`evaluation/compute_hoi_interaction_metrics.py` + method doc): penetration ratio/depth (part point clouds vs surface), contact ratio/gap, part-correctness vs VLM records, jerk, grasp stability, **C_audio** (contact within ±2 fr of audio events), **accel@events vs flight**, **MDev\*** (GT-free ARCTIC MDev) | 4-case table in `samples_known_object/hoi_interaction_evaluation/`: measured pen = 0 for basketball/football/mug, 3.3% ≤7 mm stick (vs her sign-proxy "pen rate 1.0"); basketball C_audio=1.0, part-correct 0.97; football gap 144 mm at kicks = localized iter-6 disagreement; football accel@events 0.086 vs flight 0.025 (physics signature) | KEPT |
| 10 | 07-08 | **Mesh-SDF metrics geometry** (`object_geometry.MeshGeometry`, trimesh signed distance on `--keep-origin` URDF GLB, metrics-only) → chair joins the table | chair (her SE3 + articulated URDF): pen ratio 0.401, max 11.1 mm (hands in backrest, PRE-refine — mesh Stage C queued), contact gap 3.8 mm, C_audio 1.0 | KEPT |
| 11 | 07-08 | **Differentiable SDF Stage C** (`object_geometry.SDFGridGeometry`: 48³ voxel SDF of baked URDF, torch `grid_sample` trilinear — EasyHOI-style surface hinge, research import #1/#3; `refine_body_pose_contact --object-type mesh_sdf`) | grid vs exact mesh SDF agree to 4 mm; autograd verified; chair mesh Stage C cleared 86/91 hand/foot penetrations → chair pen ratio **0.401→0.229**, max **11.1→8.9 mm**, contact gap 3.7 mm. Residual = torso on backrest (hands/feet-only mask → PROX body-vertex sets queued #4) | KEPT |
| 12 | 07-08 | **Football kick-gap diagnosis** (not a fix): per-event 3D gap of ball vs named part on the audio contact-phase trajectory | part attribution correct (closest=named at 11/12 events); but gap 56–244 mm at rounded event frames. Sub-frame window scan: f53 156→10 mm @−2, f89 112→27 mm @−3 (frame quantization, recoverable) vs f57/f112/f166/f177 no window improvement (GVHMR foot-depth vs 2D-track disagreement at fast 30 fps juggling, not timing). Confirms §4 argument + scopes queue #3 to ~half the events | LOGGED |

## 10. Five-case HOI campaign (Yixin's ask, started 2026-07-08 overnight)

**Goal:** complete human modeling + object interaction results for the five final cases
in `samples_known_object/` (01_basketball, 02_mug, 05_chair, 10_football, 11_stick), then
the HOI-level evaluation layer her object-centric evaluator lacks (human-object
penetration, hand placement/part correctness, grasp plausibility) — reported next to her
`final_result_evaluation_summary`.

**Division of labor with her pipeline:** her `benchmark_vlm_qwen/object_pose.csv` is the
object SE3 source for mug/chair/stick (rotation constrained by her line/semantic
machinery); for the two balls our audio contact-phase trajectory is the object source
(rotation = identity gauge). Human side (GVHMR + HaMeR stitch + body-side contact refine)
is ours everywhere. Audio anchors/records ours everywhere.

**Phases** (football+basketball first — cheapest, everything exists; then generalize):
- **P0** sync her branch (done), plan (this section).
- **P1** bridge `samples/basketball_01`+`football_10` full human stack into her case dirs
  (same source clips — verify frame counts 192/242 + fps), full-scene renders with hands.
- **P2** run human stack on mug/chair/stick; body-refine against her object SE3; renders
  with articraft/procedural meshes (quaternion object pose in renderer — verify the
  07-06 `inherit_grasp_rotation.py` + renderer quaternion path here).
- **P3** HOI metric battery (extend `scripts/shared/evaluation/compute_hoi_metrics.py`):
  penetration depth/non-collision ratio both directions (part point clouds vs object),
  contact-frame ratio + gap at expected contacts, **part-correctness vs VLM records**,
  timing error vs audio, jerk; cross-case `hoi_interaction_summary` table + method doc.
- **P4** SOTA import for grasp placement (WHERE on the object, WHICH hand part): research
  agents on ContactOpt/ContactGen/TOCH/GeneOH/ARCTIC-class energies + metric conventions;
  import the best 8GB-runnable term into Stage C.
- **P5** audio deepening on top (σ_a per-event weights, sustained-contact interval surface
  residual — the mug/stick-relevant §5 import #3, onset↔kinematic attribution), each via
  §9 verified iterations, balls first.

**Standing rules:** one GPU job at a time (8 GB); every kept change needs render frames +
metrics; her files under `samples_known_object/*/results/benchmark_vlm_qwen` are read-only
inputs (write ours to sibling dirs, e.g. `results/human/`, `results/hoi_eval/`).

**Status after 07-08 overnight (iters #7–9):** P0–P1 done (balls bridged as `human_*`,
refreshed body refine + full-scene renders). P2 done for stick (full stack from scratch:
GVHMR→HaMeR→stitch→Qwen records→capsule body refine→URDF-mesh render), mug (body refine vs
her SE3 + render; sphere proxy r=4.8 cm), chair (HaMeR+stitch added; render with her
articulated URDF baked at median hinge angles ≈ zero-config; body refine + metrics deferred
to mesh-SDF v2). P3 v1 done (metric layer + method doc + 4-case table). P4 research done
(`docs/research_grasp_placement_sota.md`, `docs/research_hoi_eval_metrics_sota.md`).

**Next queue (priority order):**
1. **EasyHOI SDF hinge pair** into Stage C (research import #1) — replaces center/skeleton
   distance with surface-level penetration+attraction; unlocks chair + exact mug (mesh SDF
   into `object_geometry.py`, trimesh/kaolin).
2. **ContactOpt energy with audio-derived Ĉ** (import #2) — THE audio-conditioned contact
   loss for the paper; audio/VLM records supply the target contact map.
3. Football kick-frame gap (144 mm): sub-frame kick timing + foot-pose disagreement —
   candidate fix couples §5.5 #3 onset↔kinematic attribution with per-event σ_a weights.
4. BimanGrasp ‖Gc‖ two-hand term for stick; PROX vertex sets for chair sit.
5. Vertex-level metrics (mesh SDF), foot skate + ground pen/float, silhouette mIoU;
   then regenerate the five-case table including chair.
6. Chair/mug DA3 + our audio contact-phase depth refinement of HER SE3 trajectories
   (currently object poses are hers untouched; audio refinement of mug place/lift events).
