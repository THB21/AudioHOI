# Comparison to Related Work

How AudioHOI relates to recent HOI/HSI methods, mapped **onto our own loss terms** and the
diagnostics produced this session. Our energy (see `method_losses.md` §5) is

```
T* = argmin Σ_t [ w_mask·R_mask + w_kp·R_kp + w_depth·R_depth + w_center·R_center
                + w_contact·R_contact + w_support·R_support ]  +  w_reg·R_reg
```

with audio onsets gating/timing `R_contact`, and a VLM critic (§7) sitting above the optimiser.
The principle that separates us from the LLM/VLM-generation papers: **LLM/VLM only produce
discrete semantics or pass/fail gates; the optimiser is the only continuous solver.**

Artifacts referenced below were generated this session:
`results/diagnostics/sources/*.mp4`, `results/diagnostics/curves/loss_curves.png`,
`results/diagnostics/llm_reasoning/*` (per sample).

## 1. Mapping table

| Method | Core idea | What it needs | Our term that plays the role | Evidence (this session) |
|---|---|---|---|---|
| **InterCap** | contact is a geometric constraint, not an annotation; human+object jointly | multi-view RGB-D, known mesh | `R_contact` (`method_losses.md` §5, audio-gated) | `contact_overlay.mp4` + `E_contact` curve: gap→0 at the 24 basketball contact frames; 7 football foot-contacts |
| **MOVER** | human motion constrains depth ordering, support, free space | monocular scene optimisation | `R_depth` + `R_support` + ground penalty (§3) | depth panel of `loss_curves.png`: solved `tz` vs DA3; **mug finding** — DA3 depth disagrees with pose tz → depth is a weak constraint there |
| **HOI-PAGE** | LLM infers a part-affordance graph (object parts ↔ human parts) | text prompt + object parts | LLM **discrete** prior only (never pose) — same contract as our Step-3 reasoner | `llm_reasoning/flags_claude.json`: LLM reasons over parts/semantics, emits discrete flags, not coordinates |
| **InteractAnything** | LLM relationship/affordance reasoning for open-set objects | LLM feedback loop | LLM forbidden from continuous pose / weights | Step-3 LLM outputs `{frame, field, issue, action∈{interpolate,level_align,none}}` — discrete actions, code applies them |
| **GenZI** | VLM imagines a plausible human/interaction, optimiser then solves geometry | VLM inpainting over views | VLM as critic/generator (§7) + our **Nano Banana object-removal** for occluded hands (Step 1) | `scripts/shared/hands/vlm_remove_object_frame.py` (Gemini 2.5 Flash Image); the optimiser, not the VLM, sets pose |
| **ZeroHSI** | video-generation is a zero-shot motion prior; reconstruct by differentiable alignment | generated video + diff-render | our inputs **are** generated video → trust via `R_center`/`R_kp` alignment, not blindly | `reproj_overlay.mp4` + `E_2d≡0` in `loss_curves.png`: centroid reprojection is solved exactly, so 2D is a hard, reliable anchor |
| **InterDiff / CoopDiff** | dynamic HOI must be contact-consistent *over time*, not per-frame | physics-informed diffusion | `R_reg` acceleration smoothness (§6) + contact intervals | `E_smooth` curve (basketball tz-accel spikes at contacts); **Step-3** removed the mug `yaw` π-flip (0.177→0.064 max accel) — temporal consistency without diffusion |
| **Open3DHOI / Gaussian-HOI** | HOI = joint structure of human+object+contact region+semantics; explicit contact outputs | neural/Gaussian reconstruction | structured CSV outputs incl. explicit contact points | per-sample `SUMMARY.md` + `object_contact_points` / contact-gap columns; we keep the structured outputs without photoreal reconstruction |

## 2. Where each idea actually shows up in our results

- **Contact as geometry (InterCap).** `E_contact` is not decoration: on basketball the gap
  collapses to ~0 exactly at the dribble contacts and the audio-flagged frames; this is the
  residual that pins object depth at the touch instant (`method_losses.md` §4 audio anchors).
- **Human-constrained depth (MOVER).** Our `R_depth` ties DA3 to the metric GVHMR body
  (§2 per-frame affine). The mug diagnostic exposes the *limit* of this idea honestly: when the
  object barely moves in depth (mug Z ≈ 2.0–2.1 m, near-constant) the affine is ill-conditioned
  and DA3 disagrees with the solved `tz` — so we down-weight depth there rather than trust it.
- **LLM = discrete semantics only (HOI-PAGE / InteractAnything).** Demonstrated literally in
  Step 3: Claude reasoned over the mug trajectory + the prompt ("sip and place, no tilting") and
  produced *flags*, while a deterministic pass did the continuous edit. The LLM never wrote a
  pose value.
- **VLM helps but doesn't solve (GenZI / ZeroHSI).** Our input is AI-generated video, so we
  treat it as evidence: 2D reprojection (`E_2d≡0`) is a hard anchor, and the VLM is used for
  conservative edits/gates (object removal for occluded hands, plausibility critique) — not to
  set geometry.
- **Temporal consistency (InterDiff / CoopDiff).** We get it from `R_reg` + contact intervals
  instead of a diffusion prior. The clearest example is the mug `yaw` handle-flip: a per-frame
  fit accepted a physically implausible 180° flip-and-return; the temporal reasoning step
  identified and removed it.

## 3. Why we use method proxies, not the original code

These methods assume conditions we don't have (multi-view RGB-D, calibrated rigs, motion
*generation* rather than generated-video→6D/contact recovery, or neural reconstruction with
different outputs). So we borrow the **idea as a loss term or gate** and validate via ablation
of information sources (no-contact, no-depth, no-audio, no-VLM, no-LLM-prior), each of which maps
to dropping one term above. The Step-2 diagnostics are the per-source evidence that tells us
which term actually carries the reconstruction for each object:

- ball cases: `R_center` is exact and `R_contact`/`R_reg` do the real work; `R_depth` is a soft helper.
- mug: `R_depth` is weak; rotation needs the part/handle semantics and temporal consistency
  (the `yaw` artifact), which is exactly where InterCap-style contact + InterDiff-style temporal
  ideas matter most.

## References

- InterCap — https://intercap.is.tue.mpg.de/
- MOVER — https://vlg.inf.ethz.ch/publications/Mover-Human-Aware-Object-Placement-for.html
- HOI-PAGE — https://craigleili.github.io/projects/hoipage/
- InteractAnything — https://jinluzhang.site/projects/interactanything/
- GenZI — https://openaccess.thecvf.com/content/CVPR2024/papers/Li_GenZI_Zero-Shot_3D_Human-Scene_Interaction_Generation_CVPR_2024_paper.pdf
- ZeroHSI — https://awfuact.github.io/zerohsi/
- InterDiff — https://sirui-xu.github.io/InterDiff/  ·  CoopDiff — https://arxiv.org/html/2508.07162v1
- Open-vocabulary / Gaussian-HOI 3D HOI — https://wenboran2002.github.io/3dhoi/
