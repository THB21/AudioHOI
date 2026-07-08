# SOTA research: hand-object grasp/contact placement (2026-07-08 research agent pass)

Ranked import list for AudioHOI Stage C / grasp placement; formulas verified against papers+code.
Feeds loop_plan §10 P4. Summary of the full agent report:

## Ranked imports (impact × implementability, 8GB, no retraining)
1. **ContactOpt DiffContact energy** (CVPR'21, arXiv:2104.07267, MIT code) — virtual-capsule soft
   contact C_O=min(c_rad/φ,1), asymmetric λ=3 missing-contact penalty, 2mm free interpenetration.
   ~30 lines torch. KEY: its learned target map Ĉ is replaceable by **our audio/VLM contact
   records → the audio-conditioned contact energy nobody has published**.
2. **ContactGen** (ICCV'23, arXiv:2310.03740, ckpt released, NO license file) — object-centric
   contact map + 16-part map + direction map CVAE. Solves WHERE on the object + WHICH hand part
   (mug handle vs body). Sample once per contact event, freeze as residual targets. Right-MANO
   only (mirror for left).
3. **EasyHOI SDF hinge pair** (CVPR'25, arXiv:2411.14280, MIT) — L_pen=mean max(0,−d(v)) over all
   hand verts + L_contact=Σ max(0,d(v)) over contact-region verts + L1 stay-near-HaMeR. Direct
   upgrade of our center-distance hinge to surface-level. Needs object mesh SDF (sphere proxy ok).
   Hand-side region weighting: GrabNet rhand_weight.npy (>0.8) or ObMan 6 palmar regions.
4. **DexGraspNet/DFC force closure + BimanGrasp bimanual** (ICRA'23 arXiv:2210.02697; RA-L'24
   arXiv:2411.15903) — analytic ‖Gc‖² with G∈R^{6×24} stacking BOTH hands: the only term that
   makes opposed two-palm stick grips a minimum. Soft prior weight only (zero-friction relaxation).
5. **PROX contact-vertex sets + Geman-McClure attraction** (ICCV'19, arXiv:1908.06963) — released
   SMPL-X vertex sets (gluteus 113, thighs 62, back 222, hands 725, feet 194) → chair sit.
   Gate V_s by VLM target_entity; persist matched pairs across frames (InterTrack trick,
   arXiv:2408.13953) to stop contact sliding.
6. Optional labelers: LEMON (CVPR'24 2312.08963) affordance prior; TriDi (ICCV'25 2412.06334);
   InteractVLM/PICO (CVPR'25 2504.05303 / 2504.17695 — PICO independently validates our
   chain-restricted body refine). NOT importable: TOCH (no ckpt), GeneOH (black-box denoiser),
   HOISDF (feed-forward), GraspXL (RL), ManipNet (NC demo), HOLD (10h/A100).

## Case → technique map
mug: ContactGen map + LEMON + EasyHOI | stick 2-hand: BimanGrasp Gc + per-hand ContactOpt |
basketball palm: ContactOpt w/ audio-timed Ĉ | chair: PROX sets + VLM part gating.

## Audio novelty (adversarially checked)
No method uses audio for grasp/contact placement. Near-neighbors to cite defensively:
VibeMesh (2508.00852, ACTIVE acoustics on-hand), FürElise (2410.05791, MIDI not mic),
ObjectFolder/RealImpact (2306.00956/2306.09944 — impact sound carries contact location; best
motivation cite + likely reviewer challenge). Robot contact-mic line: Hearing Touch, SonicSense,
SonicBoom. Structural precedents for evidence→contact-map conditioning: NL2Contact (ECCV'24),
ClickDiff (2407.19370). Closest full-pipeline competitor: CHOIR (2605.20992, May'26) — no audio.
Claim stays: "passive interaction audio deciding contact region + hand part inside a 4D HOI
reconstruction loop" is unclaimed.

## Recommended integration order
(1) EasyHOI hinge pair replacing center-distance hinge → all cases;
(2) ContactOpt energy with audio-derived Ĉ → THE paper contribution;
(3) ContactGen sampling for mug-class placement;
(4) BimanGrasp G for stick;
(5) PROX sets for chair.
