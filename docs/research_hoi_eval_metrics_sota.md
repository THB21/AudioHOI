# SOTA research: HOI interaction evaluation metrics (2026-07-08 research agent pass)

GT-free plausibility protocol for the five in-the-wild cases. All claims verified against
primary sources by the agent; feeds compute_hoi_interaction_metrics.py + the eval method doc.

## Citation corrections (important)
- Non-collision/contact score pair = **PSI (CVPR'20, 1912.02923)**, not PROX. PROX = 1908.06963.
- Voxel interpenetration volume protocol = ContactGen (1mm³) / ObMan (0.5cm), NOT GRAB's paper.
- CHORE/VisTracker report NO contact F-score (contacts are losses only). Contact P/R/F1 with GT
  = OMOMO/CHOIS. VisTracker = 2303.16479.
- ARCTIC CDev/MDev REQUIRE GT — adapt structure only (see MDev* below).
- CARI4D, InterTrack, HOLD: in-the-wild = qualitative only → our GT-free layer fills a real gap.

## Recommended battery (A–D)
A. Penetration: A1 non-collision vertex ratio (PSI form; saturates ~0.97-0.99 — always pair
   with A2); A2 penetration depth mean (CHOIS form) + max (ObMan form); A3 intersection volume
   (0.5cm voxels) only for mug/stick grasps. InterCap GT-quality bar: 7.2mm mean pen at contact.
B. Contact: B1 contact-frame ratio + **audio-windowed variant C_audio (novel, ours)** + false
   contact rate outside events; B2 contact gap = median closest H-O dist at expected-contact
   frames (CHOIR protocol: HOLD 36cm vs 0.43cm — exposes floating); B3 part correctness
   @{1,3,5}cm vs VLM/audio expected part (HOI-PAGE part-level accuracy precedent; COUCH for sit).
C. Temporal/physics: C1 accel/jerk human+object, **split at-event vs flight (novel)** — spike AT
   audio event = physics, elsewhere = artifact; C2 foot skate ratio (GMD thresholds 2.5cm/frame,
   h<5cm) + ground Penetrate/Float 5mm tol (PhysDiff); C3 **MDev\* = GT-free ARCTIC MDev during
   predicted contact windows** (mug/stick hold rigidity, bimanual for stick); C4 simulation
   displacement — skip v1.
D. Perceptual: D1 2AFC user study, PHOSA protocol (equal=50%), full-vs-no-audio + vs external
   baseline, ≥20-30 raters, two axes; D2 VLM judge only secondary (TRAVL shows VLMs weak on
   implausibility; binary questions, majority-of-3, disclose Qwen dual-use); D3 rendered-
   silhouette mIoU vs SAM2 masks (ties smoothness to evidence; align with Yixin's overlay proxy).

## Reviewer tiers
T1 expected: A1+A2, B1, C1, C2, D1. T2 strengthens: B2, B3, A3, D3.
T3 novel ours: audio-windowed contact ratio, event-vs-flight accel split, MDev*.
Gaming pairs to co-report: penetration↔contact; smoothness↔image alignment.

## Per-case keys
basketball: B1-audio+C1 | football: B1+C2 plant foot | mug: B2+C3 | chair: A2+C2 | stick: B2+C3 bimanual.

(Full citation list + typical values in the agent transcript; key ids: PSI 1912.02923,
ObMan 1904.05767, CHOIS 2312.03913, CHOIR 2605.20992, HOI-PAGE 2506.07209, PhysDiff 2212.02500,
GMD 2305.12577, ARCTIC 2204.13662, COUCH 2205.00541, PHOSA 2007.15649, TRAVL 2510.07550.)
