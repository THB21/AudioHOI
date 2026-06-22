# Audio→Semantic Pipeline for HOI (design + architecture comparison)

## Goal
From a (possibly generated) HOI video, extract the audio, find acoustic **signals**
(impacts/contacts/scrapes/sustained), and for each signal build a **semantic context**
by fusing the audio with the **video frames around that signal**. Consolidate everything
into **one generic semantic sheet** that downstream optimization can use to manipulate
object pose / hand pose / contact for the whole HOI — object-agnostic, so it works for
balls, mugs, hammers, drawers, etc.

This generalizes the existing basketball-specific chain
(`align_basketball_events.py` → `audio_semantics.py`) which is hard-wired to
`ball_center_y` peaks and a hand-tuned rule classifier.

## Pipeline stages (modular, `src/audio/`)
```
video.mp4 ──extract──▶ audio.wav(16k mono)
   │
   ├─ detect ──▶ onset times + scores            (3 interchangeable detectors)
   ├─ features ─▶ per-event acoustic vector        (attack/decay/centroid/bw/flatness/zcr/mfcc/hpr)
   ├─ visual_context ─▶ per-event motion/contact cues from frames + tracking + GVHMR
   ├─ classify ─▶ audio event_type + conf          (rule | cluster | pretrained)
   ├─ fuse ─────▶ audio×visual → final label + manipulation fields
   └─ sheet ────▶ ONE generic semantic sheet (CSV + JSON)
compare ─▶ run all detectors/classifiers, score them against each other → report
```

## Architectures considered for the *semantic* step (what we compare)
1. **Rule-based on DSP features (C1)** — current approach. Cheap, zero-shot, interpretable;
   brittle thresholds, no real "sound understanding", no visual grounding.
2. **Unsupervised clustering (C2)** — KMeans/GMM on standardized feature vectors discovers
   the natural acoustic modes in the clip, then maps clusters→taxonomy by centroid signature.
   No thresholds to hand-tune; adapts per clip; but cluster→label mapping still needs a rule
   and is unstable with few events.
3. **Pretrained audio tagging (C3)** — AST / PANNs (AudioSet, 521 classes) gives real semantic
   tags ("basketball bounce", "thump", "tap", "ceramic", "squeak") + an embedding per segment.
   Strongest "what is this sound" signal; mapping AudioSet→HOI-taxonomy is the only glue.
   Needs `transformers`+`torchaudio` (heavier; AudioSet not tuned to contact micro-events).
4. **Audio-visual fusion (C4)** — the key HOI step: confirm/relabel each audio onset with the
   **visual context** at that time (object velocity reversal, hand–object proximity minimum,
   optical-flow spike, which body part is nearest). This is what makes the label *grounded*
   and tells us **which entity to manipulate** (hand vs object vs support).
5. **VLM-on-frames (C5)** — ask a VLM about the frames around the onset (as the mug pipeline did
   with Qwen-VL). Richest semantics but GPU-heavy (Qwen-VL OOMs on 10GB) → implemented as an
   optional hook, not in the default comparison.

The fusion stage (C4) is not an alternative to C1–C3 but a **wrapper**: it takes the best
audio label and the visual cue and emits the final manipulation record. We compare C1/C2/C3
as the audio-label source feeding C4.

## Object-agnostic taxonomy (two physical axes + derived alias)
Sounds are described by axes every object obeys, not object-specific names:
- `interaction_mode` ∈ {impulsive, repetitive, continuous, resonant, none} → temporal constraint
- `contact_quality`  ∈ {hard, soft, friction, air, na} → constraint stiffness
- `event_type` is a derived alias of (mode×quality): strike/tap/bounce/rattle/slide/roll/ring/swish.
Maps onto any object: mug set-down=`impulsive+hard`=strike→support; grab handle=`impulsive+soft`=tap→hand;
chair drag / broom sweep / drawer=`continuous+friction`=slide; struck glass/ceramic=`resonant`=ring;
swing through air=`air`=swish. The loss reads the axes + attribution, never the object name.

## Unified semantic sheet (generic schema)
Per detected event:
`frame,time, detector, audio_score, audio_event_type, audio_conf,
 <acoustic features...>, obj_u,obj_v,obj_speed,obj_accel, nearest_part, part_dist_px,
 flow_mag, visual_cue (vel_reversal|proximity_min|flow_spike|none),
 fused_event_type, fused_conf, contact_target(part|support|either|none),
 target_entity (object|left_hand|right_hand|foot|none), manip_gamma, manip_weight, promote_anchor`

`fused_event_type` + `target_entity` + `contact_target` are the downstream handles: they say
*what happened*, *to which entity*, and *how hard to constrain it* — usable as a loss gate for
object depth, hand attachment, or support-plane contact, regardless of object class.

## Evaluation (how we decide "what works best")
No human labels, so we use **cross-modal + cross-method agreement** as the proxy score:
- **Detector quality**: temporal precision/recall of audio onsets vs a visual pseudo-GT
  (object-velocity reversals / flow spikes), and audio↔visual offset (median |Δframe|).
- **Classifier agreement**: pairwise label agreement matrix across C1/C2/C3 on the same events;
  cluster silhouette for C2; tag confidence for C3.
- **Grounding rate**: fraction of audio events with a corroborating visual cue (the higher,
  the more trustworthy the event is for loss).
Report written to `results/audio_semantics/compare_report.md` per sample.
