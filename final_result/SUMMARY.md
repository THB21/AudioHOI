# AudioHOI — Status Report

Audio-conditioned 4D human–object interaction reconstruction from monocular (generated) video.
Current state on branch `tom`. Samples: **basketball**, **football** (`samples/`), **mug**
(`samples_known_object/`).

---

## 1. The core idea: audio says *when*, vision says *where*

A single modality is incomplete:
- **Audio** only fires on *loud, rigid* contacts — a ball hitting the floor, a cup set on a
  table. Soft/silent contacts (a hand pushing the ball, fingers on a mug handle) make **no
  sound** and are invisible to audio.
- **Vision** sees the silent contacts and *where* on the body/object they happen, but has no
  precise contact instant and cannot tell a real impact from a near-miss.

So we extract events from **both** streams and fuse them into one contact record. This is the
central mechanism of the whole pipeline.

---

## 2. Pipeline (modular, `src/audio/`)

```
video.mp4
  └─ extract.py        ffmpeg → audio.wav (16 kHz mono)
  └─ detect.py         3 onset detectors (spectral-flux / onset-strength / HF-transient) + NMS union
  └─ features.py       per-event 13-D acoustic vector
  └─ classify.py       object-agnostic event type (rule / KMeans cluster / pretrained AST)
  └─ visual_context.py object kinematics + body-part proximity (from cached ViTPose) + frame motion
  └─ seed.py           DUAL-MODAL: merge audio onsets + silent visual hand-contacts
  └─ fuse.py           audio × vision → final event + manipulation handles
  └─ vlm.py / vlm_pipeline.py   per-event VLM window analysis (Qwen2-VL)
  └─ persistence.py    grasp/contact state machine (held vs touched)
  └─ contact_record.py UNIFIED contact sheet (the loss input)
  └─ body_surface_contact.py   contact point on the body surface (finger / foot)
  └─ grasp_attach.py / hand_stabilize.py   sustained-grasp corrections
  └─ loss.py + (real loss) replace audio term in the object depth optimizer
```

Run a sample: `PYTHONPATH=. python -m src.audio.pipeline --sample-dir <dir> --classifier rule`
Compare all samples/approaches: `... --compare`

---

## 3. What exactly is extracted

### 3a. Audio events (`detect.py`, `features.py`)
Per detected onset:
- **timing**: `frame`, `time`, `audio_score`
- **13 acoustic features**: attack sharpness, decay time (1/e), spectral centroid, bandwidth,
  flatness (noisiness), HF energy ratio (>2 kHz), zero-crossing rate, RMS loudness,
  harmonic/percussive ratio, 4 MFCCs.

These features *discriminate the object* even though the code is object-agnostic:
| | basketball | football | mug |
|---|---|---|---|
| attack | sharp 0.77 | sharp (strikes 0.95) | soft 0.57 |
| decay | 15 ms | 18 ms | 48 ms (dull) |
| centroid | 1435 Hz | 1587 Hz | 1254 Hz (ceramic) |
| → type | hard/repetitive | hard/impulsive | soft/dull |

→ `samples/<s>/results/audio_semantics/semantic_sheet_rule.csv`, figure `images/01_audio_extraction_3objects.png`

### 3b. Object-agnostic semantic taxonomy (`classify.py`)
Instead of object-specific labels, every sound is described on **two physical axes** that any
object obeys, plus a readable alias:
- `interaction_mode` ∈ {impulsive, repetitive, continuous, resonant, none} → *what the loss does in time*
- `contact_quality`  ∈ {hard, soft, friction, air, na} → *constraint stiffness*
- `event_type` = alias: strike / tap / bounce / rattle / slide / roll / ring / swish.

This generalizes: mug set-down = impulsive+hard = *strike→support*; grab handle = impulsive+soft
= *tap→hand*; chair drag / drawer / broom = continuous+friction = *slide*; struck ceramic =
*ring*. Same code for a ball, a mug, a guitar.

### 3c. Visual context (`visual_context.py`)
At each event, from the frames around it:
- object 2D position / speed / acceleration / vertical velocity reversal (= bounce signature)
- nearest body part + its pixel distance (from cached **ViTPose** COCO-17 — no extra net)
- frame-difference motion magnitude
→ gives the *visual cue* (`vel_reversal` / `proximity_min` / `flow_spike`) that grounds the event.

### 3d. Dual-modal seeding (`seed.py`) — the key step
- audio onsets → loud contacts (precise timing)
- **visual hand↔object proximity minima** → silent contacts the mic never heard
- merged, each event tagged `source ∈ {audio, visual, audio+visual}`.

Result on basketball: 18 audio-only events → **31** events that correctly alternate
`bounce→support [audio]` (the floor bounce) and `tap→hand [visual]` (the silent hand push) —
the real dribble structure.

### 3e. Fusion → manipulation handles (`fuse.py`)
Each event becomes the object-agnostic loss handle:
`fused_event_type, contact_target (part|support|either), target_entity
(object|left_hand|right_hand|left_foot|right_foot|support), manip_gamma (smoothness relaxation),
manip_weight (pull strength), promote_anchor`.
Attribution priority: hands are primary manipulators; a velocity reversal with no *active* limb =
surface rebound; a fast foot = kick (so a floor bounce is not falsely pinned to a near foot).

### 3f. VLM per-event analysis (`vlm.py`, `vlm_pipeline.py`)
For each event we cut a montage of frames (−4,−2,0,+2,+4) and ask **Qwen2-VL-2B** (local,
fits 8 GB): contact? which part / object? impact (none/light/medium/hard)? relevant? The audio
peak then *refines the timing* of silent visual contacts. A relevance gate drops audio with no
visual corroboration + weak energy + VLM "none" (e.g. football: 17 onsets → 11 real kicks kept).
We also adopted the teammate's fine grasp prompt (`--backend qwen_grasp`: hand_side, fingers,
object_region, grasp_type).

### 3g. Contact points (where on the human and the object)
- **Object side** (adopted from teammate): the contact point in *object-local* coords +
  semantic part, e.g. mug `handle_loop` at (−0.077, −0.036, 0) — exactly on the handle mesh.
- **Body side** (`body_surface_contact.py`): nearest SMPL-X body-surface vertex to the ball +
  named region from the nearest finger/foot joint.
  - basketball → hand: `right_palm`, `right_pinky`, `left_thumb`
  - football → foot: `left/right_foot_heel`, `toe`
  → `results/audio_semantics/body_surface_contacts.csv`

### 3h. Persistence — held vs touched (`persistence.py`, `grasp_attach.py`, `hand_stabilize.py`)
A short touch constrains one instant; a *held* object must stay attached for the whole grasp
interval (adopted from the teammate's grasp-anchor state machine). For the mug:
- `grasp_attach`: object follows the hand so the contact point stays glued → 12.5 cm → 0 cm.
- `hand_stabilize`: freeze the finger pose during a grasp (HaMeR jitters under occlusion) →
  7.4° → 1.1°.

---

## 4. How audio enters the loss

The full object-depth optimizer
(`scripts/.../run_human_ball_contact_phase_calibration_anchorinterp_generic.py`) already has
an audio term (visual contact anchors + support plane + smoothness + audio). We **replaced** its
legacy audio source with the new `contact_records.csv` via `--audio-records-csv`.

`E = w_data·Σ‖p−p_obs‖² + w_smooth·Σ relax(t)·‖p̈‖² + Σ_contacts 𝟙[relevant]·w·‖p−target‖²`
where `relax(t)=1−manip_gamma` at audio contacts → **audio decides when the motion may kink**.

Effect on basketball: with audio the dribble bounces survive at the detected contact instants;
without audio uniform smoothing flattens them (bounce kink 5× sharper with audio).
→ figure `images/02_loss_with_vs_without_audio.png`

---

## 5. 3D scene reconstruction (`scripts/shared/human_ball/render_full_scene_3d.py`)

- **Body**: GVHMR SMPL-X (`run_gvhmr.py --static-cam`).
- **Hands**: HaMeR per-frame MANO → stitched into the body (only finger articulation, anchored
  to GVHMR wrists; robust to focal mismatch in generated video).
- **Object**: ball as a textured sphere at the lifted 3D trajectory; mug as the real **6DOF
  Articraft mesh** via `--object-pose-csv` (handle anchored to the *finger* grasp point, yaw so
  the handle faces the hand, pitch tilting toward the mouth while drinking).
- **Contact marker** (optional `--contact-csv`): a sphere at the body-surface contact point.

→ figures `images/03_body_hands_3objects.png`, `images/07_mug_real_vs_render.png`

---

## 6. Status & honest limits

- **Basketball / football**: full from-scratch (SAM2 + CoTracker tracking → audio events →
  GVHMR → HaMeR → audio-gated loss → render). Reprojection ~1.4 px.
- **Mug** (new `2_mug_video.mp4`, 192 frames, a *different* take than the teammate's 240-frame
  data): body + hands measured from scratch; the **object 6DOF pose is roughly estimated**
  (handle-at-fingers + yaw-to-hand + drinking tilt), not a measured fit — the teammate's
  hand-authored mug pose cannot be regenerated for a new video.
- VLM is a small **2B** model → occasional wrong object labels; foot region naming uses the
  nearest joint (a kick reads "heel" when the ball is near the ankle/midfoot).

## Deliverables (`final_result/`)
- `videos/1_basketball_3d.mp4`, `2_football_3d.mp4`, `3_mug_3d.mp4` (+ `3_mug_3d_world.mp4`)
- `images/01..07` (audio extraction, loss with/without audio, body+hands, frames, mug real-vs-render)
