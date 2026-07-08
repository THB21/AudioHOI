# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AudioHOI** is a research project for **audio-conditioned 4D human-object interaction reconstruction**, focusing on monocular video analysis where audio and visual signals jointly constrain object and human motion estimation.

The current implementation centers on a basketball interaction baseline, with infrastructure designed to generalize toward irregular objects (hammers, mugs, drawers, chairs, brooms) through event-centric constraints rather than object-specific geometry.

### Core Research Direction

The project transitions from object-specific pipelines (e.g., basketball) toward a general framework organized around:
- **Event types** (impact, placement, closure, scrape, sustained contact) rather than object categories
- **Object state representation** (mask, bbox, sparse keypoints, contact candidates, pose proxy)
- **Multi-modal constraints** (mask reprojection, keypoint reprojection, support plane, contact, temporal smoothness, audio timing)

See `object_generalization_pipeline_en.md` for the full generalization roadmap, and `method_losses.md` for the mathematical overview of the depth-alignment, lifting, and contact-phase loss functions.

## Architecture & Main Pipeline Components

### Layer 1: Object Initialization
Scripts for detecting and segmenting the target object:
- **Manual init** (`scripts/manual_init/`): Frame extraction and SAM2 video segmentation for basketball
- **Known-object init** (`scripts/known_object_init/`): Grounding DINO + SAM2 for arbitrary object categories

### Layer 2: Object Tracking
- **CoTracker** (`scripts/shared/tracking/run_cotracker_basketball.py`): Tracks sparse ball points (center, left, right, top, bottom) for 2D trajectory
- Output: `results/tracking/ball_trajectory.csv` (fitted circle from masks: u, v, r)

### Layer 3: Human Motion (GVHMR)
- **GVHMR** (`scripts/run_gvhmr.py`): Estimates human pose/shape in world-grounded gravity-view coordinates
- Integrates HMR4D utilities for preprocessing (pose extraction, visual odometry)
- Output: `results/gvhmr/result.pkl` (per-frame SMPL-X parameters)

### Layer 3.5: Hand Pose (HaMeR)
- **HaMeR** (`scripts/shared/hands/run_hamer_hands.py`): Recovers per-frame MANO hand mesh for both hands
- Uses GVHMR wrist projections as crop hints — no separate hand detector needed; works on real and diffusion-generated video
- Stitches HaMeR finger articulation to GVHMR wrist anchors: only relative finger configuration is taken from HaMeR (not its absolute translation), making it robust to focal-length mismatches in generated content
- Applies optional Gaussian temporal smoothing to fingertip positions (important for generated video jitter)
- Outputs: `results/hands/hand_keypoints_3d.csv` (fingertip + palm in GVHMR camera frame), `results/hands/hand_mano_params.pkl` (full MANO params for downstream stitching)
- Requires: `pip install git+https://github.com/geopavlakos/hamer.git` + HaMeR checkpoint

**MANO 21-joint convention used** (thumb-first, HaMeR standard):
`0=wrist, 1-4=thumb(CMC/MCP/IP/TIP), 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky`
Fingertip indices: thumb=4, index=8, middle=12, ring=16, pinky=20

- **Stitch** (`scripts/shared/hands/stitch_hands_to_body.py`): Merges HaMeR MANO finger pose into GVHMR SMPL-X params. Converts HaMeR `hand_pose` [N,15,3,3] rot mats → axis-angle [N,45] via `cv2.Rodrigues`; linearly interpolates undetected frames. Body pose and wrist orientation stay from GVHMR; only finger articulation is replaced. Output: `results/hands/stitched_smplx_params.pkl`

### Layer 4: Audio-Visual Event Alignment
- **Event detection** (`scripts/shared/events/align_basketball_events.py`): Detects audio impacts and aligns to visual bounce frames
- Outputs: `results/events/audio_events.csv`, `visual_events.csv`, `audio_visual_alignment.csv`
- Audio-visual alignment is critical for constraining 3D reconstruction

### Layer 4.5: Metric Depth (Depth Anything 3)
- **DA3 depth** (`scripts/shared/depth/run_depth_anything_v3.py`): Per-frame metric depth on *any* object, replacing the sphere-only `Z = f·R/r` size cue.
- Monocular metric depth (even DA3's metric model) drifts in scale/offset per scene and per frame, so DA3 depth is **affine-aligned to GVHMR's metric human in the same camera frame**: SMPL-X body joints (0–21) are projected, DA3 depth is sampled at those pixels, and a robust affine `Z ≈ a·D + b` is fit **per frame** (DA3 is inferred per-frame, so its absolute scale drifts; a single global affine collapses the depth signal — per-frame nearly doubles correlation with the sphere baseline). The per-frame `(a,b)` are interpolated over sparse frames and lightly Gaussian-smoothed in time. Because the affine absorbs all global scale, the DA3→metres `focal·net/300` conversion is irrelevant and skipped — raw `pred.depth` is fed straight in.
- Object depth per frame = robust median of the aligned map over the object mask (`results/segmentation/masks/`), falling back to a disk around the `object_observations.csv` center when no mask exists (mask-free, box-tracked objects).
- Outputs: `results/depth/object_depth.csv` (`frame,time,object_z_raw,object_z_aligned_m,affine_a,affine_b,sample_count,depth_conf,source`), `depth_alignment.json`, `object_depth_preview.png`.
- Runs in the `gvhmr` env (DA3 installed there; uses `smplx` for joints). Validated on basketball: DA3-aligned depth matches the known-sphere baseline (tz correlation 0.70, ~13 cm mean diff) with no sphere assumption.

### Layer 5: 3D Reconstruction (Shared-Camera Baselines)

Two parallel branches in the shared full-image camera frame:

1. **Shared-Camera Ball Baseline** (`scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py`)
   - Lifts the 2D object trajectory to 3D with a pinhole model + least-squares fitting. The only motion prior is a temporal smoothness regularizer (penalize position acceleration) — no gravity/ballistic assumptions, so it stays object-agnostic.
   - `--depth-source sphere` (default): legacy size cue `Z = f·R/r` with fixed radius (0.12 m); X from lateral back-projection of u; Y contact-constrained (ground-anchored). Output: `results/pose6d_sharedcam/`.
   - `--depth-source depthv3`: object-agnostic. Initializes Z from `results/depth/object_depth.csv` and adds a metric-depth residual (`--depth-weight`, default 1.0); the sphere-specific mask-chamfer and size residuals are disabled (shape-independent center reprojection, temporal smoothness, and contact constraints remain). Output: `results/pose6d_sharedcam_depthv3/`.
   - In depthv3 mode, masks are optional — mask-free object-observation inputs are accepted.

2. **Contact-Phase Refinement** (`scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp_generic.py`)
   - Refines only object depth while fixing the 2D observations.
   - Contact frames are anchors (object depth pinned to the auto-selected contacting body part); free frames are solved for. The only default prior is a smoothness regularizer (depth acceleration) — no gravity/ballistic term. Pass `--ball-trajectory-csv` to refine the DA3 (`pose6d_sharedcam_depthv3`) trajectory.
   - Free-frame depth reference defaults to `--z-ref-mode anchor_segment`: piecewise-linear between anchor depths, constant before the first / after the last anchor. The old `global_shift` mode (z_init + median shift) inherits initial-depth errors — on football it pushed the clip's first/last frames ~5 m too deep. `--outside-window-mode boundary_constant` (default) additionally clamps post-solve depth outside the anchor window to the boundary anchors.
   - Optional flight-phase physics priors (`--w-phys-xz`, `--w-phys-y`, default 0 = off): on triplets where contact states say the object touches nothing, penalize X/Z acceleration and pull Y acceleration toward gravity. Object-agnostic but trusts the contact states, so opt-in.
   - Penetration correction (`--w-pen`, default 2.0 = on): the object center must stay ≥ radius + `--pen-margin-m` from human part centers (palms, feet, body joints 0–21) on free frames — a hinge residual in the solve, plus a post-clamp ray cleanup (`resolve_ray_penetration`) for boundary-clamped frames the residual can't reach. Contact-adjacent frames may legitimately sit at the boundary (fingers wrap the ball at a dribble release).
   - **VLM anchor-part reasoning**: pass `--audio-records-csv results/audio_semantics/contact_records.csv` so anchors pin to the body part the VLM identified. Chain: `src/audio` events → Qwen-VL names the contacting part per event window (`vlm_pipeline.py`) → overrides the fuse nearest-part heuristic **only if** fuse attributed a part (support/bounce verdicts stand) **and** the named part passes a geometry check (≤140 px from the object within ±3 frames; NaN = no tracking = trust VLM) → `contact_records.csv` carries `target_entity` → the solver overrides contact labels at event frames and re-picks the anchor part depth (e.g. football anchors pin to the kicking foot, not the nearer dangling hand). Regenerate records with `python -m src.audio.contact_record --sample-dir <dir> --backend qwen`.
   - The `contact_part_utils` labels also read the `anchor_type` column of the contact-candidate CSVs (hand for a dribble, foot for a kick) as fallback when no records are given.
   - Output: `results/pose6d_sharedcam_contactphase*/` variants (e.g. `..._depthv3`).

**src/audio fps note**: all `src/audio` entry points (`pipeline`, `vlm_pipeline`, `contact_record`, `compare`) now auto-detect fps from `video.mp4` (`--fps 0` = auto). The old hardcoded 24 fps default silently misplaced every event on non-24fps clips (football_10 is 30 fps → all record frames landed at 0.8× their true frame).

3. **Body-Side Contact Refinement** (`scripts/shared/human_ball/contact/refine_body_pose_contact.py`)
   - The object solver can't move the object at anchor frames (it's pinned), so a hand/foot GVHMR placed inside the object stays inside. This stage optimizes the BODY instead: per offending frame it frees only the involved part's kinematic chain (collar/shoulder/elbow/wrist or hip/knee/ankle) and solves for (a) part center outside the object (penetration hinge) and (b) at contact frames, part resting on the surface (hand on top of the dribbled ball) — with a stay-close pose prior + temporal smoothness, so corrections stay in the few-degree range.
   - Geometry is per-part point clouds, not centers: hands = wrist + 15 finger joints + 5 fingertips (SMPL-X extra joints 66–75), feet = ankle + foot + toes/heel (60–65). EVERY point must clear the surface (+`--point-margin-m` 8 mm flesh allowance) — palm-center-only left fingers inside the ball; the CLOSEST point carries the touch (`radius + 1 cm`). `--pull-range-m` (10 cm) refuses to fake contact when tracking and body pose genuinely disagree.
   - Runs after the contact-phase solver (reads its trajectory CSV). Output: `results/contact_refine/contact_refined_smplx_params.pkl` + report CSV. Renderers automatically prefer contact-refined params > stitched > raw GVHMR.
   - `conda run -n gvhmr python scripts/shared/human_ball/contact/refine_body_pose_contact.py --sample-dir samples/<name>`

### Layer 6: Rendering
- **SMPL-X overlay** (`scripts/shared/human_ball/render_smplx_pyrender_overlay.py`): Primary full-body renderer — shaded SMPL-X mesh composited over original video frames using pyrender + EGL. Auto-picks up stitched hand params (`results/hands/stitched_smplx_params.pkl`) when present. Output: `results/renders/smplx_overlay.mp4`.
- **Unified 3D scene** (`scripts/shared/human_ball/render_full_scene_3d.py`): Combines SMPL-X body + HaMeR hands + the object (placed at the lifted trajectory, default DA3 contact-phase) + an estimated ground plane in a single pyrender scene. Reuses the overlay renderer's SMPL-X build/coords helpers (import as a file). Produces two videos under `results/renders/full_scene_3d/`: `overlay.mp4` (semi-transparent mesh composited over the video, per-frame K — `--alpha` default 0.5, so the real person and the mesh can be compared together) and `world.mp4` (free orbiting camera, solid mesh, neutral bg + ground). The object is a **textured sphere proxy** by default; `--object-mesh foo.glb` drops in a real mesh (e.g. a SAM 3D Objects export) — note SAM 3D Objects needs 32 GB VRAM so it must be run off this 8 GB box. Trajectory is translation-only (no object rotation estimated), which is exact for a sphere. Coord note: overlay uses `_CAM_FLIP=diag([1,1,-1])` + `[::-1]`; world uses a Y-up transform `_CV_TO_WORLD=diag([1,-1,1])` with `look_at` and **no** vertical flip. Both are reflections (det −1) that invert face winding → reverse `faces[:, ::-1]` so normals point outward (otherwise pyrender culls the outer faces and the mesh looks see-through, e.g. eyes visible from behind); renders also set `RenderFlags.SKIP_CULL_FACES` as a safety net. **HOI-PAGE-style** (arXiv:2506.07209) aesthetic: solid warm **tan** body (`--body-color tan`, default), white background, clean ground plane with one soft cast shadow (single key light + ambient fill + non-shadowing point fills). The world view is the paper-style scene video; the body is opaque (not washed/transparent). Shadows need `np.infty = np.inf` shimmed at import (NumPy 2.0 removed `np.infty`, which `RenderFlags.SHADOWS_DIRECTIONAL` references) — the standalone overlay renderer still omits shadows for this reason. `--body-color paper` gives a neutral gray instead.
- **Direct renderer** (`scripts/shared/sharedcam/render_pose6d_sharedcam_direct.py`): Renders ball + human skeleton in shared camera frame
- **Full-body renderer** (`scripts/shared/human_ball/render_human_ball_fullbody_scene.py`): Renders complete interaction scenes with ball
- Supports world view and camera-projection view

Key implementation notes for `render_smplx_pyrender_overlay.py`:
- Sets `PYOPENGL_PLATFORM=egl` for headless GPU rendering
- Coordinate transform: `_CAM_FLIP = diag([1,1,-1])` (flip z only, GVHMR OpenCV → pyrender OpenGL); rendered image flipped vertically `[::-1]` (pyrender row-0 = bottom)
- No shadow render flags — `np.infty` was removed in NumPy 2.0, causing crashes with `SHADOWS_DIRECTIONAL`

### Layer 7: Scene Assembly
- **Build human-ball scene** (`scripts/shared/human_ball/build_human_ball_shared_scene.py`): Exports joint human-ball trajectory table

## Development Workflow

### Environment Setup

The project uses **conda** with two named environments (environment files are not yet committed — install dependencies manually):

- `gvhmr` — main pipeline: GVHMR, SMPL-X, tracking, events, pose6d lifting, rendering (pyrender + EGL), stitching
- `hamer` — HaMeR hand pose only; requires `torch==2.0.1+cu117` and `numpy<1.24` (chumpy compatibility)

**Key dependencies** (inferred from imports):
- **gvhmr**: torch, cv2, numpy, scipy, librosa, hydra, transformers, SAM2, CoTracker, smplx, pyrender, trimesh
- **hamer**: torch 2.0.1, `numpy<1.24` (chumpy uses deprecated `np.int`/`np.float` removed in 1.24), `pip install git+https://github.com/geopavlakos/hamer.git`
- **GVHMR** (inside `gvhmr`): hydra, pytorch3d, smplx, VitPose (see `scripts/third-party/GVHMR/`)

### Running the Full Pipeline (Basketball Example)

**Input**: `samples/basketball_01/video.mp4`

```bash
# 1. Prepare frames and audio
conda run -n gvhmr python -m scripts.manual_init.prepare_basketball_sample

# 2. Run SAM2 segmentation (--box x1,y1,x2,y2 optional; default 470,320,570,430)
conda run -n gvhmr python -m scripts.manual_init.run_sam2_basketball

# 3. Track ball points with CoTracker
conda run -n gvhmr python -m scripts.shared.tracking.run_cotracker_basketball

# 4. Detect audio peaks and align with visual bounces
conda run -n gvhmr python -m scripts.shared.events.align_basketball_events

# 5. Run GVHMR for human pose (requires GVHMR setup)
conda run -n gvhmr python scripts/run_gvhmr.py --video-path samples/basketball_01/video.mp4

# 5b. Run HaMeR for hand articulation (hamer env; requires HaMeR checkpoint)
conda run -n hamer python -m scripts.shared.hands.run_hamer_hands \
  --sample-dir samples/basketball_01 \
  --body-model-root scripts/third-party/GVHMR/inputs/checkpoints/body_models

# 5c. Stitch HaMeR fingers into GVHMR body params
conda run -n gvhmr python -m scripts.shared.hands.stitch_hands_to_body \
  --sample-dir samples/basketball_01 --anchor-mode first

# 5d. (Optional) Per-frame metric depth via Depth Anything 3, aligned to GVHMR
conda run -n gvhmr python -m scripts.shared.depth.run_depth_anything_v3 \
  --sample-dir samples/basketball_01

# 6. Run shared-camera ball baseline (3D lifting + fitting)
#    sphere size cue (default):
conda run -n gvhmr python -m scripts.shared.sharedcam.run_basketball_pose6d_sharedcam
#    or object-agnostic DA3 metric depth (writes results/pose6d_sharedcam_depthv3/):
#    NB: run as a file (bare `support_geometry` import isn't on the -m path)
conda run -n gvhmr python scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py \
  --sample-dir samples/basketball_01 --depth-source depthv3

# 7. (Optional) Contact-phase refinement on the DA3 trajectory (anchors + smoothness, no gravity)
conda run -n gvhmr python scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration_anchorinterp_generic.py \
  --sample-dir samples/basketball_01 \
  --body-model-root scripts/third-party/GVHMR/inputs/checkpoints/body_models \
  --ball-trajectory-csv samples/basketball_01/results/pose6d_sharedcam_depthv3/ball_pose6d_sharedcam_trajectory.csv \
  --support-geometry-json samples/basketball_01/results/pose6d_sharedcam_depthv3/support_geometry.json \
  --out-subdir pose6d_sharedcam_contactphase_depthv3

# 8a. Render shaded SMPL-X body overlay (primary; auto-uses stitched hands if present)
conda run -n gvhmr python scripts/shared/human_ball/render_smplx_pyrender_overlay.py \
  --sample-dir samples/basketball_01

# 8b. (Optional) Render ball + human skeleton in shared camera frame
conda run -n gvhmr python scripts/shared/sharedcam/render_pose6d_sharedcam_direct.py \
  --sample-dir samples/basketball_01 \
  --ball-csv samples/basketball_01/results/pose6d_sharedcam/ball_pose6d_sharedcam_trajectory.csv \
  --render-tag pose6d_sharedcam_direct --with-human

# 8c. Unified 3D scene: body + HaMeR hands + object + ground (overlay AND world orbit).
#     Run as a file (imports the sibling overlay module). Auto-picks DA3 contact-phase trajectory.
conda run -n gvhmr python scripts/shared/human_ball/render_full_scene_3d.py \
  --sample-dir samples/basketball_01 --mode both
#     Paper-style free-camera scene video only (white bg, ground shadow, solid tan mesh):
conda run -n gvhmr python scripts/shared/human_ball/render_full_scene_3d.py \
  --sample-dir samples/basketball_01 --mode world
#     Drop in a real object mesh later (e.g. SAM 3D Objects export):
#     ... render_full_scene_3d.py --sample-dir samples/basketball_01 --object-mesh path/to/object.glb
```

Each step is independent (outputs feed into next step); see `samples/basketball_01/README.md` for detailed command variations.

### Key Input/Output Contracts

**Sample directory structure** (enforced by all scripts):
```
samples/<name>/
  video.mp4
  frames/                          # PNG frames at 24 fps
  audio.wav                        # 16 kHz mono (optional)
  results/
    tracking/                      # Ball trajectory CSV
    events/                        # Audio/visual event alignment
    gvhmr/                         # Human pose from GVHMR
    pose6d_sharedcam/              # 3D ball trajectory (baseline)
    pose6d_sharedcam_contactphase/ # 3D ball trajectory (contact-refined)
    segmentation/                  # SAM2 masks
    renders/                       # Video/image renders
```

**Core CSV formats**:
- `ball_trajectory.csv`: frame, time, ball_center_x, ball_center_y, radius, source
- `ball_pose6d_sharedcam_trajectory.csv`: frame, time, tx, ty, tz, radius_m, + raw estimates
- `audio_events.csv`, `visual_events.csv`: frame/time and event type

## Known Object Initialization (Generalization Path)

For objects beyond basketball:

```bash
# 1. Prepare samples from video_sample/prompts.json (object name → detector)
conda run -n gvhmr python -m scripts.known_object_init.prepare_known_object_samples

# 2. Run Grounding DINO + SAM2 for auto-initialization
conda run -n gvhmr python scripts/known_object_init/run_grounded_sam2_sample.py \
  --sample-name <name> --object-name <object_label>
```

Uses Hugging Face transformers (Grounding DINO) for zero-shot detection + SAM2 for segmentation.

## Third-Party Dependencies

### GVHMR Integration
Located in `scripts/third-party/GVHMR/` (cloned submodule):
- Self-contained human motion recovery in gravity-view coordinates
- Entry point: `run_gvhmr.py` wraps GVHMR's Hydra-based inference
- Body models at: `scripts/third-party/GVHMR/inputs/checkpoints/body_models/` (smplx uses `body_models` basename — wrong path gives misleading "Unknown model type body" error)
- See `scripts/third-party/GVHMR/README.md` for GVHMR-specific setup and configuration

Setup:
```bash
bash scripts/setup_third_party.sh
# Then follow GVHMR installation docs
```

### HaMeR Integration
Located in `scripts/third-party/hamer/` (cloned):
- Checkpoint: `scripts/third-party/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`
- HaMeR uses relative `_DATA/` paths internally — `run_hamer_hands.py` must `os.chdir` to the hamer source dir before calling `load_hamer()`, then restore cwd
- `ViTDetDataset(right=...)` expects `np.array([float(is_right)])`, not a Python bool
- Use `DataLoader(dataset, batch_size=1, num_workers=0)` for batching — `dataset[0]` returns numpy arrays that don't have `.unsqueeze()`
- Requires `numpy<1.24` in the `hamer` env (chumpy uses deprecated `np.int`/`np.float` removed in 1.24)

## Design Principles & Current Limitations

### Monocular 3D Lifting Strategy
The current basketball baseline uses **size-based monocular depth** (known ball radius → apparent radius → Z):
- Simple but sensitive to mask radius estimation errors
- Depends on ground-contact assumptions for Y estimation
- Focal length is approximated (0.9 × image width) rather than calibrated

### Event-Centric Generalization
Rather than building per-object pipelines, the roadmap proposes:
1. Keep object initialization flexible (manual, known-category, or learned)
2. Standardize object state output (mask, bbox, keypoints, contact candidates, pose proxy)
3. Define reusable event types and their constraints
4. Jointly optimize 3D reconstruction over all constraints + audio timing

See `object_generalization_pipeline_en.md` for the full generalization strategy (Sections 1–10).

## Code Organization Notes

- **Modular by stage**: Initialization → Tracking → Events → 3D Lifting → Rendering
- **CSV-based data flow**: Scripts read/write standard CSV formats for traceability and post-hoc analysis
- **Conda-separated environments**: `gvhmr` (main pipeline + rendering) vs. `hamer` (hand pose only; pinned numpy<1.24)
- **Minimal assumptions**: Baseline uses basic pinhole camera, fixed sphere shape, no learned priors (yet)
- **Scripts vs. GVHMR**: AudioHOI scripts are lightweight; GVHMR is external, self-contained reference implementation

## Common Development Patterns

- **Argument parsing**: All scripts use `argparse` with `--sample-dir`, `--output-dir` conventions
- **Hydra config**: GVHMR uses Hydra; main scripts use argparse (simpler for prototyping)
- **Visualization**: Uses `matplotlib` (with `Agg` backend for headless rendering) + OpenCV for video I/O
- **Reprojection validation**: Most 3D scripts generate reprojection CSVs and PNGs to validate lifting quality
