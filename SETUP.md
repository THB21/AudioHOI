# Setup on a fresh host

Notes for getting AudioHOI running from scratch on a new machine. The heavy stuff
(model weights, third-party clones) is deliberately not in git, so this is the part
you have to redo by hand.

## 1. Clone and branch

```bash
git clone git@github.com:THB21/AudioHOI.git
cd AudioHOI
git checkout tom      # working branch
```

## 2. Conda environments

Two envs, kept separate because HaMeR pins an old numpy/torch that fights with the
main stack.

```bash
# main pipeline: GVHMR, SMPL-X, SAM2, CoTracker, Depth Anything 3, pyrender, audio
conda create -n gvhmr python=3.10 -y
conda activate gvhmr
# torch (CUDA build for your GPU), then:
pip install opencv-python numpy scipy librosa hydra-core transformers smplx \
            pyrender trimesh matplotlib

# hand pose only
conda create -n hamer python=3.10 -y
conda activate hamer
pip install "torch==2.0.1" "numpy<1.24"
pip install git+https://github.com/geopavlakos/hamer.git
```

Rendering is headless via EGL: the renderers set `PYOPENGL_PLATFORM=egl` themselves,
so a GPU with EGL is enough (no display needed).

## 3. Third-party code and weights (not in git)

```bash
bash scripts/setup_third_party.sh      # clones GVHMR / HaMeR under scripts/third-party/
```

Then drop the checkpoints in place — these are big and live outside the repo:

- GVHMR body models: `scripts/third-party/GVHMR/inputs/checkpoints/body_models/`
  (the smplx basename must be `body_models`, or you get a misleading
  "Unknown model type body" error)
- HaMeR checkpoint: `scripts/third-party/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`
- Depth Anything 3 and SAM2 weights per their upstream instructions

`models/` and `third-party/` are gitignored; nothing in there is transferred by git.

## 4. Run something

Single-object basketball baseline (the original path, documented step by step in
`samples/basketball_01/README.md` and `CLAUDE.md`):

```bash
conda run -n gvhmr python -m scripts.manual_init.prepare_basketball_sample
conda run -n gvhmr python -m scripts.manual_init.run_sam2_basketball
conda run -n gvhmr python -m scripts.shared.tracking.run_cotracker_basketball
# ... GVHMR, HaMeR, depth, lifting, render — see CLAUDE.md
```

Generic multi-object pipeline (the current mainline — one entry point, fixed stage
chain, see `docs/current_generic_pipeline_mainline.md`):

```bash
conda run -n gvhmr python scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case basketball
```

Cross-case reporting:

```bash
python scripts/shared/generic_contact_pipeline/tools/run_final_summary.py
```

## Layout

```
scripts/manual_init/            basketball frame + SAM2 init
scripts/known_object_init/      Grounding-DINO + SAM2 for arbitrary objects
scripts/shared/                 tracking, events, depth, lifting, rendering
scripts/shared/generic_contact_pipeline/
                                current mainline: core / components / stages / tools
scripts/third-party/            GVHMR, HaMeR (cloned, not in git)
src/audio/                      audio -> event -> semantic contact pipeline
samples/                        single-object samples (basketball, football, hammer...)
samples_known_object/           multi-object generic-pipeline cases
assets/object_meshes/           object proxy meshes (.glb)
docs/                           method + evaluation notes
```

Deeper detail on every stage, the coordinate conventions, and the known gotchas lives
in `CLAUDE.md`.
