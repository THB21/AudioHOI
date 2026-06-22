<h2 align="center" width="100%">
AudioHOI: Audio-Conditioned 4D Human-Object Interaction Reconstruction
</h2>

<div>
<div align="center">
    NAMES<sup>1</sup>&emsp;
</div>

<div>
<div align="center">
    <sup>1</sup>Technical University of Munich<br>
</div>

---

Audio-conditioned 4D human-object interaction reconstruction from monocular video.
The pipeline reconstructs the human (GVHMR body + HaMeR hands) and the object together
in one shared, metric camera frame, using audio and visual signals to constrain the
interaction. It is moving toward a general, **zero-shot** framework that works on any
object rather than per-object pipelines.

## Pipeline at a glance

`video → object init (SAM2 / Grounding-DINO) → tracking (CoTracker) → events (audio + visual)
→ human (GVHMR + HaMeR) → metric depth (Depth Anything 3, scaled to GVHMR) → 3D lifting
→ contact-phase refinement → unified 3D scene render`

Object depth is **object-agnostic**: instead of assuming a known sphere radius, we read
per-frame metric depth from Depth Anything 3 and scale it to the GVHMR body. The object
is lifted by a least-squares energy of data terms (mask, keypoints, depth, contact) with a
single smoothness regularizer — no physics priors.

## Environments

Two conda envs (install deps manually; see `CLAUDE.md`):

```bash
# main pipeline: GVHMR, SMPL-X, tracking, events, depth (DA3), lifting, rendering
conda create -n gvhmr python=3.10
# hand pose only (HaMeR; needs numpy<1.24)
conda create -n hamer python=3.10
```

Third-party (GVHMR, HaMeR, Depth Anything 3, SAM 3D Objects) are cloned/installed via
`scripts/setup_third_party.sh`.

## Docs

- `CLAUDE.md` — full architecture, per-stage scripts, commands, and gotchas.
- `method_losses.md` — mathematical overview of the depth alignment, lifting, and
  contact-phase losses, plus the generalized zero-shot energy and the VLM agentic layer.
- `object_generalization_pipeline_en.md` — the generalization roadmap.
- `samples/basketball_01/README.md` — end-to-end commands on the basketball example.
