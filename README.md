# AudioHOI

Audio-conditioned 4D human-object interaction reconstruction from monocular video.

This repository combines object reconstruction, human reconstruction, audio
processing, contact refinement, HOI evaluation, and full-scene rendering. The
intended workflow is:

```text
video + case config
  -> generic object SE3 pipeline
  -> VLM/LLM audit trace and final object evaluation
  -> human/audio/contact refinement layer
  -> HOI interaction evaluation and full-scene render
```

The object pipeline produces the object trajectory. The human layer reads that
trajectory as input; it does not replace the object solver.

## Integrated Repository

- Canonical branch: `main`
- Base object pipeline: `vlm-gated-mainline`
- Integrated components: human modeling, audio events, body-side contact
  refinement, HOI metrics, and final full-scene renders.

The latest known final object result name is:

```text
benchmark_vlm_qwen
```

## Versioned Results

The compact nine-case trajectories, unified metrics, Audio/VLM ablations,
object-factor removals, and provenance manifests are published under
results_release/. Validate them from the repository root with:

    python scripts/release/validate_results_release.py

Large input and inspection videos are kept out of the compact numeric bundle
and are distributed through the results-v1 GitHub release:

https://github.com/THB21/AudioHOI/releases/tag/results-v1

Paper LaTeX, PDFs, BibTeX, and paper figures are maintained separately and are
not part of this code repository.

## Runtime Environment Map

Do not rely on the currently activated shell Python for pipeline work. Use the
repo runtime map in:

```text
scripts/shared/generic_contact_pipeline/configs/runtime_envs.yaml
```

The checked package inventory is documented in:

```text
evalute/runtime_environment_inventory.md
```

Short version:

| Task | Environment | Python |
| --- | --- | --- |
| Generic object pipeline, final evaluator, ablation evaluator | `audiohoi` | `/home/yang/miniconda3/envs/audiohoi/bin/python` |
| Local Qwen-VL stage/final visual judge | `qwen_vl` | `/home/yang/miniconda3/envs/qwen-vl/bin/python` |
| DA3 depth and point-cloud preprocessing | `da3` | `/home/yang/miniconda3/envs/da3/bin/python` |
| GVHMR body reconstruction | `gvhmr` | `/home/yang/miniconda3/envs/gvhmr/bin/python` |
| HaMeR hand reconstruction and pyrender diagnostics | `hamer` | `/home/yang/miniconda3/envs/hamer/bin/python` |
| Optional body render experiments | `bodyrender` | `/home/yang/miniconda3/envs/bodyrender/bin/python` |
| Articraft wrapper / asset generation check | `articraft` | `/home/yang/miniconda3/envs/articraft-py312/bin/python` |
| Optional SAM3D object experiments | `sam3d_objects` | `/home/yang/miniconda3/envs/sam3d-objects-inference/bin/python` |

## Main Pipeline Overview

```text
Stage -1  LLM semantic prior
Stage 0   preprocess manifest
Stage 1   generic object observation
Stage 2   generic contact / anchor state
Stage 3   generic SE3 pose init
Stage 4   generic sequence SE3 optimizer
Stage 5   object render
Stage 6   compare
Stage 6.5 LLM CSV audit
Stage 7   loss / residual analysis
             |
             v
Human layer: GVHMR + HaMeR + audio/VLM contact records
             |
             v
Body-side contact refinement + HOI metrics + full-scene render
```

The only intended object-pipeline entrypoint is:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case stick \
  --result-name benchmark_vlm_qwen \
  --from-stage stage-1 \
  --to-stage stage7 \
  --llm-mode mistral \
  --vlm-mode qwen
```

Use `--vlm-limit 0` for full Qwen evaluation. A nonzero limit is debug only.

## Stage Contracts

### Stage -1: LLM Semantic Prior

Input:

- case YAML from `scripts/shared/generic_contact_pipeline/configs/cases/`
- sample metadata from `samples_known_object/<case>/metadata.json`
- optional prompt/profile seed files

Output:

- `hoi_profile.json`
- `hoi_profile_resolved.yaml`
- `prompt_context.json`
- `llm_prior_trace.json`

Role:

- LLM writes discrete semantic priors only.
- It does not output continuous pose, coordinates, or object trajectory.

### Stage 0: Preprocess Manifest

Input:

- reusable preprocessing artifacts: segmentation, tracking, DA3 depth, GVHMR,
  audio/events when available

Output:

- `stage0_inputs_manifest.json`
- `stage0_metrics.json`

Role:

- Records which inputs are generated, reused, missing, or failed.
- It does not solve object pose.

### Stage 1: Generic Object Observation

Input:

- SAM2/object masks, CoTracker/object points, depth, case geometry config

Output:

- `object_observations.csv`
- `object_correspondence.csv`
- `object_surface_points.csv`
- `object_semantic_points.csv`
- optional line-object artifacts:
  - `line_observations.csv`
  - `line_correspondence.csv`

Role:

- Normalizes every object into the shared observation contract.
- For line-like objects, visual line evidence is not treated as the physical
  object by itself; correspondence and geometry constraints are carried forward.

### Stage 2: Generic Contact / Anchor

Input:

- object observations/correspondence
- human pose/keypoints when available
- VLM gate evidence when running with Qwen

Output:

- `contact_candidates.csv`
- `anchor_state.csv`
- `object_contact_points.csv`
- `stage2_metrics.json`

Role:

- Converts hand/foot/body/floor/table proximity into a shared contact schema.
- Separates:
  - `contact_observed`
  - `contact_persistent`
  - `anchor_update_allowed`
  - `pose_anchor_allowed`
  - stable object-local anchor coordinates

### Stage 3: Generic SE3 Pose Init

Input:

- observations
- correspondence
- contact/anchor hints
- geometry adapter

Output:

- `object_pose_init.csv`
- `stage3_metrics.json`

Required schema:

```text
frame,time,tx,ty,tz,qw,qx,qy,qz,...
```

All five current objects have this SE3 schema:

- basketball
- football
- mug
- chair
- stick

For symmetric objects such as balls, rotation may be weakly constrained, but
the schema is still SE3.

### Stage 4: Generic Sequence SE3 Optimizer

Input:

- `object_pose_init.csv`
- `contact_candidates.csv`
- `anchor_state.csv`
- observation/correspondence artifacts
- VLM/LLM gates and audits

Output:

- `object_pose_pre_smooth.csv`
- `object_pose.csv`
- `physical_smooth_residuals.csv`
- `pose_jump_audit.csv`
- `optimizer_decisions.csv`
- `stage4_metrics.json`

Role:

- This is the final object trajectory solver.
- It applies visual, anchor, depth, geometry, velocity, acceleration,
  static/freeze, and gate-controlled residuals.
- Old policy files are not independent pipeline branches. They are compatibility
  seed/residual builders inside the mainline.

Important current implementation detail:

- `scripts/shared/generic_contact_pipeline/stages/main/stage4_contact_refine.py`
  calls `components/mainline/sequence_refine.py`.
- `sequence_refine.py` may run `generic_line_physical_smooth` as the official
  line-object seed/visual prior, then the output still goes through the shared
  SE3 sequence optimizer.

### Stage 5: Render

Input:

- final `object_pose.csv`
- object geometry asset/proxy/URDF/mesh
- human projection when available

Output:

- object-only render videos
- with-human render videos
- render manifest

Generic render code reads the SE3 quaternion schema.

### Stage 6 / 6.5 / 7: Evaluation and Audit

Input:

- stage artifacts
- renders
- VLM traces
- residual/audit CSV files

Output:

- `stage6_compare_report.json`
- `llm_csv_audit_*`
- `loss_analysis/*`
- `pipeline_manifest.json`
- `vlm_trace/00_input` through `vlm_trace/06_evaluation`

## VLM / LLM Audit Flow

VLM and LLM do not directly generate pose. They inspect evidence and produce
gates/audits.

Pipeline flow:

```text
stage evidence
  -> VLM/LLM question
  -> raw response
  -> parsed label/score
  -> gate or audit row
  -> residual weight / anchor update / freeze decision
```

Standard trace outputs:

- `vlm/stage*/vlm_queries.csv`
- `vlm/stage*/vlm_results.csv`
- `vlm/stage*/vlm_gates.csv`
- `stage_audit/stage*/llm_audit_results.csv`
- `stage_audit/stage*/stage_audit_gates.csv`
- `vlm_trace/04_gating/gate_timeline.csv`
- `vlm_trace/06_evaluation/evaluation_summary.json`
- `vlm_trace/06_evaluation/qa_audit_report.html`

`pipeline_manifest.json` now records both:

- the launch-time `vlm_mode` / `llm_mode`
- the actual VLM/stage-audit artifacts found on disk

This matters because some historical result directories were repaired or
re-exported after the original run; the actual artifact summary is the reliable
source for whether VLM/LLM evidence exists.

## Human / Audio / HOI Layer

This section documents the inputs and outputs used for human reconstruction,
audio processing, contact refinement, HOI evaluation, and full-scene rendering.

Input:

- `samples_known_object/<case>/results/benchmark_vlm_qwen/object_pose.csv`
- GVHMR body parameters
- HaMeR hand parameters
- audio events / contact records
- object geometry as sphere, capsule, or mesh/SDF

Output:

- human parameters:
  - `gvhmr/result.pkl` or `human_gvhmr/result.pkl`
  - `hands/stitched_smplx_params.pkl` or
    `human_hands/stitched_smplx_params.pkl`
  - `contact_refine/contact_refined_smplx_params.pkl` or
    `human_contact_refine/...`
- HOI metrics:
  - `results/hoi_eval/hoi_interaction_metrics.json`
  - cross-case table under `samples_known_object/hoi_interaction_evaluation/`
- full-scene renders:
  - `renders/*human_full_scene_3d*/*.mp4`

Main scripts:

- `scripts/shared/human_ball/contact/refine_body_pose_contact.py`
- `scripts/shared/human_ball/contact/object_geometry.py`
- `scripts/shared/human_ball/render_full_scene_3d.py`
- `scripts/shared/evaluation/compute_hoi_interaction_metrics.py`
- `src/audio/*`

The human renderer supports:

- generic SE3 pose CSV:
  `frame,tx,ty,tz,qw,qx,qy,qz`
- legacy mug pose CSV:
  `frame,tx,ty,tz,yaw,pitch,roll,scale`

## Final Results

Curated final artifacts:

- `final_result/SUMMARY.md`
- `final_result/images/`
- `final_result/videos/`

Per-case object result:

```text
samples_known_object/<case>/results/benchmark_vlm_qwen/
```

Per-case human/HOI notes:

```text
samples_known_object/<case>/results/HUMAN_RESULTS_README.md
```

Canonical current unified final HOI evaluation:

```text
samples_known_object/final_result_evaluation/final_evaluation_human_readable.md
samples_known_object/final_result_evaluation/final_evaluation_detailed.csv
samples_known_object/final_result_evaluation/final_evaluation_summary_manifest.json
```

Legacy object-only `final_result_evaluation_summary.*` files are not part of
the current final result folder. They can be regenerated from
`run_final_summary.py` only for compatibility, but the current report should use
the HOI files above.

HOI-level interaction summary:

```text
samples_known_object/hoi_interaction_evaluation/hoi_summary.md
samples_known_object/hoi_interaction_evaluation/hoi_summary.csv
```

## Evaluation Layers

### Object-Level Evaluation

Code:

```text
scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/
scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py
scripts/shared/generic_contact_pipeline/core/evaluation/final_evaluator.py
scripts/shared/generic_contact_pipeline/core/evaluation/final_summary.py  # legacy object-only summary
```

Measures:

- SE3 schema completeness
- overlay proxy / mask or line confidence
- contact proxy
- anchor drift
- penetration/floating proxy from object pipeline residuals
- jump count
- static-tail drift
- VLM judge
- LLM CSV audit

Generate the final-only table:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py
```

This default evaluates the frame-aligned deliverables declared in
`final_result/evaluation_manifest.json` and writes `final_result/evaluation/`.
Use `--source pipeline-result --result-name <name>` only for historical pipeline
regression results.

Pipeline post-run flags:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case stick \
  --result-name benchmark_vlm_qwen \
  --run-final-evaluator
```

For materialized method variants, use:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case all \
  --run-ablation-evaluation
```

`--run-benchmark` is legacy and should not be used as the main final-result or ablation table.

### HOI-Level Evaluation

Code:

```text
scripts/shared/evaluation/compute_hoi_interaction_metrics.py
```

Measures:

- human-object penetration frame ratio
- penetration depth mean/max
- non-collision ratio
- contact frame ratio
- contact gap
- part correctness
- audio-window contact ratio
- object jerk
- grasp stability

Method docs:

```text
docs/final_result_evaluation_method_cn.md
docs/final_result_evaluation_method_en.md
docs/hoi_interaction_evaluation_method_en.md
docs/research_hoi_eval_metrics_sota.md
```

## Current Nine Cases

| Public case | Pipeline case | Sample directory | Frozen audio ablation | World comparison |
| --- | --- | --- | --- | --- |
| basketball | `basketball` | `01_basketball` | yes | yes |
| football | `football` | `10_football` | yes | yes |
| mug | `mug` | `02_mug` | yes | yes |
| chair | `chair` | `05_chair` | yes | yes |
| stick | `stick` | `11_stick` | yes | yes |
| back-view basketball | `back_view_basketball` | `12_back_view_basketball` | yes | yes |
| volleyball | `volleyball` | `13_volleyball` | yes | yes |
| ping-pong | `pingpong_wall` | `14_pingpong_wall` | yes | yes |
| suitcase | `suitcase_drag` | `15_suitcase_drag` | yes | yes |

## Nine-Case Audio Ablation

The controlled comparison tests whether audio improves a visual and VLM-guided
4D HOI reconstruction. Both arms use the same video, VLM configuration, human
parameters, object model, renderer, and camera. The `vlm` arm disables audio
events before contact extraction. The `vlm_audio` arm additionally permits
VLM-verified audio events to influence contact timing and the reconstructed
object trajectory.

The compact, frame-aligned trajectories used for the released renders are in:

```text
final_result/nine_case_audio_ablation_inputs/<case>/vlm.csv
final_result/nine_case_audio_ablation_inputs/<case>/vlm_audio.csv
```

These frozen inputs make the released result videos reproducible without
committing extracted frame caches or model checkpoints. A case may have
identical accepted trajectories in both arms. Such a zero delta is retained as
a valid ablation result.

After activating a Python environment with the GVHMR rendering dependencies,
generate all nine world-view pairs with:

```bash
python scripts/shared/evaluation/run_nine_visual_vlm_audio_ablation.py \
  --mode both \
  --world-only
```

Use `--python /path/to/python` when the subprocess environment differs from the
active interpreter. The required licensed SMPL-X body model can be installed
with:

```bash
bash scripts/setup_body_models.sh
```

To render selected cases only:

```bash
python scripts/shared/evaluation/run_nine_visual_vlm_audio_ablation.py \
  --mode both \
  --world-only \
  --cases basketball,mug,pingpong
```

To rebuild the comparison videos and manifest from existing renders:

```bash
python scripts/shared/evaluation/run_nine_visual_vlm_audio_ablation.py \
  --mode both \
  --world-only \
  --resume
```

The release contains three videos per case:

```text
deliverables/nine_case_visual_vlm_audio_ablation/world_results/<case>/vlm/world.mp4
deliverables/nine_case_visual_vlm_audio_ablation/world_results/<case>/vlm_audio/world.mp4
deliverables/nine_case_visual_vlm_audio_ablation/world_results/<case>/comparison/world_vlm_vs_vlm_audio.mp4
```

This gives 18 unary method renders and 9 labelled comparisons. The clean world
renderer uses the stitched SMPL-X and HaMeR parameters, reconstructed object
trajectory, object geometry, a ground plane, and an orbiting camera. It does
not add contact markers, colored hands, camera overlays, or an audio HUD.

### Unary VLM Study

The supplemental perceptual evaluation scores each of the 18 method renders
independently from 1 to 5. The judge is blind to the method and receives no
audio label. It scores contact timing, contact location, object motion,
physical plausibility, temporal smoothness, interaction realism, and overall
quality from the complete orbiting video.

Install the Gemini client and run the evaluation with:

```bash
python -m pip install google-genai
GEMINI_API_KEY=<key> \
python scripts/shared/evaluation/run_nine_world_gemini_unary.py --resume
```

The committed scores and paired audio deltas are under:

```text
deliverables/nine_case_visual_vlm_audio_ablation/unary_vlm_evaluation/
```

Across the nine scenes, the mean unary overall-quality score is 2.33 for VLM
and 2.67 for VLM with audio. This is a descriptive result from the committed
Gemini 3.6 Flash run rather than a statistical significance claim.

This VLM study is supplemental. The canonical geometric, contact, temporal,
penetration, and audio-window metrics remain those produced by the official
evaluation code under `scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/`.

## Verification

Fast code/schema checks:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python -m compileall -q \
  scripts/shared/generic_contact_pipeline \
  scripts/shared/evaluation \
  scripts/shared/human_ball \
  src/audio

PYTHONDONTWRITEBYTECODE=1 /home/yang/miniconda3/bin/pytest -q \
  tests/test_final_hoi_evaluation.py \
  tests/test_vlm_trace_final_evaluator_benchmark.py
```

Check that final videos are H.264:

```bash
python - <<'PY'
from pathlib import Path
import subprocess, json
paths = [p for root in [Path('final_result'), Path('samples_known_object')]
         for p in root.rglob('*.mp4') if p.is_file()]
counts = {}
for p in paths:
    out = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name', '-of', 'json', str(p)
    ], text=True)
    codec = json.loads(out).get('streams', [{}])[0].get('codec_name', '')
    counts[codec] = counts.get(codec, 0) + 1
print(counts)
PY
```

Expected current result:

```text
{'h264': 210}
```
