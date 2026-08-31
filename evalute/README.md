# Evalute Folder

This folder contains the next final evaluation design for AudioHOI. The spelling follows the requested folder name `evalute`.

Files:

- `final_evaluation_design_cn.md`: Chinese final evaluation protocol.
- `final_evaluation_design_en.md`: English final evaluation protocol.
- `code_implementation_plan.md`: concrete code/module implementation plan.
- `ablation_design.md`: ablation benchmark design.
- `runtime_environment_inventory.md`: checked Python environments and which runtime should run each pipeline/evaluation tool.

Design principles:

1. Hard metrics first; VLM is not the primary overlay evaluator.
2. Contact, penetration, and floating must be co-reported because they trade off.
3. Evaluation follows the meeting guidance: evaluate every loss/constraint that the method claims to use.
4. HOI-PAGE is used as part-level inspiration, but AudioHOI adds video-conditioned overlay and audio-aware temporal metrics.

Implemented first pass:

- `scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/`
- `scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py`
- `scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py`

Current implemented scope:

1. Merge object SE3 metrics, hard overlay metrics, HOI contact/penetration metrics, and audio-aware temporal fields into `evaluation/final_evaluation_summary.json`.
2. Write cross-case `final_result/evaluation/final_evaluation_detailed.csv`.
   - current complete final-result scope is basketball and football only;
   - mug/chair/stick are excluded from the final-result table until their final videos and paired object/human/contact/gate artifacts are complete and frame-aligned.
3. Enforce ablation method/result-directory mapping and write `final_result/evaluation/ablation/ablation_table.csv`.
   - ablation currently follows the same complete final-result scope: basketball and football only.
   - current default methods map to materialized `benchmark_*` result directories;
   - `ablation_table.csv` includes `pose_sha256`, `same_pose_as_baseline`, and `metrics_identical_to_baseline`;
   - `ablation_report.md` explains when a variant changes pose but not selected final metrics.
4. Compute overlay by mask-pair IoU when observed masks and rendered masks exist.
5. If rendered masks are missing, generate evaluation-only render masks under `evaluation/render_masks/`, then compute IoU/coverage/false coverage.
   - sphere/ball cases use proxy circle masks from pose + observation radius.
   - URDF/Articraft cases use full geometry masks when a parseable URDF exists.
   - source is explicitly reported as `generated_eval_proxy_render_mask_iou` or `generated_eval_full_geometry_mask_iou`.
6. Export part-level contact artifacts:
   - `evaluation/hoi_contact_pairs.csv`
   - `evaluation/hoi_contact_intervals.csv`
   - `evaluation/hoi_contact_metrics.csv`
   These are built from `object_contact_points.csv` / `contact_candidates.csv` plus `anchor_state.csv`, with `hoi_interaction_metrics.json` kept as the aggregate summary. `contact_proxy` is derived from `contact_gap_mm` as `exp(-contact_gap_mm / 50)`, so higher is better and the raw gap remains visible.
7. Export part-coverage artifacts:
   - `evaluation/human_parts.csv`
   - `evaluation/object_parts.csv`
   - `evaluation/object_part_vocab_map.csv`
   - `evaluation/part_metrics.csv`
   These expose which human/object parts have evidence. Object parts are reported with canonical names for metrics and preserve raw config/contact/surface names for audit.
8. Export final VLM/LLM QA audit artifacts under `vlm_trace/06_evaluation/` whenever `run_final_hoi_evaluator.py` runs:
   - `pipeline_qa_summary.csv`
   - `pipeline_qa_summary.json`
   - `pipeline_qa_summary.md`
   - `vlm_eval_queries.csv`
   - `vlm_eval_raw_responses.jsonl`
   - `vlm_eval_parsed_scores.csv`
   - `vlm_eval_summary.json`
   - `llm_eval_summary.md`
   - `qa_audit_report.html`
   These artifacts currently use the existing trace/evidence and a metric-grounded dry-run final judge unless a real Qwen final judge is explicitly run. They are audit outputs only and do not generate or overwrite pose.
9. Export motion-regime-aware temporal metrics:
   - `evaluation/temporal_plausibility_metrics.csv`
   - `evaluation/temporal_plausibility_metrics.json`
   These split coarse jump checking into translation spikes, rotation spikes, event-aligned spikes, non-event spikes, high-speed recall, oversmooth rate, static-tail drift, and temporal failure intervals.
10. Export gate-impact metrics:
   - `evaluation/gate_impact_metrics.csv`
   - `evaluation/gate_impact_metrics.json`
   These record whether gates changed residuals, anchor update permission, freeze/interpolation, feedback reweighting, and pre-smooth to final pose deltas. This is the ablation evidence for whether VLM/LLM/audio gates actually affected optimization.

Still planned:

1. Replace remaining ball proxy masks with true sphere rasterization from 3D radius/camera when calibrated render masks are available.
2. Add observed-vs-stable local drift for all object families, not only stick/local-s contacts.
3. Real Qwen final judge over selected evidence panels.
4. Clean-room `eval_*` ablation reruns if we need results separate from the currently materialized `benchmark_*` directories.
5. Add true paired full/no-audio/no-VLM/no-LLM final-result variants so gate impact and final hard metrics can both answer contribution size.

## Object Part Vocabulary Normalization

The evaluator now separates metric vocabulary from raw evidence vocabulary.

- `object_parts.csv`
  - `part`: canonical object part used by metrics.
  - `raw_parts`: raw names that were merged into that canonical part.
  - `surface_point_rows`: evidence count from object surface/semantic points.
  - `contact_evidence_rows`: observed contact evidence count after canonicalization.
- `object_part_vocab_map.csv`
  - one row per raw part name;
  - records `raw_part -> canonical_part`;
  - records whether the raw name came from case config, surface points, or HOI contact pairs.

Examples:

| Raw names | Canonical metric part |
| --- | --- |
| `handle`, `handle_loop` | `handle` |
| `cup_body`, `body_shell`, `body_shell_or_occluded_handle_region` | `cup_body` |
| `rim`, `rim_ring` | `rim` |
| `front_leg`, `rear_leg` | `legs` |
| `backrest`, `backrest_board` | `back` |
| `main_body` | `shaft` |
| `ball_boundary` | `surface` |
| `ball_bottom`, `floor_support` | `support_region` |

This matters because `object_part_contact_coverage` is not a string-match score. It asks whether the semantic part slots relevant to the object family have evidence after alias normalization.

## VLM/LLM QA Audit Artifacts

The final evaluator writes a QA layer in addition to hard metrics. This answers "what did VLM/LLM inspect?" without letting VLM/LLM directly create pose.

| File | Meaning |
| --- | --- |
| `pipeline_qa_summary.csv/json/md` | aggregated pipeline-stage VLM/LLM questions, answers, gates, affected constraints, and optimizer-effect flags |
| `vlm_eval_queries.csv` | representative final-evaluation questions with frame/window and evidence path |
| `vlm_eval_raw_responses.jsonl` | raw answer payload for each query |
| `vlm_eval_parsed_scores.csv` | parsed scores, failure stage hint, affected constraint, and whether optimizer behavior changed |
| `vlm_eval_summary.json` | aggregate QA score and failure-stage summary |
| `llm_eval_summary.md` | CSV/metric audit summary for gate, anchor, smooth, and static-tail consistency |
| `qa_audit_report.html` | human-readable QA report linking question, answer, evidence, and affected constraint |

Current mode:

```text
source = metric-grounded dry-run final judge
changed_optimizer_behavior = 0
```

That means the QA layer explains and audits the final result. It does not rerun optimization and does not replace hard metrics. A real Qwen final judge can replace the dry-run answer source later while preserving the same artifact schema.
