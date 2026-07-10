# Generic Pipeline Tools

Primary user-facing tools:

- `run_final_hoi_evaluator.py`: writes the current HOI final evaluation summary, including object 6DoF, hard overlay, contact/anchor, physical, temporal, and QA artifacts.
- `run_ablation_evaluation.py`: compares real materialized method variants such as `no_audio`, `no_vlm`, `no_llm`, and `no_contact_anchor`.
- `run_final_summary.py`: legacy object-only summary. Keep it for compatibility, but do not use it as the main final result table.
- `run_final_evaluator.py`: evaluates one result directory with hard metrics, VLM visual judge artifacts, and LLM CSV audit.
- `run_benchmark.py`: compares multiple real method variants. Use this only for method comparison, not for reporting a single final result.

Preprocess and inspection helpers:

- `run_sam2_object.py`
- `run_cotracker_object_points.py`
- `render_contact_candidates.py`
- `export_vlm_qa_report.py`
- `evaluate_overlay_quality.py`
- `evaluate_visible_line_overlay.py`

The final result report should use `run_final_hoi_evaluator.py`. The benchmark report should not be used as the primary final-result table.

Current final-result command:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py \
  --result-name benchmark_vlm_qwen \
  --output-dir samples_known_object/final_result_evaluation
```

Equivalent post-run pipeline flag:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case stick \
  --result-name benchmark_vlm_qwen \
  --run-final-evaluator
```

Current ablation command:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py \
  --output-dir samples_known_object/ablation_evaluation
```

Equivalent post-run pipeline flag:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case all \
  --run-ablation-evaluation
```
