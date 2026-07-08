# Generic Pipeline Tools

Primary user-facing tools:

- `run_final_summary.py`: writes the final-only evaluation summary for the current final result of each case.
- `run_final_evaluator.py`: evaluates one result directory with hard metrics, VLM visual judge artifacts, and LLM CSV audit.
- `run_benchmark.py`: compares multiple real method variants. Use this only for method comparison, not for reporting a single final result.

Preprocess and inspection helpers:

- `run_sam2_object.py`
- `run_cotracker_object_points.py`
- `render_contact_candidates.py`
- `export_vlm_qa_report.py`
- `evaluate_overlay_quality.py`
- `evaluate_visible_line_overlay.py`

The final result report should use `run_final_summary.py`. The benchmark report should not be used as the primary final-result table.
