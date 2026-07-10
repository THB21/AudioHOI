# Final Result Evaluation Folder

Current reporting files:

- `final_evaluation_human_readable.md`: six-column human-readable final HOI table.
- `final_evaluation_detailed.csv`: detailed hard metrics for the current final result of each case.
- `final_evaluation_summary_manifest.json`: manifest for the current unified final HOI evaluator.

Current command:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py \
  --result-name benchmark_vlm_qwen \
  --output-dir samples_known_object/final_result_evaluation
```

Legacy object-only files:

- `final_result_evaluation_summary.md`
- `final_result_evaluation_summary.csv`
- `final_result_evaluation_summary.html`
- `final_result_evaluation_summary_manifest.json`

The legacy files are kept for compatibility only. Do not use them as the current final HOI reporting table.
