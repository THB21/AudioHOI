# Final Result Evaluation Folder

Canonical current reporting files:

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

Legacy object-only note:

`run_final_summary.py` can regenerate `final_result_evaluation_summary.*` for
compatibility, but those files are intentionally not kept in this final result
folder. They use an older object-only metric view and should not be used as the
current final HOI reporting table.
