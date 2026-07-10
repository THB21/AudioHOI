# Ablation Evaluation Folder

This folder contains method-variant evaluation over real materialized result directories. The current default variants are:

- `full_audio_vlm_llm`
- `audio_enabled`
- `no_audio`
- `no_vlm`
- `no_llm`
- `no_contact_anchor`

Current command:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_ablation_evaluation.py \
  --output-dir samples_known_object/ablation_evaluation
```

Important files:

- `ablation_table.csv`: per-case, per-method metrics and metadata.
- `ablation_delta_table.csv`: metric deltas against `full_audio_vlm_llm`.
- `ablation_report.md`: human-readable audit showing audio/VLM/LLM flags, result directory, pose hash equality, and metric equality.
- `ablation_evaluation_manifest.json`: row counts and missing-result count.

The current run has real result directories for all default variants. `missing_results` should be `0`.
