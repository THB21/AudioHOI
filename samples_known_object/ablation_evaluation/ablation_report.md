# Ablation Evaluation Report

This report compares real result directories. It does not reuse one result under multiple method labels.

## Summary

- rows: 30
- ok rows: 30
- missing rows: 0
- non-baseline rows with identical object_pose.csv hash: 4
- non-baseline rows with identical selected metrics: 9
- delta rows: 25

## How to read

- `same_pose_as_baseline=True` means the variant's `object_pose.csv` is byte-identical to `full_audio_vlm_llm` for that case.
- `metrics_identical_to_baseline=True` means the selected final metrics are identical to the baseline, even if files differ.
- If pose differs but metrics are identical, the current metrics are not sensitive to that variant or shared aggregate HOI metrics dominate the table.
- `audio`, `VLM`, `LLM`, and `flags` show the intended variant configuration; this is what prevents the table from silently reusing one result under several method labels.

## Variant audit

| case | method | status | result | audio | VLM | LLM | flags | same pose | same metrics | contact proxy | overlay | final pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basketball | full_audio_vlm_llm | ok | benchmark_vlm_qwen | True | qwen | mistral |  | True | True | 0.665 | 0.831 | True |
| basketball | audio_enabled | ok | benchmark_audio_enabled | True | qwen | mistral |  | False | True | 0.665 | 0.831 | True |
| basketball | no_audio | ok | benchmark_no_audio | False | qwen | mistral | no_audio | False | True | 0.665 | 0.831 | True |
| basketball | no_vlm | ok | benchmark_baseline_no_vlm | True | none | mistral | no_vlm | False | True | 0.665 | 0.831 | True |
| basketball | no_llm | ok | benchmark_no_llm | True | qwen | none | no_llm | False | True | 0.665 | 0.831 | True |
| basketball | no_contact_anchor | ok | benchmark_no_anchor | True | qwen | mistral | no_contact_anchor | False | True | 0.665 | 0.831 | True |
| football | full_audio_vlm_llm | ok | benchmark_vlm_qwen | True | qwen | mistral |  | True | True | 0.056 | 0.805 | True |
| football | audio_enabled | ok | benchmark_audio_enabled | True | qwen | mistral |  | True | True | 0.056 | 0.805 | True |
| football | no_audio | ok | benchmark_no_audio | False | qwen | mistral | no_audio | False | False | 0.056 | 0.805 | True |
| football | no_vlm | ok | benchmark_baseline_no_vlm | True | none | mistral | no_vlm | True | True | 0.056 | 0.805 | True |
| football | no_llm | ok | benchmark_no_llm | True | qwen | none | no_llm | True | True | 0.056 | 0.805 | True |
| football | no_contact_anchor | ok | benchmark_no_anchor | True | qwen | mistral | no_contact_anchor | True | True | 0.056 | 0.805 | True |
| mug | full_audio_vlm_llm | ok | benchmark_vlm_qwen | True | qwen | mistral |  | True | True | 0.729 | 0.451 | False |
| mug | audio_enabled | ok | benchmark_audio_enabled | True | qwen | mistral |  | False | False | 0.729 | 0.459 | False |
| mug | no_audio | ok | benchmark_no_audio | False | qwen | mistral | no_audio | False | False | 0.729 | 0.459 | False |
| mug | no_vlm | ok | benchmark_baseline_no_vlm | True | none | mistral | no_vlm | False | False | 0.729 | 0.459 | False |
| mug | no_llm | ok | benchmark_no_llm | True | qwen | none | no_llm | False | False | 0.729 | 0.459 | False |
| mug | no_contact_anchor | ok | benchmark_no_anchor | True | qwen | mistral | no_contact_anchor | False | False | 0.729 | 0.459 | False |
| chair | full_audio_vlm_llm | ok | benchmark_vlm_qwen | True | qwen | mistral |  | True | True | 0.929 | 0.172 | False |
| chair | audio_enabled | ok | benchmark_audio_enabled | True | qwen | mistral |  | False | False | 0.929 | 0.172 | False |
| chair | no_audio | ok | benchmark_no_audio | False | qwen | mistral | no_audio | False | False | 0.929 | 0.172 | False |
| chair | no_vlm | ok | benchmark_baseline_no_vlm | True | none | mistral | no_vlm | False | False | 0.929 | 0.172 | False |
| chair | no_llm | ok | benchmark_no_llm | True | qwen | none | no_llm | False | False | 0.929 | 0.172 | False |
| chair | no_contact_anchor | ok | benchmark_no_anchor | True | qwen | mistral | no_contact_anchor | False | False | 0.929 | 0.196 | False |
| stick | full_audio_vlm_llm | ok | benchmark_vlm_qwen | True | qwen | mistral |  | True | True | 0.338 | 0.226 | False |
| stick | audio_enabled | ok | benchmark_audio_enabled | True | qwen | mistral |  | False | False | 0.338 | 0.29 | False |
| stick | no_audio | ok | benchmark_no_audio | False | qwen | mistral | no_audio | False | False | 0.338 | 0.29 | False |
| stick | no_vlm | ok | benchmark_baseline_no_vlm | True | none | mistral | no_vlm | False | False | 0.338 | 0.291 | False |
| stick | no_llm | ok | benchmark_no_llm | True | qwen | none | no_llm | False | False | 0.338 | 0.29 | False |
| stick | no_contact_anchor | ok | benchmark_no_anchor | True | qwen | mistral | no_contact_anchor | False | False | 0.338 | 0.29 | False |
