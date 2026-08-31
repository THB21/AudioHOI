# LLM CSV Audit Summary: volleyball

This audit reads structured CSV/JSON outputs and emits discrete labels only.
It does not modify pose, contact points, loss weights, or any result CSV.

| query | label | blocking | reason |
|---|---|---:|---|
| schema_completeness | pass | False | required schemas are present |
| stage_consistency | pass | False | stage outputs are mutually consistent |
| object_specific_rules | depth_outlier | True | large_depth_step:0.556m |
| failure_range_summary | depth_outlier | False | inspect listed frames |

## Failure Labels

- `depth_outlier` from `object_specific_rules`: large_depth_step:0.556m
- `depth_outlier` from `failure_range_summary`: inspect listed frames
