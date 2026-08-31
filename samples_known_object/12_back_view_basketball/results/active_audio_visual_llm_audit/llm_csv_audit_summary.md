# LLM CSV Audit Summary: back_view_basketball

This audit reads structured CSV/JSON outputs and emits discrete labels only.
It does not modify pose, contact points, loss weights, or any result CSV.

| query | label | blocking | reason |
|---|---|---:|---|
| schema_completeness | schema_missing | True | missing required schema fields/files |
| stage_consistency | stage_inconsistent | True | missing_optimizer_decisions |
| object_specific_rules | pass | False | case-specific rules pass |
| failure_range_summary | depth_outlier | False | inspect listed frames |

## Failure Labels

- `schema_missing` from `schema_completeness`: missing required schema fields/files
- `stage_inconsistent` from `stage_consistency`: missing_optimizer_decisions
- `depth_outlier` from `failure_range_summary`: inspect listed frames
