# LLM CSV Audit Summary: football

This audit reads structured CSV/JSON outputs and emits discrete labels only.
It does not modify pose, contact points, loss weights, or any result CSV.

| query | label | blocking | reason |
|---|---|---:|---|
| schema_completeness | pass | False | required schemas are present |
| stage_consistency | pass | False | stage outputs are mutually consistent |
| object_specific_rules | pass | False | case-specific rules pass |
| failure_range_summary | rotation_jump | False | inspect listed frames |

## Failure Labels

- `rotation_jump` from `failure_range_summary`: inspect listed frames
