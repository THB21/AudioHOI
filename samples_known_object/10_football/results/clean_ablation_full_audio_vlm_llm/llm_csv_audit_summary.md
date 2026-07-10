# LLM CSV Audit Summary: football

This audit reads structured CSV/JSON outputs and emits discrete labels only.
It does not modify pose, contact points, loss weights, or any result CSV.

| query | label | blocking | reason |
|---|---|---:|---|
| schema_completeness | pass | False | All required CSV outputs present with expected schemas for football case. |
| stage_consistency | pass | False | Stage2 contact candidates, Stage4 contact outputs, VLM gates, and metrics are mutually consistent. |
| object_specific_rules | pass | False | Case-specific semantic rules for contact, phase, static/freeze, and side assignments are valid. |
| failure_range_summary | rotation_jump | False | Frames 203 and low-trust optimizer frames require inspection for rotation jumps. |

## Failure Labels

- `rotation_jump` from `failure_range_summary`: Frames 203 and low-trust optimizer frames require inspection for rotation jumps.
