# Temporal + Gate Impact Implementation Plan

Status: done

Goal: fully materialize the documented final-result temporal evaluation and make ablation reports able to answer whether VLM, LLM, and audio changed constraints, residuals, anchor updates, freeze/interpolation, and final poses.

## Steps

1. done - Inspect current final evaluator, temporal metrics, ablation runner, and available gate artifacts.
   - Findings: current final outputs include velocity/acceleration/jerk and audio event split, but not spike taxonomy, high-speed preservation, oversmooth rate, or gate impact deltas.
2. done - Implement `temporal_plausibility_metrics` as a hard metric block.
   - Outputs: `temporal_plausibility_metrics.csv/json`.
   - Fields: translation/rotation spike counts, event-aligned and non-event spike counts, high-speed recall, oversmooth rate, static drift proxies, temporal failure intervals.
3. done - Implement `gate_impact_metrics` as an artifact/audit block.
   - Outputs: `gate_impact_metrics.csv/json`.
   - Fields: gate counts, residual switch counts, anchor update allowed/blocked counts, freeze/interpolation counts, pre/post/final pose deltas.
4. done - Wire both blocks into final summary, human-readable table, pipeline manifest, and ablation fields/deltas/report.
5. done - Update evaluation and ablation docs so the table explains the new metrics and admits what is direct evidence versus proxy.
6. done - Regenerate final-result evaluation and available ablation evaluation.
7. done - Verify with compile/tests and inspect generated outputs for non-empty new fields.

## Verification

- `run_final_hoi_evaluator.py` regenerated `final_result/evaluation/final_evaluation_detailed.csv`.
- `run_ablation_evaluation.py` regenerated `final_result/evaluation/ablation/ablation_table.csv`.
- `compileall` passed.
- `pytest -q tests/test_final_hoi_evaluation.py tests/test_vlm_trace_final_evaluator_benchmark.py` passed: 31 passed, 1 skipped.
- `git diff --check` passed.

## Constraints

- Do not claim VLM/LLM/audio improvements from identical result directories.
- Missing gate artifacts must be reported as `missing_*`, not converted into fake zeros.
- VLM/LLM do not generate pose; gate impact only records whether their gates changed optimizer inputs/decisions.
- Mug remains excluded from canonical final-result evaluation unless the user re-enables it.
