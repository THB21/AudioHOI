# Final Result Evaluation Source Migration

## Goal

Make `final_result/` the canonical source of final-result evaluation. Every evaluated
deliverable must bind the published video to the exact object trajectory, human model,
contact evidence, audio evidence, and source video that produced it.

## Constraints

- Do not evaluate historical `benchmark_vlm_qwen` directories as final deliverables.
- Do not combine artifacts with different frame counts or different source videos.
- Video evidence drives visual/VLM evaluation.
- Paired CSV/PKL evidence drives hard 6DoF, contact, penetration, and temporal metrics.
- Missing paired evidence must be explicit; no substitution from another run.

## Verified Current State

- `final_result/videos/1_basketball_3d.mp4`: 192 frames; paired 192-frame object and human artifacts exist.
- `final_result/videos/2_football_3d.mp4`: 242 frames; paired 242-frame object and human artifacts exist.
- `final_result/videos/3_mug_3d.mp4`: 192 frames; the checked-in mug object trajectory has 240 frames and must not be used for this deliverable.
- No chair or stick video is currently published under `final_result/videos/`.

## Steps

| Step | Status | Output |
|---|---|---|
| Define canonical final-result manifest and validate frame alignment | done | `final_result/evaluation_manifest.json` |
| Add manifest-backed final-result source loader | done | `final_result_sources.py` |
| Route final evaluator to canonical final deliverables | done | evaluator CLI/core changes |
| Add mismatch and missing-pair tests | done | `tests/test_final_hoi_evaluation.py` |
| Regenerate final-result evaluation outputs | done | `final_result/evaluation/` |
| Update evaluation documentation | done | Chinese and English docs |

## Acceptance

- The report identifies `final_result/videos/...` as the evaluated visual artifact.
- Every hard metric records its exact source path and frame-alignment status.
- The 192-frame mug video never consumes the 240-frame mug trajectory.
- Basketball and football hard metrics are recomputed from their paired final artifacts.
- Historical five-case benchmark results remain available only as benchmark/regression data.
