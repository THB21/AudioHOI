# Related-work Proxy Summary: basketball

This report maps inspected related-work ideas to AudioHOI proxy variants. It does not claim full reproduction of every external method.

| variant | status | missing information | evidence |
|---|---|---|---|
| video_only_tracking | method_proxy_defined | mesh/contact/audio/LLM/VLM | available metrics are read from v2 full outputs; missing-information condition is a related-work proxy, not a full rerun |
| mesh_only_alignment | method_proxy_defined | human contact/audio/VLM | available metrics are read from v2 full outputs; missing-information condition is a related-work proxy, not a full rerun |
| human_only_contact | method_proxy_defined | object semantic parts/mesh | available metrics are read from v2 full outputs; missing-information condition is a related-work proxy, not a full rerun |
| no_audio_event | method_proxy_defined | lift/place/static audio cues | available metrics are read from v2 full outputs; missing-information condition is a related-work proxy, not a full rerun |
| no_vlm_gate | executable_variant_defined | forced-choice visual verification | run stage_ablation --run-variant A3_v2_llm_prior_only to materialize full CSV/render outputs |
| no_llm_prior | executable_variant_defined | semantic HOI prior profile | run stage_ablation --run-variant A2_v2_no_llm_prior to materialize full CSV/render outputs |
| no_contact_refine | materialized_proxy | Stage4 contact/depth/SE(3) refinement | object_pose_init.csv compared with object_pose.csv |
| ours_full | materialized | none | stage6_compare_report.json + stage7_loss_residuals.csv + six render videos |

Interpretation:

- `materialized` means the current v2 directory contains concrete CSV/render evidence.
- `materialized_proxy` means the proxy is available from existing intermediate outputs, such as Stage3 before Stage4 refinement.
- `executable_variant_defined` means the variant has deterministic runner settings but has not necessarily been rendered in this directory yet.
- `method_proxy_defined` records the comparison axis for related-work discussion using the full v2 evidence.
