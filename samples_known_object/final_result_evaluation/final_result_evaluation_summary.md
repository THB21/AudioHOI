# Final Result Evaluation Summary

This table evaluates only the current final result for each case. It is not a benchmark and does not compare methods.

| Case | Final Result | Frames | SE3 Pose | Translation | Rotation | Contact GT | Contact F1 | Contact Proxy | Overlay Proxy | Anchor Drift Mean ↓ | Anchor Drift Max ↓ | Penetration Rate ↓ | Floating Rate ↓ | Jump Count ↓ | Static Drift Max ↓ | Geometry Spread ↓ | VLM Judge ↑ | LLM Audit | Failure Stage | Final Pass | Evaluation Summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basketball | benchmark_vlm_qwen | 192.000000 | yes | yes | yes | proxy_only |  | 0.176661 | 1.000000 | 0.178786 | 3.841463 | 1.000000 | 0.987261 | 0.000000 | 0.000000 |  | 0.750000 | pass | stage2_contact | no | /mnt/hdd/AudioHOI/samples_known_object/01_basketball/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/evaluation_summary.json |
| football | benchmark_vlm_qwen | 242.000000 | yes | yes | yes | proxy_only |  | 0.144876 | 1.000000 | 9.346088 | 30.919140 | 0.984127 | 0.988827 | 0.000000 | 0.000000 |  | 0.750000 | pass | stage2_contact | no | /mnt/hdd/AudioHOI/samples_known_object/10_football/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/evaluation_summary.json |
| mug | benchmark_vlm_qwen | 240.000000 | yes | yes | yes | proxy_only |  | 0.748385 | 0.800743 | 0.206064 | 0.596717 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |  | 0.750000 | pass | stage2_contact | no | /mnt/hdd/AudioHOI/samples_known_object/02_mug/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/evaluation_summary.json |
| chair | benchmark_vlm_qwen | 192.000000 | yes | yes | yes | proxy_only |  | 0.786656 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  | 1.000000 | pass | pass | yes | /mnt/hdd/AudioHOI/samples_known_object/05_chair/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/evaluation_summary.json |
| stick | benchmark_vlm_qwen | 240.000000 | yes | yes | yes | proxy_only |  | 0.863820 | 1.000000 | 0.082190 | 0.652714 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | pass | stage2_contact | no | /mnt/hdd/AudioHOI/samples_known_object/11_stick/results/benchmark_vlm_qwen/vlm_trace/06_evaluation/evaluation_summary.json |

Notes: `Contact F1` is blank unless manual labels exist. `Contact Proxy` is reported separately and must not be read as ground-truth F1.
