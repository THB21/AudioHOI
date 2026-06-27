# Generic Pipeline v2 主线审计记录

日期：2026-06-27

## 当前主线

主线结果目录：

```text
samples_known_object/<case>/results/generic_pipeline_v2_llm_vlm_gate/
samples_known_object/<case>/results/renders/generic_pipeline_v2_llm_vlm_gate/
```

运行入口：

```bash
python scripts/shared/generic_contact_pipeline/run_pipeline.py \
  --case all \
  --result-name generic_pipeline_v2_llm_vlm_gate \
  --from-stage stage-1 \
  --to-stage stage7 \
  --llm-mode mistral \
  --vlm-mode dry-run
```

角色划分：

- Mistral：Stage -1 text-only LLM，生成离散 HOI semantic profile。
- Qwen-VL：真实图像 VLM gate，只做 forced-choice 检查。
- Optimizer：唯一做连续 6D / depth / contact / temporal / static 求解。

## Stage 主线

```text
stage-1  LLM semantic HOI prior
stage0   preprocess manifest / SAM2 / CoTracker / DA3 / audio / GVHMR availability
stage1   2D object observation + local semantic points
stage2   contact candidates
stage3   initial 2D-to-3D / 6D pose
stage4   contact/depth/anchor refinement
stage5   six-video render
stage6   baseline comparison
stage7   loss / residual logging
```

## 本次全量结果

四个 solved cases 已全量跑到 Stage7：

| case | LLM | fallback | prompt context | CSV/frame/pose/render compare | loss rows |
|---|---|---:|---:|---|---:|
| basketball | Mistral | false | 5 | pass | 192 |
| football | Mistral | false | 5 | pass | 242 |
| mug | Mistral | false | 18 | pass | 240 |
| chair | Mistral | false | 10 | pass | 192 |

Stage6 检查均通过：

- required CSVs exist and non-empty
- frame count matches solved baseline
- pose delta gate pass
- phase/event gate pass
- six standard render videos exist
- render codec check pass

chair 额外质量门通过：

```text
semantic_2d_median_within_1p10: true
semantic_2d_p90_within_1p10: true
contact_median_within_1p20: true
contact_p90_within_1p35: true
freeze_pass: true
```

## 标准输出

每个 case 都有：

```text
hoi_profile.json
prompt_context.json
llm_prior_trace.json
stage0_inputs_manifest.json
object_observations.csv
object_local_points.csv
contact_candidates.csv
object_pose_init.csv
object_pose.csv
object_contact_points.csv
stage6_compare_report.json
loss_analysis/per_frame_residuals.csv
pipeline_manifest.json
```

每个 case 都有六个标准视频：

```text
object_only/overlay.mp4
object_only/camera3d.mp4
object_only/side_yz.mp4
with_human/overlay.mp4
with_human/camera3d.mp4
with_human/side_yz.mp4
```

球类 overlay、mug camera3d 等抽查均为 H264 / yuv420p。

## 真实 Qwen-VL Gate 测试

已用真实 Qwen-VL 串行测试关键 stages。每次只启动一个 Qwen-VL 进程，避免 10GB GPU 并行 OOM。

测试报告：

```text
samples_known_object/generic_pipeline_v2_test_reports/qwen_stage_smoke_report_generic_pipeline_v2_llm_vlm_gate.json
```

覆盖：

| case | stages | result |
|---|---|---|
| basketball | stage1 / stage2 / stage4 / stage5 | all returncode 0 |
| football | stage1 / stage2 / stage4 / stage5 | all returncode 0 |
| mug | stage1 / stage2 / stage4 / stage5 | all returncode 0 |
| chair | stage1 / stage2 / stage4 / stage5 | all returncode 0 |

观测：

- stage1/stage2/stage5 多数为 pass。
- stage4 在若干 case 返回 `pass_with_unclear_frames`，这是预期行为：unclear 不进入连续优化，只禁止该帧 anchor/contact update。
- 并行启动多个 Qwen-VL 会触发 OOM；真实 VLM 测试必须串行执行，或降低模型/量化/resize。

## VLM Blocking / Gate 行为测试

为了验证 reject/unclear 是否真的进入 gate，而不是只写文本标签，做了一个隔离 synthetic gate test。

报告：

```text
samples_known_object/generic_pipeline_v2_test_reports/synthetic_gate_test_report.json
```

结果：

```text
pass frame: 10 -> contact residual kept
reject frame: 20 -> blocked_by_vlm_reject, contact residual disabled
unclear frame: 30 -> no update, contact residual disabled
```

这证明：

- `reject` 会形成 blocking decision。
- `reject/unclear` 都会进入 `disabled_contact_frames`。
- gate 层只开关 predefined residual，不直接输出 pose/坐标/loss weight。

## Mistral Ablation

已生成 Stage4 级 ablation，所有变体均独立写入子目录，不覆盖主线：

```text
samples_known_object/<case>/results/generic_pipeline_v2_llm_vlm_gate/<variant>/
```

变体：

```text
A2_v2_no_llm_prior
A3_v2_llm_prior_only
A4_v2_vlm_gate_only
A5_v2_llm_prior_plus_vlm_gate
A6_v2_no_contact_gate
A7_v2_no_depth_gate
A8_v2_no_anchor_propagation
```

检查结果：

- 四个 case × 七个变体均有 `stage4_metrics.json`、`object_pose.csv`、`pipeline_manifest.json`。
- A3/A5/A6/A7/A8 均为 `llm_mode=mistral`，且 `mistral_status=ok`、`fallback_used=false`。
- A2/A4 为 no-LLM 对照，`llm_mode=none`。

汇总文件：

```text
samples_known_object/<case>/results/generic_pipeline_v2_llm_vlm_gate/ablation_summary.csv
samples_known_object/<case>/results/generic_pipeline_v2_llm_vlm_gate/ablation_summary.json
samples_known_object/<case>/results/generic_pipeline_v2_llm_vlm_gate/ablation_summary.md
```

## 当前仍需人工/未来处理的项

以下不是“主线能否跑通”的阻塞项：

1. 人工视频复核：
   - 代码层 compare 已通过。
   - 仍建议人工看 `with_human/overlay.mp4` 和 `with_human/side_yz.mp4`，尤其是 football 抖动、mug handle、chair 双手 endpoint。

2. 新 object zero-shot：
   - 当前审计范围是四个 solved cases 的主线统一。
   - 新 object 需要新的 sample metadata / mask / tracks / depth / body data，不能仅凭当前四个 case 的结果证明。

3. 全量真实 Qwen-VL：
   - 关键 stages 已串行 smoke 通过。
   - 若需要最终论文实验，可把 `--limit 2` 放开，但这会显著增加时间和显存占用。

## 当前结论

当前 `generic_pipeline_v2_llm_vlm_gate` 已经可以作为四个 solved cases 的完整主线运行：

- Mistral LLM prompt-aware Stage -1 可用。
- Qwen-VL 关键 stage 串行 smoke 可用。
- 四个 case 全部能从 Stage -1 跑到 Stage7。
- 六视频标准输出齐全。
- Stage6 baseline comparison 全部通过。
- 每帧 loss/residual logging 已生成。
- VLM reject/unclear gate 行为已通过 synthetic test。
- Mistral ablation 已独立产出，不覆盖主线。
