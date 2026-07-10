# Final Result Evaluation Method

This document evaluates deliverables published under `final_result/`. The default source is
`final_result/evaluation_manifest.json`. The current evaluated set contains basketball and
football because both have frame-aligned final video, source video, object pose,
contact-refined human parameters, and contact evidence. Historical `benchmark_vlm_qwen`
results are pipeline regressions and require explicit `--source pipeline-result`.

Every hard metric must use the exact data paired with the published video. A video without
matching pose/human artifacts can receive visual review only; another run's CSV must not be
substituted for 6DoF or physical metrics.

## Evaluated Target

Every final case must expose a unified SE3 pose schema:

- Translation: `tx, ty, tz`
- Quaternion rotation: `qw, qx, qy, qz`

Basketball, football, mug, chair, and stick all use this schema. For spherical objects, rotation is weakly constrained or treated as an identity gauge because self-rotation is not reliably observable from the current proxy geometry. Mug, chair, and stick rotations are more directly tied to rendering and physical consistency.

## Three-Layer Evaluation

### 1. Hard Metrics

Hard metrics are computed from machine artifacts and do not depend on VLM explanations:

- `Contact F1`: computed only when manual contact labels exist.
- `Contact Proxy`: a confidence proxy used when manual labels are missing. It must not be interpreted as F1.
- `Overlay Proxy`: proxy alignment score from mask, track, line, or render evidence.
- `Anchor Drift`: drift between stable anchors and current observed anchors.
- `Penetration Rate`: proxy for human-object or support penetration.
- `Floating Rate`: proxy for unsupported object gaps.
- `Jump Count`: number of jump frames in `pose_jump_audit.csv`.
- `Static Drift`: maximum drift during a static tail segment.
- `Geometry Spread`: object length or scale variation; rigid objects such as sticks should not change physical length.

### Hard Metric Formulas

Let `p_t ∈ {0,1}` be the predicted contact state at frame `t`, `y_t ∈ {0,1}` the manual contact label, `c_t ∈ [0,1]` the contact confidence, `d_t` the anchor drift, and `pen_t` / `float_t` the penetration and floating proxy values.

**Contact F1** is computed only when manual labels exist:

```text
TP = Σ 1[p_t = 1 and y_t = 1]
FP = Σ 1[p_t = 1 and y_t = 0]
FN = Σ 1[p_t = 0 and y_t = 1]
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Contact F1 = 2 * Precision * Recall / (Precision + Recall)
```

When manual labels are missing, `Contact F1` is left blank. Proxy scores are not reported as F1.

**Contact Proxy** uses the mean human-surface to object-surface gap on expected-contact frames:

```text
Contact Proxy = exp(-contact_gap_mm / 50)
```

**Overlay** compares the source-video SAM2 mask with the object mask projected from the paired final pose:

```text
Overlay = mean_t IoU(observed_object_mask_t, projected_object_mask_t)
```

For a sphere, the projected radius is:

```text
r_px = fx * radius_m / tz
```

**Anchor Drift**:

```text
d_t = || observed_local_t - stable_local_t ||_2
```

For line objects, the evaluator uses the local coordinate along the line:

```text
d_t = | observed_local_s_t - stable_local_s_t |
```

The report writes:

```text
Anchor Drift Mean = mean_t(d_t)
Anchor Drift Max = max_t(d_t)
```

**Penetration Rate** uses signed distance between contact-refined SMPL-X part vertices and
the object surface with 3 mm mesh slack:

```text
Penetration Rate = #frames(any vertex depth > 3 mm) / #valid frames
```

**Jump Count**:

```text
Jump Count = Σ_t 1[
  visual_spike_t = 1
  or contact_spike_t = 1
  or smoothness_spike_t = 1
]
```

The current temporal evaluator also reports motion-regime-aware spikes:

```text
v_t = ||T_t - T_{t-1}||
a_t = ||v_t - v_{t-1}||
omega_t = angle(q_{t-1}^{-1} q_t)
alpha_t = |omega_t - omega_{t-1}|

translation_spike_t = 1[a_t > threshold_translation]
rotation_spike_t = 1[alpha_t > threshold_rotation]
threshold = max(floor, median(values) + 3 * MAD(values), percentile_95(values))
```

Audio/contact windows split spikes by expected motion regime:

```text
event_aligned_spike_count = Σ 1[spike_t and t in event_window]
non_event_spike_count = Σ 1[spike_t and t not in event_window]
```

High-speed preservation and over-smoothing:

```text
high_speed_recall = #event windows with preserved acceleration peak / #event windows
oversmooth_rate = #event windows with suppressed acceleration peak / #event windows
```

**Static Drift**:

```text
Static Drift Max = max_t(static_tail_drift_m_t)
```

**Geometry Spread**:

```text
Geometry Spread = max_t(length_t) - min_t(length_t)
```

Primary inputs:

- `object_pose.csv`
- `object_observations.csv`
- `contact_candidates.csv`
- `anchor_state.csv`
- `physical_smooth_residuals.csv`
- `pose_jump_audit.csv`
- `temporal_plausibility_metrics.csv`
- `gate_impact_metrics.csv`

**Gate Impact** is for ablation analysis. It does not judge visual quality directly;
it answers whether VLM/LLM/audio gates actually affected optimization:

```text
gate_active_count = Σ active_gate
optimizer_reweighted_frames = Σ 1[feedback_reweight_reason exists]
anchor_update_blocked_count = Σ 1[anchor_update_allowed = 0]
freeze_interpolation_frames = Σ 1[freeze/interpolation/static_tail residual enabled]
pose_delta_translation_max = max_t ||T_final_t - T_pre_smooth_t||
pose_delta_rotation_max = max_t angle(q_pre_smooth_t^{-1} q_final_t)
```

### 2. VLM Visual Judge

The VLM visual judge is a visual audit layer. It asks questions such as:

- Is the target object visible and tracked correctly?
- Does the rendered overlay align with the real object?
- Are contact points attached to plausible hands, feet, floor, table, or body regions?
- Are floating, penetration, jumps, wrong orientation, or object length changes visible?
- Is the full interaction physically plausible?

The report separates internal pipeline VLM gates from the final visual judge. The VLM does not generate poses and does not override geometric facts. It only produces visual audit scores and failure explanations.

### 3. LLM CSV Auditor

The LLM CSV auditor reads CSV/JSON artifacts and checks pipeline consistency:

- Did gates actually affect the optimizer?
- Did anchors become stale?
- Did static-tail freeze activate when needed?
- Were pose jumps recorded and suppressed?
- Did the failure originate from observation, contact, pose init, optimizer, or render?

The LLM also does not output coordinates or poses. It only outputs audit conclusions.

## Final Summary Table

Generate the final-only summary table with:

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py
```

Inspect `final_result/evaluation/source_validation.csv` first, then
`final_result/evaluation/final_evaluation_detailed.csv`.

Default outputs:

- `samples_known_object/final_result_evaluation/final_evaluation_detailed.csv`
- `samples_known_object/final_result_evaluation/final_evaluation_human_readable.md`
- `samples_known_object/final_result_evaluation/final_evaluation_summary_manifest.json`

Each row evaluates one case's current final result only. It does not include baselines or ablations. Use the benchmark report only when comparing method contributions.

The older `run_final_summary.py` / `final_result_evaluation_summary.*` outputs are legacy object-only summaries. They remain for compatibility, but they are not the current final HOI reporting table.

## Acceptance Criteria

A final result should satisfy:

- `SE3 Pose = yes`
- `Jump Count = 0`, or every remaining jump has a clear audit explanation
- Low static drift during static segments
- Low penetration and floating rates
- Reasonable `Contact Proxy` or manual `Contact F1`
- A concrete `Failure Stage`, not a vague "render bad" explanation

## Why Many Current Cases Show `stage2_contact`

In the current final summary, basketball, football, mug, and stick often show `Failure Stage = stage2_contact`. This does not mean the final render is necessarily globally bad. It means the evaluator traces the first failing evidence source to **contact and anchor evidence**:

- These cases do not have manual contact labels, so `Contact F1` is blank and the evaluator must use `Contact Proxy`.
- `anchor_drift_fail` is triggered when `Anchor Drift Max > 0.08m`. Basketball, football, mug, and stick currently exceed this threshold.
- Basketball, football, and mug also derive floating/penetration proxies from `contact_depth_offset_m`; non-zero depth offsets can produce high `Floating Rate` or `Penetration Rate`.
- Failure-stage priority is overlay/geometry → contact/anchor → physical optimizer. Therefore, once `anchor_drift_fail` is true, the summary reports `stage2_contact` even when penetration or floating proxies also fail.

So `stage2_contact` should be read as: **the final pose has an SE3 schema and may have zero pose jumps, but contact/anchor evidence is still not strong enough for the evaluator to mark the result as fully passing.** This is why chair passes while the other cases remain `pass=no`.
