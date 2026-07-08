# Final Result Evaluation Method

This document describes how the current generic HOI pipeline evaluates final results. A "final result" means the selected final result directory for each case, `benchmark_vlm_qwen` by default. This is not a benchmark comparison table and does not compare baseline, ablation, or oracle-contact variants.

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

**Contact Proxy**:

```text
Contact Proxy = mean_t( c_t if contact_observed_t = 1 else 0 )
```

**Overlay Proxy** first uses mask, track, or line confidence:

```text
Overlay Proxy = mean_t(clip(mask_iou_t or observation_conf_t or track_conf_t, 0, 1))
```

If direct confidence is unavailable, it is inferred from jitter:

```text
Overlay Proxy = mean_t( 1 / (1 + max(0, jitter_px_t) / 20) )
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

**Penetration Rate / Floating Rate**:

When explicit residuals exist:

```text
Penetration Rate = mean_t(1[penetration_depth_t > 1e-4])
Floating Rate = mean_t(1[floating_gap_t > 1e-3])
```

When explicit residuals are missing, the current evaluator derives a proxy from `contact_depth_offset_m`:

```text
penetration proxy: contact_depth_offset_m < -1e-6
floating proxy: contact_depth_offset_m > 1e-6
```

**Jump Count**:

```text
Jump Count = Σ_t 1[
  visual_spike_t = 1
  or contact_spike_t = 1
  or smoothness_spike_t = 1
]
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
python scripts/shared/generic_contact_pipeline/tools/run_final_summary.py
```

Default outputs:

- `samples_known_object/final_result_evaluation/final_result_evaluation_summary.csv`
- `samples_known_object/final_result_evaluation/final_result_evaluation_summary.md`
- `samples_known_object/final_result_evaluation/final_result_evaluation_summary.html`
- `samples_known_object/final_result_evaluation/final_result_evaluation_summary_manifest.json`

Each row evaluates one case's current final result only. It does not include baselines or ablations. Use the benchmark report only when comparing method contributions.

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
