# VLM-Gated Interval Candidate Selection

## Objective

Prevent a locally useful occlusion repair from degrading visible portions of an object trajectory. The production solver remains case-name independent. VLM only chooses between finite solver candidates and never emits pose values, loss weights, or continuous corrections.

## Scope

- Object trajectory reconstruction only.
- Human pose remains a read-only observation and render context.
- Stage 4 writes isolated candidates; accepted output is unchanged until the existing publication gate passes.
- No downstream human refinement, ablation evaluation, or new regression-test suite is part of this change.

## Candidates

Each Stage 4 run may produce two attempts from the same typed inputs and state contract:

1. `stable`: the ordinary generic solve without amodal completion factors.
2. `occlusion_challenger`: the same generic solver with VLM-approved amodal-mask measurements active only inside bounded missing-observation intervals.

Both attempts have complete state, residual, factor-ledger, hard-metric, and provenance artifacts. Candidate generation cannot read accepted, baseline, or historical solved poses.

## VLM Decision Contract

For each bounded risk interval, evidence contains the original RGB/mask context plus synchronized renders from both candidates. Qwen returns exactly one label:

- `keep_stable`
- `use_occlusion_challenger`
- `reject_both`
- `unclear`

`unclear` is a strict no-update decision and therefore keeps `stable`. `reject_both` blocks publication and preserves both attempts for diagnosis. VLM cannot create a pose, alter a threshold, or select frames outside the declared interval.

## State Composition

The selector starts from the complete stable trajectory. It copies challenger SE(3) states only inside intervals labeled `use_occlusion_challenger`. A short boundary transition is generated with linear translation blending and shortest-path quaternion SLERP. Quaternion normalization is mandatory. Frames outside the selected interval and transition are byte-identical to the stable candidate.

The composed result is a third isolated attempt with parent attempt IDs, query ID, evidence hash, response hash, chosen interval, transition frames, and per-frame source recorded in provenance.

## Failure Handling

- Missing or hash-mismatched VLM evidence: `reject_both` and block publication.
- Valid `unclear`: keep stable without changing factor activation.
- Invalid quaternion, NaN, or transition discontinuity: reject composed candidate.
- Hard-metric regression outside the selected interval: reject composed candidate.
- Candidate render failure: block VLM selection; never fall back to stale accepted renders.

## Verification

Verification uses diagnostics rather than adding a new test suite:

- confirm the VLM evidence references both current attempt renders;
- confirm frames outside selected intervals match stable state exactly;
- confirm all quaternion norms remain within numerical tolerance of one;
- report translation and angular step statistics around both boundaries;
- render risk-window montages before presenting the candidate;
- confirm accepted output hashes are unchanged while publication is blocked.

## Suitcase Acceptance Target

For the current suitcase sequence, frames 1–145 and 173–240 must retain the approved stable trajectory. The VLM-approved occlusion interval is 146–172. The interval-selected render must preserve the stable opening and frame-106 region while using the visually superior occlusion challenger only inside that interval.
