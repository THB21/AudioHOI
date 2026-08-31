# Back-view basketball: active audio/contact validation

## Scope and provenance

This directory is an isolated Stage-4 sphere candidate. It reads the teammate-owned
`object_observations.csv` and read-only `human_sites.csv`; it does not overwrite the
canonical `object_pose.csv`. The candidate was produced after inspecting the source
sequence at every audio event and recording the accepted interpretation in
`../../../audio_contact_analysis/trusted_event_ledger.csv`.

The canonical production `GenericSequenceExecutor` does not currently expose an
audio residual. This candidate therefore validates the audio/contact formulation in
the sphere solver; it must not be described as an accepted canonical Stage-4 result
until that residual is ported to the production executor and its gates pass.

## Event interpretation

- Frames 1--3: visually verified right-hand contact; hard sphere-surface anchor.
- Peaks 24, 43, 59, 79, and 97: strong audio peaks coincident with visible floor
  reversals; soft floor-contact anchors plus an impact/kink residual.
- Frames 109--240: the player gathers and visibly holds the ball; sustained nearest
  hand sphere-surface anchors. Weak late audio events do not create new impacts.

Audio is used as a temporal gate, not as an unconditional semantic label. A strong
peak becomes a floor constraint only after visual/support classification; otherwise
a hand constraint requires compatible projected hand geometry.

## Controlled result

The matched no-audio control is in `../active_visual_no_audio/`. Both variants use
the same input observations, human sites, hard initial hand contact, and sustained
held-ball interval. Only audio events, audio floor anchors, and audio impact terms are
disabled in the control.

| Measurement | Full audio + visual | Visual, audio disabled |
|---|---:|---:|
| Audio floor anchors | 5 | 0 |
| Audio impact terms | 11 | 0 |
| Trusted hard hand anchors | 3 | 3 |
| Sustained held-ball anchors | 130 | 130 |
| Mean absolute depth error to the five perspective floor targets | 158.4 mm | 1028.5 mm |
| 3D acceleration p95 (frame-space, m/frame^2) | 0.1156 | 0.0836 |
| 3D acceleration maximum (frame-space, m/frame^2) | 0.3075 | 0.3058 |

The audio-conditioned floor-target error is **84.6% lower** than the matched
no-audio control. The full/no-audio 3D position delta is concentrated before the
gather: mean 140.6 mm over the complete sequence and maximum 1526.0 mm at frame 24.
After frame 109 both runs intentionally agree because the trusted visible held-ball
constraint, not audio, explains that phase.

At frames 109--240 the center-to-hand-site sphere-surface residual has median
absolute magnitude 8.4 mm. Its p95 is 177.7 mm, so this proxy metric is not yet a
publication-ready mesh contact metric: a wrist/hand joint site is not the SMPL-X hand
surface, and the render visibly contacts/intersects the hand even where the point-site
residual is larger. A final evaluation should use mesh-to-object signed distance.

## Artifacts

- `generic_sphere_sequence_candidate.csv`: complete 240-frame 3D object candidate,
  contact labels, audio scores, and anchor provenance.
- `generic_sphere_sequence_residuals.csv`: per-frame depth/contact residual trace.
- `generic_sphere_sequence_attempt.json`: parameters, immutable input hashes, counts,
  and candidate hashes.
- `trusted_event_ledger.csv` in the analysis directory: the frame-level accepted and
  rejected audio/contact interpretation.

## Remaining production work

Port the floor, impact, and trusted hand-contact residuals into the canonical
capability problem / `GenericSequenceExecutor`; add mesh-surface contact distance and
floor support gates; then accept or reject the candidate without modifying the
teammate's source trajectory in place.

## Existing VLM/LLM correction integration

The teammate's hash-verified Stage-4 Qwen arbitration is now consumed by the
final correction audit. It contains eight effective decisions: one
`both_consistent` pass for frames 1--3 and seven `unclear` decisions. There are
no VLM rejects, so the existing Qwen evidence does not block this candidate;
unclear decisions remain conservative and do not directly update pose.

This case was launched with `llm_mode=none`. Its Stage-1/2 LLM result tables are
empty and explicitly document deterministic fallback, so no live LLM result is
claimed. The deterministic table audit and bounded correction recipes are in
`llm_vlm_correction_integrated_v2/`.

The corrected audit no longer compares a sphere center directly with a hand
center and no longer audits floor anchors against hand sites. It still reports
a genuine hand-site proxy gap at frames 127--129, 145, and 148--160. A trial
with stronger sustained-contact weight did not improve that interval, so the
current better trajectory is retained. This interval requires a mesh-surface
hand contact factor or improved hand sites; it is not silently forced to pass.

### Actual Stage 6.5 LLM implementation

The repository's real `stage_llm_csv_audit.py` was also run in an isolated
result directory with this candidate installed as `object_pose.csv`, using
`--llm-mode mistral`. The live `mistral-small-latest` request succeeded and
returned schema-valid decisions for all four audit questions. The trace records
`used: true`, response ID `38eda56d503e432dbcad1fafa0ebd368`, 11,628 prompt
tokens, 517 completion tokens, and 12,145 total tokens. No credential is stored
in the result artifacts.

The live LLM audit is blocking for two independent reasons: this isolated sphere path
does not publish the production contracts `physical_smooth_residuals.csv`,
`motion_regime.csv`, `pose_jump_audit.csv`, `optimizer_decisions.csv`, and
`object_contact_points.csv`; and it reports depth-jump review frames 21, 22,
and 27. The case-specific object rules passed. These are real integration gaps,
not LLM-generated pose corrections. The complete Stage 6.5 query/result/summary
artifacts are included in the deliverable under `llm_stage65_actual/`.
