# Integrated generic Stage-4 audio-anchor candidates

These candidates are derived from the teammate-owned frozen inputs. They do not
overwrite `full_audio_vlm/object_pose.csv` or the frozen ablation poses.

## Intervention

`generic_sphere_sequence` now consumes typed audio events when audio is enabled:

- strong events near support states become soft 3D foot-height floor anchors;
- non-support events may become soft sphere-surface hand anchors only after a
  hand/object reprojection gate;
- audio events activate a minimum motion-kink residual;
- trusted hand-event intervals use exact camera-ray sphere-surface contact;
- `disable_audio_events` removes every audio-derived residual and artifact.

## Back-view result

| Metric | Full audio | No audio |
|---|---:|---:|
| detected audio events | 11 | 0 |
| audio floor anchors | 5 | 0 |
| audio hand anchors | 0 | 0 |
| trusted visual hand-contact frames | 2 | 2 |
| mean absolute support gap at strong bounces | 6.21 mm | 45.85 mm |
| hand surface gap, frame 161 | 0.00 mm | 0.00 mm |
| hand surface gap, frame 162 | 0.00 mm | 0.00 mm |

The absence of audio hand anchors is now an explicit, correct result: the strong
audio peaks coincide with support/bounce states. They are no longer silently
discarded, and they are not incorrectly relabelled as hand contacts.

Artifacts:

- `audio_integrated/generic_sphere_sequence_candidate.csv`
- `audio_integrated/generic_sphere_sequence_residuals.csv`
- `audio_integrated/generic_sphere_sequence_attempt.json`
- `audio_integrated_no_audio/` contains the controlled no-audio candidate.
