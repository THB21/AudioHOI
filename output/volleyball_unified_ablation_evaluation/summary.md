# Volleyball object reconstruction and ablation summary

Full uses the same generic sphere sequence solver as the other sphere cases.
SAM2 supplies the visible ball mask, CoTracker supplies temporal hypotheses,
audio supplies impact timing, Qwen supplies the discrete single-ball/contact
relation, and GVHMR supplies read-only palm sites. Human state is not optimized.

| Metric | Full | No-audio | No-VLM |
|---|---:|---:|---:|
| Publication | accepted | candidate blocked | candidate blocked |
| Visible-mask error P95 (px) | 4.11 | 4.10 | 569.14 |
| Frames 145--180 identity error P95 (px) | 0.04 | 0.04 | 704.14 |
| Contact gap P95 (mm) | 77.07 | 84.57 | 141.11 |
| Contact gap max (mm) | 91.05 | 91.07 | 141.11 |
| Trajectory jerk P95 (m/frame^3) | 0.1210 | 0.1183 | 0.2007 |

VLM provides the dominant gain in this clip: it declares one physical ball and
allows the solver to prefer the visible SAM2 component over a stale persistent
track, including bounded interpolation across short missing-mask intervals.
The 145--180 failure interval falls from 704.14 px to 0.04 px P95.

Audio provides a smaller but measurable contact gain. With the same VLM and
visible-mask evidence, audio-timed event gating lowers contact-gap P95 from
84.57 mm to 77.07 mm and is the difference between a blocked candidate and an
accepted Full publication. Audio is not claimed to improve every visible-frame
2D position.

The six deliverable videos are under
`samples_known_object/13_volleyball/results/renders/volleyball_full/` in
`object_only/` and `with_human/`. The two blocked ablations retain only
`ablation_pose.csv` and provenance; they do not overwrite canonical outputs.
