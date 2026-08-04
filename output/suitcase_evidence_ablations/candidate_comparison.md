# Suitcase object-only evidence ablation comparison

All four candidates use the same rigid asset, Stage 1–3 measurements, generic
sequence solver, loss, thresholds, and 80-evaluation budget. Only typed audio
and semantic evidence factors are removed. No candidate was promoted to the
canonical `object_pose.csv`.

| variant | total projection p95 (px) | point p95 (px) | contact p95 (m) | rotation path 111–163 (deg) | rotation p95 111–163 (deg/frame) |
|---|---:|---:|---:|---:|---:|
| full | 23.827 | 26.307 | 0.08732 | 289.919 | 12.533 |
| no VLM | 21.245 | 28.533 | 0.09123 | 161.278 | 9.584 |
| no audio | 23.757 | 28.528 | 0.09097 | 163.769 | 9.816 |
| vision only | 13.344 | 28.939 | 0.09148 | 159.113 | 9.602 |

Interpretation: the current measurable benefit is joint evidence activation.
Audio identifies a moving supported interval and VLM supplies the calibrated
turn relation; together they recover the long turn and improve point/contact
fit. Neither stream alone recovers that turn. The full candidate still fails
the point and contact publication gates and retains visible rotational jitter,
so this is evidence of gain, not an accepted final result.
