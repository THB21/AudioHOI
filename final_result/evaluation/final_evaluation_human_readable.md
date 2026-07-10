# Final HOI Evaluation Human-Readable Summary

Reader-facing table: only metrics with current final-result evidence are shown in the main cells.
Unavailable anchor drift, floating, legacy jump count, and static-tail drift are listed below the table when their source artifacts are absent.

| Case | Object 6DoF | Visual Overlay | Contact/Anchor | Physical | Temporal |
| --- | --- | --- | --- | --- | --- |
| basketball | SE3=yes; frames=192; T/R valid=1/1 | IoU=0.789; mask coverage=0.982; false coverage=0.2; source=generated eval proxy render mask iou | proxy=0.48; gap=36.72mm; contact frames=0.474; observed rows=183; part correct=1 | penetration rate=0.213; depth mean/max=13.3mm/49.01mm; contact-physics tradeoff=0.356 | jerk=0.005; T/R spikes=9/0; event/non-event spikes=9/0; high-speed recall=0.871; oversmooth=0.129; failures=[] |
| football | SE3=yes; frames=242; T/R valid=1/1 | IoU=0.77; mask coverage=0.952; false coverage=0.206; source=generated eval proxy render mask iou | proxy=0.007; gap=250.59mm; contact frames=0.062; observed rows=50; part correct=0.615 | penetration rate=0.017; depth mean/max=19.65mm/53.55mm; contact-physics tradeoff=0.031 | jerk=0.008; T/R spikes=12/0; event/non-event spikes=10/2; high-speed recall=0.842; oversmooth=0.158; failures=[{"start":83,"end":84}] |

## Unavailable Evidence Notes

- basketball: anchor drift: no stable/observed local anchor coordinates in final contact artifacts; floating: no final support-gap/floor-state artifact; legacy jump_count: replaced by motion-regime spike metrics; static-tail drift: no explicit static interval for this final result.
- football: anchor drift: no stable/observed local anchor coordinates in final contact artifacts; floating: no final support-gap/floor-state artifact; legacy jump_count: replaced by motion-regime spike metrics; static-tail drift: no explicit static interval for this final result.
