# Canonical Final-Result Evaluation

This directory evaluates the published deliverables listed in
`final_result/evaluation_manifest.json`.

Current complete final-result scope:

- `basketball`
- `football`

These are the only cases that currently have aligned final videos, source videos,
object 6DoF trajectories, human/contact artifacts, temporal/audio artifacts, and
paired gate-trace evidence. Mug is intentionally excluded until its published
192-frame video has matching 192-frame source video, object pose, and human/contact
artifacts. Chair and stick are not published under `final_result/videos/`, so they
are not part of this final-result table.

## Source rule

- Visual deliverable: `final_result/videos/...`
- Visual alignment reference: the paired source video and SAM2 masks
- Object 6DoF: the exact trajectory used by the final render
- Human/contact/penetration: contact-refined SMPL-X plus body-surface contact records
- Temporal/audio: the same trajectory plus paired audio contact records
- Motion-regime temporal: translation/rotation spikes split by audio/contact event windows
- Gate impact: the final rendered pose/human/contact data stay tied to the published
  deliverable, while gate timeline, optimizer decisions, anchor state, residuals, and
  pre-smooth pose are read from the paired `benchmark_vlm_qwen` pipeline trace declared
  in `gate_trace_result_dir`.

`source_validation.csv` is the first file to inspect. A row is eligible for hard
metrics only when final video, source video, object pose, human parameters, and contact
data exist and their frame counts agree.

## Run

```bash
/home/yang/miniconda3/envs/audiohoi/bin/python \
  scripts/shared/generic_contact_pipeline/tools/run_final_hoi_evaluator.py
```

Use `--source pipeline-result` only for historical pipeline regression evaluation.

## Key outputs

- `final_evaluation_detailed.csv`: machine-readable summary across evaluated cases.
- `final_evaluation_human_readable.md`: six-column reader-facing table.
- `<case>/evaluation/temporal_plausibility_metrics.csv`: translation/rotation spikes,
  event-aligned versus non-event spikes, high-speed recall, oversmooth rate, and static-tail drift.
- `<case>/evaluation/gate_impact_metrics.csv`: gate counts, residual reweighting,
  anchor update permissions, freeze/interpolation frames, and pre-smooth to final pose deltas.
- `ablation/`: method-variant evaluation for the current complete final-result
  cases only, currently basketball and football.
