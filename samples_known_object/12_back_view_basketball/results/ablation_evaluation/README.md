# Back-view basketball ablation evidence

This directory freezes the independent object-stage outputs for the back-view
basketball held-out case.  The three primary variants differ only in typed
audio/VLM evidence availability:

- `full`: audio enabled, Qwen VLM evidence enabled.
- `no_vlm`: audio enabled, VLM semantic evidence disabled.
- `no_audio`: audio disabled, the same twelve Stage 4 Qwen queries/evidence as
  `full` retained to hold the VLM intervention fixed.

`pure_solver_no_audio_no_vlm` is retained as an additional reference, but is
not one of the three rows in the unified ablation table.  No ablation videos
are rendered; only their pose CSVs are frozen.  `final_full/` contains the
previously accepted Full render paired with the frozen Full pose.

## Current result

The unified evaluator confirms three distinct pose hashes and valid method
manifests.  Audio has a favorable contact-gap effect: Full is 167.69 mm versus
218.50 mm for No-audio (50.80 mm improvement).  VLM does not yet show a
positive hard-metric gain in this case: No-VLM is 70.14 mm and has a slightly
higher overlay score (0.64810 versus 0.64356).  Therefore this case is archived
as a reproducible but **not yet promotion-ready** ablation; it must not be used
to claim a positive VLM improvement.

The fixed-query policy prevents removal of audio from changing the VLM query
population (an adaptive run attempted to expand 12 queries to 967).  That
interrupted diagnostic is intentionally excluded from this archive.

See `unified/ablation_table.csv`, `unified/ablation_delta_table.csv`, and
`artifact_manifest.json` for exact values and hashes.
