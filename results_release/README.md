# AudioHOI results release

This directory is the compact, versioned result bundle for the nine supported
interaction sequences. It contains final object trajectories and numeric
evaluation records only. Paper sources, model weights, third-party repositories,
preprocessing caches, solver attempts, raw VLM conversations, and render-debug
directories are intentionally excluded.

## Contents

- poses/<case>/object_pose.csv: one frozen selected trajectory per case.
- metrics/main_object_metrics.csv: common metrics for all nine cases.
- metrics/metric_dictionary.csv: units, direction, and definitions.
- metrics/case_specific_metrics.csv: challenge-specific diagnostics.
- ablations/multimodal_ablation_metrics.csv: Audio+VLM, Audio-only, VLM-only,
  and neither for the four challenge cases.
- ablations/objective_loss_ablation_*.csv: object-factor removal results.
- ablations/configs/: frozen ablation availability and input-hash audits.
- manifests/pose_manifest.csv: expected frame count and SHA-256 for every
  released trajectory.
- manifests/case_config_manifest.csv: runtime config, metadata, and geometry.
- manifests/media_manifest.csv: provenance and SHA-256 for release videos.
- SHA256SUMS: checksums for every file in this directory except itself.

## Validate

Run this command from the repository root:

    python scripts/release/validate_results_release.py

The validator checks all nine cases, exact frame counts, finite SE(3) fields,
quaternion normalization, file hashes, metric coverage, and the four-condition
challenge ablation matrix.

## Large media

Input videos, final object-only overlays, final overlays with the read-only
human skeleton, and challenge comparison videos are distributed separately:

https://github.com/THB21/AudioHOI/releases/tag/results-v1

Their original paths and hashes are recorded in manifests/media_manifest.csv.
The Git repository is authoritative for code and numeric artifacts; the GitHub
release is authoritative for large media.

