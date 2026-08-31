# Suitcase ablation evidence

This directory freezes the Full, No-VLM, and No-audio 6DoF trajectories and
binds the case-specific semantic/orientation diagnostics to their exact
SHA-256 hashes.

## VLM result

The Full trajectory uses the VLM-selected terminal symmetry branch. It flips
the incorrect opposite broad-face solution by about 169.65 degrees. The handle
broad-face margin changes from -1.69 to +1.88, the left/right side-exposure
margin from -0.96 to +0.66, and the frames 208-240 mean broad-face margin from
-1.85 to +1.91. Frames 1-164 are numerically identical to the pre-projection
Full trajectory, so the gain does not come from sacrificing the earlier mask
fit.

## Audio result

Over frames 179-240, Full reduces translation path by 22.32% and net rotation
drift by 27.18% relative to No-audio. Total rotation path is not reduced
(45.04 degrees Full versus 40.67 degrees No-audio), so the supported claim is
moderate motion-interval/drift improvement, not rotational-jitter removal.

The matching final videos are under `final_full_6d/`; the exact trajectories
are under `frozen_poses/`.
