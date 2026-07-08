# HaMeR hand-overlay diagnostic summary

- sample_dir: `samples/basketball_01`
- HaMeR IMAGE_SIZE: 256
- HaMeR FOCAL_LENGTH: 5000
- crop_scale: 0.28
- rescale_factor: 2.0

## Per-frame numbers

| frame | side | ds | crop_px | bbox_final | gvhmr_uv | hamer_full_uv | stitched_uv | ball% | notes |
|---|---|---:|---:|---:|---|---|---|---:|---|
| 1 | left | 1.57 | 202 | 403 | (748,330) | (746,322) | (748,330) | n/a |  |
| 1 | right | 1.58 | 202 | 403 | (516,325) | (517,320) | (516,325) | n/a |  |
| 20 | left | 1.58 | 202 | 403 | (762,357) | (759,353) | (762,357) | n/a |  |
| 20 | right | 1.58 | 202 | 403 | (541,397) | (539,386) | (541,397) | n/a |  |
| 60 | left | 1.58 | 202 | 403 | (752,386) | (743,376) | (752,386) | n/a |  |
| 60 | right | 1.58 | 202 | 403 | (606,409) | (594,389) | (606,409) | n/a |  |
| 100 | left | 1.58 | 202 | 403 | (737,382) | (730,379) | (737,382) | n/a |  |
| 100 | right | 1.58 | 202 | 403 | (536,457) | (531,450) | (536,457) | n/a |  |
| 140 | left | 1.58 | 202 | 403 | (743,386) | (737,383) | (743,386) | n/a |  |
| 140 | right | 1.58 | 202 | 403 | (517,421) | (512,413) | (517,421) | n/a |  |
| 180 | left | 1.57 | 202 | 403 | (749,389) | (738,380) | (749,389) | n/a |  |
| 180 | right | 1.57 | 202 | 403 | (547,376) | (538,378) | (547,376) | n/a |  |

## How to read this

- **ds (downsampling)** is `bbox_final_size / patch_size` (patch_size = HaMeR's pretrained ViT input edge, normally 256). HaMeR was trained on crops where the hand fills a meaningful portion of the patch. Rough heuristic:
  - `ds ≈ 0.5–2.0`: in-distribution.
  - `ds > 3`: crop is large relative to the patch — hand becomes tiny inside the 256² input.
  - `ds < 0.3`: crop is much smaller than the patch — hand is heavily upscaled, may show no usable detail.
- **patch image** (`*_patch.png`) shows what the network actually sees. If the hand is barely a few pixels wide, retrain-quality input is impossible regardless of stitching.
- **crop image** (`*_crop.png`):
  - orange skeleton = HaMeR's own prediction reprojected via its own depth. If this looks correct, HaMeR works on this crop.
  - green skeleton = stitched output (`run_hamer_hands.py` formula) projected via `K_fullimg`. If orange is right but green is wrong → the **stitch** is the bug (probably wrist 3D from GVHMR or coord frame).
  - cyan X = GVHMR wrist projection. If green X ≠ where the actual hand is, the anchor itself is wrong.
- **ball overlap** ≳ 40% on the holding hand means the crop is dominated by ball texture; HaMeR pretraining barely covers this regime.