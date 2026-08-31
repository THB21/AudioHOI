#!/usr/bin/env python3
"""Run all nine cases with visual proposals and VLM, with audio fully disabled."""
from nine_modality_ablation import main


if __name__ == "__main__":
    main("visual_vlm_only")
