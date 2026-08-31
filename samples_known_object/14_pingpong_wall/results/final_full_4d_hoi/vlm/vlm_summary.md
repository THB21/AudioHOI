# VLM Verification Summary: pingpong_wall

Mode: local Qwen-VL forced-choice verification.

- Total queries: 30
- Total result rows: 30

## Stage Decisions

- stage1: pass blocking=False gates={'pass': 6} `/tmp/audiohoi-nine/samples_known_object/14_pingpong_wall/results/final_full_4d_hoi/vlm/stage1/stage_decision.json`
- stage2: blocked_by_vlm_reject blocking=True gates={'pass': 6, 'reject': 4} `/tmp/audiohoi-nine/samples_known_object/14_pingpong_wall/results/final_full_4d_hoi/vlm/stage2/stage_decision.json`
- stage3: pass_with_unclear_frames blocking=False gates={'unclear': 3} `/tmp/audiohoi-nine/samples_known_object/14_pingpong_wall/results/final_full_4d_hoi/vlm/stage3/stage_decision.json`
- stage4: pass_with_unclear_frames blocking=False gates={'unclear': 8, 'pass': 3} `/tmp/audiohoi-nine/samples_known_object/14_pingpong_wall/results/final_full_4d_hoi/vlm/stage4/stage_decision.json`
