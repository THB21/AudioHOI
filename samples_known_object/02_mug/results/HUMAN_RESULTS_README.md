# Human-side results (2026-07-08 overnight run)

`human_*`, `hands`, `contact_refine`, `audio_semantics`, and `gvhmr` contain
human-modeling outputs. `benchmark_vlm_qwen` contains object-pipeline outputs.

Chain: GVHMR (camera frame, K matches the case config) → HaMeR fingers stitched →
src/audio events + Qwen-VL contact records → body-side contact refinement AGAINST
Yixin's benchmark_vlm_qwen/object_pose.csv SE3 trajectory (sphere/capsule surface
geometry, scripts/shared/human_ball/contact/object_geometry.py) → full-scene render
(renders/*human_full_scene_3d*, --keep-mesh-origin + URDF-baked GLB so her SE3 pose
drives the actual object mesh).

HOI interaction metrics: results/hoi_eval/hoi_interaction_metrics.json, cross-case table
samples_known_object/hoi_interaction_evaluation/. Method: docs/hoi_interaction_evaluation_method_en.md
