# HOI Interaction Evaluation Summary (human-object layer)

Companion to Yixin's final_result_evaluation_summary (object layer). Human = GVHMR+HaMeR
stitched + body-side contact refine (sphere/capsule/mesh-SDF geometry). Object = audio
contact-phase trajectory (balls) or benchmark_vlm_qwen SE3 (mug/chair/stick).
Method + formulas: docs/hoi_interaction_evaluation_method_en.md.

| case | object_type | human_params | n_frames | pen_frame_ratio | pen_depth_max_mm | contact_frame_ratio | contact_gap_mm | part_correct_ratio | audio_events | contact_ratio_audio_windows | accel_at_events | accel_in_flight | object_jerk | grasp_stability_mm | mdev_star_mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| basketball | sphere | contact_refine | 192 | 0.0 | 0.0 | 0.6198 | 20.39 | 0.968 | 16 | 1.0 | 0.0437 | 0.05437 | 0.007007 | 1.01 | 31.35 |
| football | sphere | contact_refine | 242 | 0.0 | 0.0 | 0.0248 | 143.95 | 0.864 | 11 | 0.455 | 0.08574 | 0.02511 | 0.01167 |  |  |
| mug | sphere | contact_refine | 240 | 0.0 | 0.0 | 0.4667 | 15.78 | 1.0 | 7 | 0.571 | 0.00032 | 0.00059 | 1e-06 |  |  |
| chair | mesh | contact_refine | 192 | 0.2292 | 8.91 | 0.8542 | 3.68 | 0.667 | 3 | 1.0 | 0.00079 | 0.00053 | 1e-06 |  |  |
| stick | capsule | contact_refine | 240 | 0.0333 | 7.44 | 0.3708 | 54.22 | 0.875 | 8 | 0.5 | 0.02333 | 0.01436 | 0.000888 |  |  |

Reading: pen_* is MEASURED human-object interpenetration (hand/foot point clouds vs object
surface; chair via true mesh SDF, both directions). 0 for basketball/football/mug after body
refine; stick 3.3%/≤7.4mm; chair 23%/≤8.9mm after mesh-SDF Stage C (residual = torso leaning
on the backrest — hands/feet-only refine mask; PROX body-vertex sets are queue #4). Yixin's
object-centric 'Penetration Rate ~1.0' for balls/mug is a contact_depth_offset sign proxy, not
geometry — these columns supersede it. contact_ratio_audio_windows = audio events with real
geometric contact within ±2 frames (audio-visual consistency, novel). football's 144mm mean
kick-gap: sub-frame diagnosis shows ~half the events truly touch 2-3 frames off the rounded
event (foot depth ok, frame quantization); the rest are GVHMR foot-depth vs 2D-track limits at
fast 30fps juggling — queue #3 (sub-frame + onset-attribution) recovers the first half only.
