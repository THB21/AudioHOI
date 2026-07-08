# Grasp-contact constraint — 02_mug_v2

- object: `mug_oriented_pose.csv`  | frames: 192  | grasp frames (within 0.12 m): 170

| metric | before | after |
|---|---|---|
| mean \|gap\| on grasp frames (m) | 0.0100 | 0.0035 |
| max \|gap\| on grasp frames (m) | 0.0738 | 0.0245 |
| grasp frames floating >5cm | 2 | 0 |

0 gap = closest fingertip/palm exactly on the object surface. Released frames (hand far from the object, >0.12 m) are left untouched. Correction is a smoothed radial translation of the grasping hand; finger configuration is preserved.

Outputs: `hand_keypoints_3d_grasp.csv`, `grasp_gap_before_after.png`.