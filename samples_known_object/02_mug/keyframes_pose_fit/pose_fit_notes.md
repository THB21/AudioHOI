# Mug 3D Pose Fitting Keyframes

These frames are selected to constrain a rigid canonical mug proxy rather than redrawing the handle per frame.

- `001_pickup_contact_start`: human hand contact fixes object-side handle/contact region.
- `034_side_motion_handle_occluded`: side-on motion; handle projection can collapse toward a line.
- `067_drinking_mid`: cup near mouth; constrains body pose under occlusion.
- `131_annotated_contact_region`: existing object-side contact mask frame.
- `155_putdown_visible_handle`: clear side/handle, good for C-shape handle fitting.
- `198_table_stable_handle`: stable visible handle after putdown.
- `239_final_static_handle`: check no post-putdown jitter.

Next step: generate or manually provide clean mug image(s), then fit Articraft canonical body/handle/rim/bottom to these keyframes.
