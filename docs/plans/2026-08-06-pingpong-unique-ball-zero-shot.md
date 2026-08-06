# Ping-pong unique-ball zero-shot execution plan

## Goal and boundary

Run the held-out `14_pingpong_wall` sample from Stage 0 through the generic object solver and object render. Preserve exactly one physical ping-pong-ball identity even where the generated video visibly duplicates the ball. The paddle is an observed rigid tool, the wall is static, and GVHMR is read-only object evidence. No downstream human reconstruction code is changed.

## Invariants

- One optimized sphere state per frame; duplicate visual candidates never become additional states.
- VLM supplies a discrete identity/visibility/contact decision, never XYZ or a quaternion.
- Audio supplies paddle/wall event timing and may gate direction changes, never direct position.
- Audio-gated impact frames are the only planned exceptions to the strong temporal-acceleration factor.
- The implementation is capability-driven (`sphere`, observed tool, static plane, identity candidates), with no ping-pong-specific solver.
- Baseline/no-audio/no-VLM artifacts remain separate from the full result.

## Steps

1. **completed — Materialize and audit Stage 0 evidence**
   - Extract frames/audio, run available mask/track/depth/body/audio tasks, and identify duplicate-ball intervals.
   - Record all generated artifacts and hashes.
2. **completed — Add generic single-identity observation arbitration**
   - Represent multiple sphere detections as candidates for one entity.
   - Compile temporal, event-order, visibility and VLM identity gates into candidate selection.
3. **completed — Reconstruct the observed rigid paddle**
   - Materialize the approved Articraft paddle as a second object entity with a per-frame rigid pose.
   - Use paddle image evidence as the primary observation and read-only GVHMR wrist evidence only as an initialization/occlusion aid.
   - Joint audio/motion events remain typed alternately as paddle-face and practice-wall impacts; they gate timing only and deliberately do not invent metric contact XYZ.
4. **completed — Run ablations and full generic solver**
   - Preserve pure solver/no-audio/no-VLM pose CSVs.
   - Run full audio+VLM trajectory and produce provenance ledgers.
5. **completed — Render and audit the two-object result**
   - Render the reconstructed sphere and Articraft paddle together in overlay and camera-space 3D.
   - Verify unique ball identity, rigid paddle geometry, paddle-hand continuity and ball/paddle event proximity.
   - The joint overlay, local camera-space 3D view and full rigid-mesh impact audit are complete.
6. **completed — Commit the correction locally**
   - Commit only source/config/manifest changes and compact accepted evidence; do not push unless separately authorized.

## Produced files

- `samples_known_object/14_pingpong_wall/articraft/` — approved rigid paddle asset (existing)
- `output/pingpong_articraft_review/` — approved paddle preview (existing)
- `samples_known_object/14_pingpong_wall/results/pingpong_stage0_audit/object_pose.csv` — accepted full trajectory, 240 rows and one object state per frame
- `samples_known_object/14_pingpong_wall/results/pingpong_no_audio/object_pose.csv` — accepted no-audio pose-only ablation
- `samples_known_object/14_pingpong_wall/results/pingpong_no_vlm/ablation_pose.csv` — blocked, uncorrected no-VLM pose-only ablation
- `samples_known_object/14_pingpong_wall/results/renders/pingpong_stage0_audit/object_only/overlay.mp4` — final full object overlay
- `output/pingpong_full_review/final_duplicate_windows_sheet_v2.jpg` — frames 36, 139–147 and 225–228 duplicate-window audit
- `scripts/shared/generic_contact_pipeline/configs/assets/table_tennis_paddle_fixed_rigid.json` — fixed-rigid mesh, semantic faces and handle feature contract
- `samples_known_object/14_pingpong_wall/articraft/megapose/fixed_rigid_asset_mm.ply` — MegaPose provider mesh in millimetres
- `samples_known_object/14_pingpong_wall/results/pingpong_stage0_audit/paddle_observed/megapose/rigid_pose_hypotheses.jsonl` — 50 hypotheses at 10 automatically selected visible keyframes
- `samples_known_object/14_pingpong_wall/results/pingpong_stage0_audit/paddle_observed/paddle_pose.csv` — 240-frame observed-rigid SE(3) trajectory
- `output/pingpong_rigid_contact_candidate/object_pose.csv` — sphere trajectory with rigid-face impact constraints
- `output/pingpong_rigid_contact_candidate/rigid_face_contacts.csv` — per-impact surface point, gap and relative-normal-motion audit
- `samples_known_object/14_pingpong_wall/results/renders/pingpong_rigid_contact_candidate/ball/overlay.mp4` — current ball+paddle object overlay candidate
- `samples_known_object/14_pingpong_wall/results/renders/pingpong_rigid_contact_candidate/ball/camera3d.mp4` — joint sphere and observed-rigid paddle camera-space 3D view
- `samples_known_object/14_pingpong_wall/results/renders/pingpong_rigid_contact_articraft_solid/ball/overlay.mp4` — final joint overlay using the Articraft URDF visual materials
- `samples_known_object/14_pingpong_wall/results/renders/pingpong_rigid_contact_articraft_solid/ball/camera3d.mp4` — final solid Articraft paddle and metric-radius sphere 3D view
- `output/pingpong_persistent_grasp_candidate/paddle_pose.csv` — observed-rigid paddle pose with an exact persistent palm-to-handle point constraint
- `samples_known_object/14_pingpong_wall/results/renders/pingpong_persistent_grasp_fixed_view/with_human/camera3d.mp4` — fixed-view camera-space 3D containing the read-only GVHMR skeleton, Articraft paddle and sphere

## Execution notes

- 2026-08-06: user explicitly retained the Gemini duplicate-ball failure so VLM can resolve identity while the final reconstruction remains one physically real ball.
- 2026-08-06: first Stage 0 attempt reached CoTracker but exhausted the 10 GB GPU at full 1280 px width. The generic tracker work width was reduced to 512 while retaining original-coordinate outputs; completed frame/SAM2 artifacts remain hash-validated inputs.
- 2026-08-06: Stage 0 found 13 ambiguous/multi-candidate frames. The full run keeps one sphere entity and VLM makes a discrete candidate decision under a 24 px persistent-track hard validator. The no-VLM candidate is blocked by the production hard gate.
- 2026-08-06: 26 raw audio peaks were reduced to the declared 10 impacts by a generic audio-peak plus visual-direction-change gate. The resulting typed timeline contains 10 impact states and no inherited floor/rolling state.
- 2026-08-06: full Stage 4 passed with objective 162.38 versus 184.33 for no-audio. In duplicate windows, full reprojection error is 0.42 px mean / 1.50 px p95, while no-VLM is 92.58 px mean / 178.25 px p95.
- 2026-08-06: correction after review — the first accepted render reconstructed only the ball. The Articraft paddle was present only as an asset/semantic entity, so paddle pose reconstruction and the two-object render were reopened and are required before this case is complete.
- 2026-08-06: paddle correction — SAM2 produced 240 paddle-face masks, CoTracker produced 26 persistent rigid points per frame, and MegaPose produced five hypotheses at each of ten automatically selected keyframes. A generic observed-rigid sequence builder selected the red-face-consistent branch, used the read-only right hand as bounded handle/depth evidence, and published a separate 240-frame `paddle_pose.csv`; no human state was optimized.
- 2026-08-06: robust translation filtering reduced paddle translation-step p95 from 0.318 m/frame to 0.116 m/frame. The current two-object overlay and camera-space 3D view contain the reconstructed Articraft paddle and the unique-ball full trajectory.
- 2026-08-06: contact correction — joint audio peaks, sphere-to-paddle-mask proximity and a refractory gate identified seven actual paddle impacts at frames 17, 47, 74, 117, 151, 200 and 233. The generated clip contains seven apparent paddle/wall cycles despite requesting five in the generation prompt; the reconstruction follows the observed video rather than the requested count.
- 2026-08-06: each paddle impact now targets the closest surface of the actual Articraft paddle mesh, has zero sphere-surface gap, approaches before impact and separates after impact. An earlier analytic ellipse approximation was rejected after it showed up to 9 mm disagreement at the real mesh rim. The paddle remains anchored by MegaPose/SAM2/CoTracker/read-only hand evidence; contact refinement changes only the sphere trajectory and does not move human or paddle state to manufacture contact.
- 2026-08-06: final-render correction — the solver still consumes the Articraft-derived collision mesh, while final overlay and camera-space 3D now consume the original Articraft URDF visual components and materials. The previous sparse mesh wireframe is no longer the accepted visual output; the red rubber, black rubber, blade wood and handle are rendered as solid components, and the ball is drawn with its metric 20 mm radius.
- 2026-08-06: persistent-grasp correction — the prior observed-rigid builder used the right palm only as weak depth/image evidence and allowed a 160+ mean handle gap of about 0.23 m. The production candidate now enforces the handle feature and read-only GVHMR palm as one 3D pivot on every frame, while rotating the paddle about that pivot so its blade center remains aligned with the SAM2 observation. The resulting maximum grasp gap is numerical zero and the blade-center reprojection is 2.98 px mean / 8.29 px p95. A fixed-view 3D render with the read-only skeleton is published for review.
