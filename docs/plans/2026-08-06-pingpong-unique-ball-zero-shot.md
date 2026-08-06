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
5. **in progress — Render and audit the two-object result**
   - Render the reconstructed sphere and Articraft paddle together in overlay and camera-space 3D.
   - Verify unique ball identity, rigid paddle geometry, paddle-hand continuity and ball/paddle event proximity.
   - The joint overlay, local camera-space 3D view and full rigid-mesh impact audit are complete.
6. **pending — Commit the correction locally**
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
- `output/pingpong_mask_direction_megapose_twist_candidate/paddle_pose.csv` — observed-rigid paddle pose with an immutable `right_hand` pivot, SAM2 direction constraint and MegaPose face-twist constraint
- `output/pingpong_mask_direction_megapose_twist_candidate/articraft_solid/object_only/overlay.mp4` — corrected Articraft paddle overlay without the rejected double-paddle offset
- `output/pingpong_cotracker_planar_pnp_production_candidate/paddle_pose.csv` — isolated MegaPose-initialized planar-PnP candidate using persistent CoTracker surface correspondences and right-hand handle reprojection
- `output/pingpong_cotracker_planar_pnp_candidate/articraft_solid/object_only/overlay.mp4` — current endpoint-review overlay; not yet canonical
- `output/pingpong_free_flight_vlm_gate/free_flight_vlm_decisions.json` — production Qwen forced-choice contact semantics for numerically detected free-flight reversals
- `output/pingpong_planar_pnp_exact_grasp_recomputed_contacts/object_pose.csv` — Full sphere trajectory using Audio impact endpoints and only VLM-approved free-flight repairs
- `output/pingpong_planar_pnp_exact_grasp_no_vlm_contacts/object_pose.csv` — controlled no-VLM pose-only ablation with identical Audio and paddle inputs
- `output/pingpong_planar_pnp_exact_grasp_no_audio_contacts/object_pose.csv` — controlled no-audio pose-only ablation with an empty event stream
- `output/pingpong_unified_ablation_evaluation/ablation_table.csv` — repository-standard unified ablation metrics for Full/no-Audio/no-VLM
- `output/pingpong_unified_ablation_evaluation/ablation_delta_table.csv` — repository-standard deltas relative to Full
- `output/pingpong_unified_ablation_evaluation/ablation_report.md` — method/provenance audit and causal interpretation

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
- 2026-08-06: persistent-grasp correction — the prior observed-rigid builder used the right palm only as weak depth/image evidence and allowed a 160+ mean handle gap of about 0.23 m. Two exact-pivot revisions were rejected: independently fitting each mask produced quaternion/twist flips, while over-regularizing rotation produced a visible double-paddle offset (38 px at frame 240). The audited exact-pivot candidate fixed `grasp_site_id=right_hand` for all 240 frames and forbade hand switching, but its numerical-zero 3D gap was later shown to be incompatible with the known-size paddle projection because GVHMR hand depth is uncertain. Human state remained read-only and the exact-depth constraint was not promoted.
- 2026-08-06: handle-ambiguity correction — review showed that SAM2 segmented only the blade in thin-edge frames and omitted the occluded handle; all 26 CoTracker queries were consequently blade-only. Low-confidence MegaPose twist (for example IoU 0.158 near frame 63) could therefore rotate a correctly tracked thin edge into a broad face while still satisfying the center factor. The generic rigid builder now supports a one-dimensional `mask_silhouette_twist` factor: the immutable right-hand pivot and SAM2 centroid fix the grasp axis, while a temporally regularized silhouette-shape search chooses the remaining rotation from the mask width/aspect evidence. This resolves the ambiguity without asking VLM for a continuous quaternion; VLM remains appropriate only for a discrete red-face/black-face or handle-end semantic choice.
- 2026-08-06: endpoint root-cause correction — the exact 3D hand pivot above was rejected after review of frames 71, 96 and 161. At each swing endpoint the palm occludes part of the blade, so a visible-only SAM2 mask is not an amodal scale/depth observation. More importantly, the read-only GVHMR hand depth differs from the known-size paddle depth by roughly 0.3–0.7 m; treating it as zero-variance 3D truth systematically enlarged the overlay and pulled the twist away from the tracked blade.
- 2026-08-06: generic planar-track candidate — 24 blade tracks were lifted into asset-local coordinates from the reliable frame-2 MegaPose pose and then consumed as per-frame planar PnP correspondences. The front-red-face semantic normal and temporal continuity select the planar ambiguity, the right hand supplies a strong handle reprojection term, and frames without enough tracks are interpolated. This is a geometry-capability path rather than a ping-pong branch. It recovers 236/240 PnP frames with track reprojection 2.79 px median / 5.28 px p95; rotation step is 5.15° median / 13.44° p95 / 17.22° maximum. Endpoint review now shows broad red face at frame 71, true thin edge at frame 96 and broad red face at frame 161. Canonical paddle and ball-contact outputs remain unchanged pending full-video approval.
- 2026-08-06: Audio/VLM isolation — the numerical solver first proposes an interval only when progress toward the next Audio-defined event reverses and a non-contact turn exceeds 35°. Qwen does not judge XYZ or dynamics; it classifies the visible contact at the peak-turn evidence as paddle, wall, floor, none or unclear. Only `floor_contact` outside the declared interaction vocabulary or `no_visible_contact` authorizes a physical-arc repair. For frames 157–200 Qwen identified the generated extra floor bounce at frames 176/178 with confidence 0.99. Full reduces the interior maximum turn from 176.12° to 4.45°; the same-input no-VLM run retains 176.12°. The no-audio run has no impact endpoints, reaches only 1/7 intended paddle reversals and has 264.3 mm mean gap to the Full paddle-contact surfaces, versus numerical zero for Full.
- 2026-08-06: unified ablation evaluation — the repository-standard evaluator now consumes the same three materialized object-stage variants. Full/no-Audio/no-VLM overlay IoU is 0.6441/0.6452/0.6617; the small visual-only advantage of no-VLM is reported rather than hidden. Full/no-Audio/no-VLM contact gap is approximately 0/264.3/0 mm and contact proxy is 1.0/0.0051/1.0. Full versus no-VLM high-speed recall is 0.9286 versus 0.6429 and oversmooth rate is 0.0714 versus 0.3571. Full has one non-event spike, no-Audio has twelve and no-VLM has two. Under the existing contact/penetration tradeoff formula, Full/no-Audio/no-VLM scores are 1.0/0.0711/1.0, so no-Audio fails the unified final gate while Full and no-VLM pass. The evaluator table is hash-stable across two consecutive reruns.
