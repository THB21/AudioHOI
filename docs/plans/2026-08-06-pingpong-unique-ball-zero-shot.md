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
3. **completed for the object-only run — Build observed paddle and wall evidence**
   - Keep the approved Articraft paddle descriptor as an observed rigid-tool asset; do not create a second optimized object state.
   - The approved paddle remains an observed rigid tool. Joint audio/motion events are typed alternately as paddle-face and practice-wall impacts; they gate timing only and deliberately do not invent metric contact XYZ.
4. **completed — Run ablations and full generic solver**
   - Preserve pure solver/no-audio/no-VLM pose CSVs.
   - Run full audio+VLM trajectory and produce provenance ledgers.
5. **completed — Render and audit the object result**
   - Render overlay and 3D object trajectory.
   - Verify unique identity, continuity and scale; audit paddle/wall event classes without inventing metric contact XYZ.
6. **completed — Commit locally**
   - Commit only source/config/manifest changes and compact accepted evidence; do not push unless separately authorized.

## Produced files

- `samples_known_object/14_pingpong_wall/articraft/` — approved rigid paddle asset (existing)
- `output/pingpong_articraft_review/` — approved paddle preview (existing)
- `samples_known_object/14_pingpong_wall/results/pingpong_stage0_audit/object_pose.csv` — accepted full trajectory, 240 rows and one object state per frame
- `samples_known_object/14_pingpong_wall/results/pingpong_no_audio/object_pose.csv` — accepted no-audio pose-only ablation
- `samples_known_object/14_pingpong_wall/results/pingpong_no_vlm/ablation_pose.csv` — blocked, uncorrected no-VLM pose-only ablation
- `samples_known_object/14_pingpong_wall/results/renders/pingpong_stage0_audit/object_only/overlay.mp4` — final full object overlay
- `output/pingpong_full_review/final_duplicate_windows_sheet_v2.jpg` — frames 36, 139–147 and 225–228 duplicate-window audit

## Execution notes

- 2026-08-06: user explicitly retained the Gemini duplicate-ball failure so VLM can resolve identity while the final reconstruction remains one physically real ball.
- 2026-08-06: first Stage 0 attempt reached CoTracker but exhausted the 10 GB GPU at full 1280 px width. The generic tracker work width was reduced to 512 while retaining original-coordinate outputs; completed frame/SAM2 artifacts remain hash-validated inputs.
- 2026-08-06: Stage 0 found 13 ambiguous/multi-candidate frames. The full run keeps one sphere entity and VLM makes a discrete candidate decision under a 24 px persistent-track hard validator. The no-VLM candidate is blocked by the production hard gate.
- 2026-08-06: 26 raw audio peaks were reduced to the declared 10 impacts by a generic audio-peak plus visual-direction-change gate. The resulting typed timeline contains 10 impact states and no inherited floor/rolling state.
- 2026-08-06: full Stage 4 passed with objective 162.38 versus 184.33 for no-audio. In duplicate windows, full reprojection error is 0.42 px mean / 1.50 px p95, while no-VLM is 92.58 px mean / 178.25 px p95.
