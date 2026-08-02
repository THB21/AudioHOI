# VLM Interval Candidate Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate stable and occlusion-aware object trajectories in the same Stage 4 run, let Qwen choose per bounded risk interval, and compose an isolated candidate without changing reliable visible frames.

**Architecture:** Stage 4 solves a stable problem and a generic amodal-observation challenger from the same typed inputs. Both are rendered before Qwen runs. A forced-choice interval gate verifies evidence hashes and selects `stable`, `challenger`, or `reject`; the composer copies challenger SE(3) only inside approved intervals with in-interval translation blending and quaternion SLERP. The accepted publisher remains unchanged and no historical accepted pose is read.

**Tech Stack:** Python, NumPy, SciPy least-squares and quaternion math, OpenCV/Pillow evidence rendering, Qwen3-VL forced-choice gate, existing GenericSequenceExecutor and provenance artifacts.

---

### Task 1: Add typed interval-selection decisions and state composition

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/gates/interval_candidate_selection.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/gates/__init__.py`

- [ ] **Step 1: Define immutable decision records**

Add `IntervalCandidateDecision` with `query_id`, `start_frame`, `end_frame`, `label`, evidence/response hashes, and provider/model fields. Add `IntervalCandidateSelectionLedger` with `status`, `blocking`, and ordered decisions. Labels are exactly `keep_stable`, `use_occlusion_challenger`, `reject_both`, and `unclear`.

- [ ] **Step 2: Implement verified VLM-result loading**

Read `vlm/stage4/vlm_queries.csv` and `qwen_raw_results.json`, retain only `interval_candidate_selection_check`, verify each evidence SHA-256, and map missing/tampered evidence to a blocking `reject_both`. A valid `unclear` remains nonblocking and means stable.

```python
def load_interval_candidate_selection(
    *, result_dir: Path,
) -> IntervalCandidateSelectionLedger:
    """Load hash-verified forced-choice interval decisions."""
```

- [ ] **Step 3: Implement SE(3) interval composition**

Add a pure function that starts from stable states, applies challenger states only for `use_occlusion_challenger`, and keeps both boundary transitions inside the selected interval. Translation uses a linear blend; quaternion uses shortest-path SLERP and is normalized. `unclear` and `keep_stable` do not change any state. `reject_both` returns no composed result.

```python
def compose_interval_selected_result(
    stable: GenericSequenceSolveResult,
    challenger: GenericSequenceSolveResult,
    ledger: IntervalCandidateSelectionLedger,
    *,
    transition_frames: int,
) -> GenericSequenceSolveResult | None:
    ...
```

- [ ] **Step 4: Add composition diagnostics**

Return provenance containing parent attempt IDs, selected intervals, per-frame source, maximum quaternion-norm error, boundary translation steps, and the exact list of frames changed from stable.

- [ ] **Step 5: Run focused diagnostics**

Use an inline diagnostic with synthetic seven-DOF rigid states. Expected output: `unclear_changed_frames=[]`, selected frames remain inside the declared interval, quaternion norm error is below `1e-9`, and frames outside the interval are exactly equal.

- [ ] **Step 6: Commit Task 1**

Commit only the new gate/composer files with message `feat: add interval candidate selection contract`.

### Task 2: Produce stable and challenger attempts from one Stage 4 input contract

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`
- Modify: `scripts/shared/generic_contact_pipeline/stages/main/stage4_contact_refine.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [ ] **Step 1: Add an explicit mask-observation override**

Extend `prepare_capability_object_problem` with an optional override that can set `use_artifact_bbox` to `false` for stable preparation and `completed_only` for challenger preparation. Apply it to an in-memory copy of configuration; never mutate `CaseProfile.data`.

```python
def prepare_capability_object_problem(
    *,
    profile: CaseProfile,
    result_dir: Path,
    repository_root: Path,
    body_models_root: Path,
    factor_arbitration_mode: str,
    mask_artifact_bbox_policy_override: bool | str | None = None,
) -> CapabilityObjectProblemPreparation:
    ...
```

- [ ] **Step 2: Solve two isolated attempts**

Refactor the repeated Stage 4 solve/projection sequence into a local helper. Run `stable` with no completed-mask bbox replacement. When a validated amodal completion interval exists, run `occlusion_challenger` with `completed_only`. Both attempts use the same StateSpec, measurements, contact constraints, factor vocabulary, optimizer, and case-independent projection policies.

- [ ] **Step 3: Write role-specific attempt provenance**

Under `generic_stage4_candidate/attempt_roles.json`, record stable/challenger attempt IDs, preparation hashes, completion-manifest hash, and `baseline_pose_read=false`. Do not inspect or copy accepted `object_pose.csv`.

- [ ] **Step 4: Render both attempts before VLM**

Publish each attempt to its own candidate CSV and render:

```text
generic_stage4_candidate/stable/object_pose_candidate.csv
generic_stage4_candidate/stable/object_only/overlay.mp4
generic_stage4_candidate/occlusion_challenger/object_pose_candidate.csv
generic_stage4_candidate/occlusion_challenger/object_only/overlay.mp4
```

- [ ] **Step 5: Verify attempt isolation**

Confirm both pose files exist, have 240 rows and unit quaternions, share a StateSpec hash, have different attempt IDs, and leave the accepted output hash unchanged.

- [ ] **Step 6: Commit Task 2**

Commit the Stage 4 dual-attempt implementation with message `feat: generate stable and occlusion challenger attempts`.

### Task 3: Add forced-choice A/B evidence and Qwen labels

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/gates/vlm.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/vlm_query_templates.yaml`
- Modify: `scripts/shared/generic_contact_pipeline/stages/gates/stage_vlm_qwen.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/gates/vlm_gates.py`

- [ ] **Step 1: Register the new Stage 4 query type**

Add `interval_candidate_selection_check` to the Stage 4 query vocabulary with forced choices `keep_stable`, `use_occlusion_challenger`, `reject_both`, and `unclear`. Its windows come only from validated bounded amodal-completion intervals.

- [ ] **Step 2: Materialize synchronized pairwise evidence**

For each representative risk frame, create a two-row temporal evidence image. The top row is labeled `A STABLE`; the bottom row is labeled `B OCCLUSION CHALLENGER`. Each row shows frames `representative-1`, `representative`, and `representative+1`, cropped identically from the corresponding current-attempt render. Include the original visible mask/rail/contact overlays without reading a stale accepted render.

- [ ] **Step 3: Use a narrow forced-choice prompt**

The prompt asks which candidate better follows visible suitcase pixels, rigid rail/body geometry, continuous hand-handle relation, floor support, and neighboring-frame motion. It explicitly says to choose `unclear` when evidence cannot distinguish the candidates and forbids estimating pose or weights.

- [ ] **Step 4: Map labels conservatively**

Map `keep_stable` and `use_occlusion_challenger` to pass, `unclear` to no update, and `reject_both` to blocking rejection. Do not downweight any factor for `unclear`.

- [ ] **Step 5: Run Qwen on current candidate evidence**

Run only `interval_candidate_selection_check` using the configured Qwen environment and local model. Expected artifacts are query CSV, evidence PNG, raw JSON result, normalized result CSV, and stage decision with verified hashes.

- [ ] **Step 6: Commit Task 3**

Commit evidence and gate changes with message `feat: add VLM interval candidate arbitration`.

### Task 4: Compose and publish only the selected isolated candidate

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/stages/main/stage4_contact_refine.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/optimization.py`

- [ ] **Step 1: Apply decisions only after Qwen rerun**

On the first Stage 4 pass, stable remains the publication candidate and both renders are prepared. On the automatic post-VLM Stage 4 refresh, load the interval ledger and compose only decisions backed by current evidence hashes.

- [ ] **Step 2: Preserve stable on uncertainty**

If every decision is `keep_stable` or `unclear`, reuse stable states exactly. If any decision is `reject_both` or evidence is invalid, block publication and preserve both attempts.

- [ ] **Step 3: Write selected-attempt provenance**

Write the composed attempt through `write_isolated_sequence_attempt`, including both parent IDs and the interval-selection ledger hash. The candidate pose records per-frame source as stable, transition, or challenger.

- [ ] **Step 4: Re-run hard metrics and render the composed result**

Evaluate the existing object publication gate, render the current composed candidate, and ensure Stage 4 VLM evidence routing points to this current render rather than Stage 5 or accepted output.

- [ ] **Step 5: Verify the suitcase intervals**

Report exact changed-frame sets relative to stable. Expected: no changed states in frames 1–145 or 173–240; only VLM-approved frames inside 146–172 may differ. Report quaternion norms and translation/angular boundary steps before presenting the video.

- [ ] **Step 6: Commit Task 4**

Commit composition and publication integration with message `feat: compose VLM-selected object trajectory intervals`.

### Task 5: Improve the challenger without case-specific solver logic

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/factors/activation.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [ ] **Step 1: Diagnose challenger residuals only inside 146–172**

Compare stable and challenger point, mask, line, contact-relative-velocity, support, and temporal residual blocks per frame. Record which generic factors improve or regress; do not tune from the final render alone.

- [ ] **Step 2: Restrict amodal-derived factor activation to approved frames**

Ensure completed-mask bbox/principal-axis measurements carry source provenance and an active mask matching only approved completion frames. Their residuals must be zero outside those frames, preventing global visual-factor contamination.

- [ ] **Step 3: Keep temporal coupling local**

Solve the challenger as a bounded local subproblem with stable boundary states fixed at frames 145 and 173. Optimize only frames 146–172 using the same generic factor executor. This removes the need to splice historical trajectories and prevents temporal leakage into frame 121 or the visible tail.

- [ ] **Step 4: Preserve physical constraints**

Keep rigid-body quaternion normalization, hand-handle contact relative velocity, support/penetration, line reprojection when rails are visible, and temporal velocity/acceleration active. No suitcase-specific optimizer, case-name branch, or hand-authored pose is allowed.

- [ ] **Step 5: Render and compare risk windows**

Render frames 145–173 plus a montage containing 1, 106, 121, 145, 152, 159, 166, 172, 173, and 190. Reject the challenger if it improves occlusion but changes stable frames outside the local subproblem or introduces larger boundary jumps.

- [ ] **Step 6: Commit Task 5**

Commit local challenger improvements with message `feat: localize occlusion challenger optimization`.

### Task 6: Final diagnostics and handoff

**Files:**
- Modify: `docs/current_generic_pipeline_mainline.md`

- [ ] **Step 1: Run source validation**

Run Python compilation for modified modules and `git diff --check`. Do not run or add the repository-wide pytest suite.

- [ ] **Step 2: Verify object-only boundary**

Confirm `human_state_optimized=false`, no downstream human refinement is invoked, and human skeleton remains read-only render context.

- [ ] **Step 3: Verify accepted-output isolation**

Compare accepted output hashes from before and after all attempts. Expected: unchanged while the automatic hard gate remains blocking.

- [ ] **Step 4: Document the production status**

Record that interval arbitration is production-wired, list Qwen decisions and current hard-gate blockers, and state honestly whether the challenger is visually better. Do not claim zero-shot completion while hard publication gaps remain.

- [ ] **Step 5: Commit Task 6**

Commit documentation and final diagnostics with message `docs: record interval arbitration status`.
