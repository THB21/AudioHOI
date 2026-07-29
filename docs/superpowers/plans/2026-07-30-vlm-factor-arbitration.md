# Generic VLM Factor Arbitration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make profile-driven VLM forced-choice decisions alter generic visual/contact factor activation tiers with complete provenance, then validate the change in an isolated stick attempt.

**Architecture:** Existing InteractionState activation remains the base policy. A new factor-arbitration module parses normalized VLM decisions, maps only generic factor kinds to `active/downweighted` overrides, and merges them monotonically at the factor-compiler boundary. The production solver consumes only merged activation intervals and never reads case identity or free-form VLM text.

**Tech Stack:** Python dataclasses, CSV/JSON artifacts, existing Qwen3-VL provider, SciPy generic sequence executor, existing pytest and Phase-0 verifiers.

---

### Task 1: Typed Arbitration Ledger and Monotonic Activation Merge

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/factors/vlm_arbitration.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/factors/__init__.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/factors/compiler.py`

- [ ] **Step 1: Define generic decision types and validation**

Add `FactorGateDecision` and `FactorArbitrationLedger` dataclasses. Labels are exactly `visual_observation_reliable`, `contact_relation_reliable`, `both_consistent`, and `unclear`; intervals must be positive; affected factor ids must be nonempty; decisions must carry provider/model/evidence and prompt/response hashes.

- [ ] **Step 2: Implement monotonic merge**

Implement `merge_factor_activation_ledger(base, arbitration)` by expanding base intervals per frame, applying only `active -> downweighted`, preserving `inactive`, recompressing intervals, and appending `vlm_factor_arbitration:<decision_id>` provenance. Reject attempts to activate a lower base status or target a missing factor id.

- [ ] **Step 3: Expose the types and compiler input**

Export the new API and add an optional arbitration ledger parameter to `build_compiled_factor_ledger()`. Merge before constructing `CompiledFactor`, so downstream runtime code remains unchanged.

- [ ] **Step 4: Run existing factor verification**

Run:

```bash
python -m pytest -q tests/test_factors.py tests/test_sequence_solver_shadow.py
```

Expected: existing tests pass; no new test file is created.

- [ ] **Step 5: Commit**

```bash
git add scripts/shared/generic_contact_pipeline/core/factors
git commit -m "feat: add generic VLM factor arbitration"
```

### Task 2: Profile-Driven Reliability Query and Evidence

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/configs/vlm_query_templates.yaml`
- Modify: `scripts/shared/generic_contact_pipeline/core/gates/vlm.py`
- Modify: `scripts/shared/generic_contact_pipeline/stages/gates/stage_vlm_qwen.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/gates/vlm_gates.py`

- [ ] **Step 1: Add the forced-choice query vocabulary**

Register `constraint_reliability_check` for Stage 4 with choices `visual_observation_reliable|contact_relation_reliable|both_consistent|unclear`. Add it to the generic Stage 4 query list and Qwen pass-label handling without assigning pose-repair authority.

- [ ] **Step 2: Remove object-specific Stage 4 wording**

Replace “brown predicted stick / green tracked stick” with geometry-neutral labels: predicted object geometry, visual measurement, and read-only human contact site. Use typed supplemental line measurements when present, without branching on case name.

- [ ] **Step 3: Add query provenance columns**

Extend query artifacts with interval, evidence hash, and generic factor-role columns. Preserve these fields in Qwen raw results; VLM result rows continue to contain normalized forced-choice labels only.

- [ ] **Step 4: Limit reliability queries to auditable conflict windows**

Select frames from existing risk/contact artifacts, coalesce neighboring frames into intervals, and create one temporal evidence package per interval. Other Stage 4 query policies remain unchanged.

- [ ] **Step 5: Verify query generation without model inference**

Generate Stage 4 queries for the existing stick result and inspect that no query text contains `stick`, basketball, mug, chair, or a case name; verify every reliability row has a nonempty evidence hash and interval.

- [ ] **Step 6: Commit**

```bash
git add scripts/shared/generic_contact_pipeline/configs/vlm_query_templates.yaml scripts/shared/generic_contact_pipeline/core/gates scripts/shared/generic_contact_pipeline/stages/gates/stage_vlm_qwen.py
git commit -m "feat: add generic VLM reliability queries"
```

### Task 3: Convert VLM Results into Solver Arbitration

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/gates/factor_arbitration.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/gates/__init__.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/attempt_artifacts.py`

- [ ] **Step 1: Parse evaluated reliability results**

Implement a loader that joins Stage 4 query rows with Qwen raw results by query id, verifies evidence hashes, normalizes labels, maps generic visual and contact factor kinds to factor ids from the compiled plan, and emits a `FactorArbitrationLedger`.

- [ ] **Step 2: Enforce fail-closed semantics**

Treat missing, invalid, or `unclear` results as both groups `downweighted`; mark the ledger as blocking for canonical promotion. Treat `vlm_mode=none` as `not_evaluated` and do not claim VLM execution.

- [ ] **Step 3: Feed arbitration at the adapter boundary**

Load the ledger in `prepare_capability_object_problem()` from the selected result directory and pass it into factor compilation / problem preparation without adding object identity dispatch. Record the ledger hash and evaluated status in preparation artifacts.

- [ ] **Step 4: Preserve attempt provenance**

Write the full arbitration ledger into `attempts/<id>/vlm_gates.json`, add its hash to `status.json`, and ensure `continuous_pose_override=false`.

- [ ] **Step 5: Run existing solver verification**

Run:

```bash
python -m pytest -q tests/test_factors.py tests/test_sequence_solver_shadow.py tests/test_factor_residual_evaluator.py
python scripts/shared/generic_contact_pipeline/tools/verify_sequence_problem_shadow.py
python scripts/shared/generic_contact_pipeline/tools/verify_sequence_solver_diagnostics.py
```

Expected: all existing checks pass and no canonical output is written.

- [ ] **Step 6: Commit**

```bash
git add scripts/shared/generic_contact_pipeline/core/gates scripts/shared/generic_contact_pipeline/core/solver
git commit -m "feat: compile VLM gates into generic solver factors"
```

### Task 4: Execute Qwen Arbitration and Isolated Stick Re-solve

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/tools/prepare_generic_object_problem.py`
- Modify: `docs/interaction_state_conditioned_generic_solver_plan.md`

- [ ] **Step 1: Add explicit CLI control**

Add `--vlm-arbitration {off,required}`. In `required` mode, refuse solving unless an evaluated arbitration ledger exists; pass the ledger to attempt evidence and keep `--allow-accepted-write` independent.

- [ ] **Step 2: Generate conflict queries**

Run Stage 4 query generation for `stick --result-name benchmark_vlm_qwen`, then check the reliability query count, intervals, evidence hashes, and absence of object-specific wording.

- [ ] **Step 3: Run the configured Qwen environment**

Invoke `/home/yang/miniconda3/envs/qwen-vl/bin/python` with the existing local model at `/mnt/hdd/AudioHOI/models/modelscope/Qwen/Qwen3-VL-8B-Instruct`, evaluating the reliability queries. Do not overwrite other stage results with a debug-limited run.

- [ ] **Step 4: Re-solve into a fresh isolated directory**

Run the generic object problem tool with `--vlm-arbitration required`, `--solve`, no accepted-write authorization, and a fresh `/tmp` candidate directory. Report termination, endpoint reprojection p95, contact p95/max, activation changes, and arbitration provenance.

- [ ] **Step 5: Compare pose/render evidence**

Use existing object-only and GVHMR-skeleton render tooling against the new candidate. Do not optimize human state. Keep the candidate blocked if hard geometry or regression gates fail.

- [ ] **Step 6: Update the maintained plan and commit**

Record exact commands, metrics, result hashes, remaining gaps, and the object-only boundary in `docs/interaction_state_conditioned_generic_solver_plan.md`, then commit all verified changes locally without pushing.

### Task 5: Final Existing-Suite Verification

**Files:**
- No new files.

- [ ] **Step 1: Run the related existing suite**

```bash
python -m pytest -q tests/test_interaction_state_ir.py tests/test_factors.py tests/test_factor_residual_evaluator.py tests/test_sequence_solver_shadow.py
```

- [ ] **Step 2: Run artifact verifiers**

```bash
python scripts/shared/generic_contact_pipeline/tools/verify_sequence_problem_shadow.py
python scripts/shared/generic_contact_pipeline/tools/verify_sequence_solver_diagnostics.py
python scripts/shared/generic_contact_pipeline/tools/verify_candidate_sandbox.py
```

- [ ] **Step 3: Check boundaries and worktree**

Run `git diff --check`, confirm no case-name branching was added under core solver/factors/state/geometry, confirm no canonical accepted output changed, and confirm the original worktree remains untouched.

