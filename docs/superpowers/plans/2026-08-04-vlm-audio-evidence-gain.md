# VLM and Audio Evidence Gain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the suitcase full run use real VLM orientation/occlusion evidence and real interval-level audio motion evidence inside the same generic object solver, then materialize fair full/no-VLM/no-audio/vision-only results whose visible differences are caused only by removed evidence.

**Architecture:** Stage 0 produces typed `AudioEventIR` intervals and uncertainty-triggered VLM `SemanticRelationIR` records. The interaction estimator merges those records with vision/contact measurements, and the generic factor compiler activates semantic orientation/topology and audio motion/freeze residuals without any `case_name` branch or direct VLM pose output. Four isolated ablation result directories share the same state, geometry, initialization, weights, solver budget, publisher, and renderer; only evidence streams are disabled.

**Tech Stack:** Python 3, NumPy, SciPy, OpenCV, ffmpeg, Qwen VLM provider, existing `generic_contact_pipeline` typed IR/factor/runtime framework, YAML/JSONL/CSV provenance.

---

## Scope and invariants

- Object pose is the only optimized state. GVHMR joints may be read as observations and rendered for HOI inspection, but are never optimized or published as a human result.
- No object-specific optimizer and no `case_name` branch may be added under `core/solver`, `core/factors`, `core/state`, or `core/geometry`.
- VLM may emit only forced-choice semantic relations and confidence. It may not emit XYZ, quaternion, Euler angles, factor weights, or an accepted pose.
- Audio may emit event/interval labels, confidence, and envelope statistics. It may not emit object coordinates or orientations.
- The full/no-VLM/no-audio/vision-only runs use identical geometry, seed, factor weights, robust losses, solver bounds, iteration budget, and publication gates.
- Ablations remove their evidence and dependent factors; they do not add noise, alter observations, weaken unrelated factors, or copy a hand-edited pose.
- The existing canonical `object_pose.csv` is not overwritten. Every attempt is written to a new result directory until explicit visual approval.
- Per user instruction, add no pytest files and do not run repository-wide pytest. Verification uses real artifacts, focused Python assertions, compilation checks, hashes, metrics, and rendered videos.

## File map

### New files

- `scripts/shared/generic_contact_pipeline/core/audio_events/envelope.py` — extract interval-level sustained motion, silence, onset, offset, short tug, and seam-click evidence from `audio.wav`.
- `scripts/shared/generic_contact_pipeline/core/gates/semantic_relations.py` — define typed VLM semantic relation records, uncertainty-triggered query selection, strict forced-choice parsing, and provenance writing.
- `scripts/shared/generic_contact_pipeline/core/solver/semantic_factor_inputs.py` — build solver inputs for face visibility, facing relation, heading topology, and audio motion envelope factors.
- `scripts/shared/generic_contact_pipeline/tools/run_suitcase_evidence_ablations.py` — materialize four isolated variants with one shared command contract; the implementation must remain profile/capability-driven and contain no pose edits.
- `scripts/shared/generic_contact_pipeline/tools/evaluate_evidence_gain.py` — compute causal VLM/audio metrics, provenance completeness, hashes, and a concise Markdown/CSV report.

### Modified files

- `scripts/shared/generic_contact_pipeline/core/audio_events/types.py` — extend event type and interval fields while retaining v1 compatibility.
- `scripts/shared/generic_contact_pipeline/core/audio_events/adapters.py` — load both peak-only v1 and interval-aware v2 artifacts.
- `scripts/shared/generic_contact_pipeline/tools/run_audio_event_extract.py` — atomically write v2 events plus frame envelope.
- `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py` — declare/validate the new audio artifacts and hashes.
- `scripts/shared/generic_contact_pipeline/core/gates/vlm.py` — register semantic-orientation query generation and invoke the existing Qwen provider.
- `scripts/shared/generic_contact_pipeline/core/interaction/estimator.py` — merge interval audio and semantic relations into orthogonal visibility/contact/motion state.
- `scripts/shared/generic_contact_pipeline/core/interaction/timeline.py` — persist semantic/audio IDs and interval provenance.
- `scripts/shared/generic_contact_pipeline/core/factors/types.py` — add generic semantic/audio factor kinds.
- `scripts/shared/generic_contact_pipeline/core/factors/activation.py` — state-dependent activation for the new factor kinds.
- `scripts/shared/generic_contact_pipeline/core/factors/compiler.py` — compile the new factor kinds with explicit input IDs and gate provenance.
- `scripts/shared/generic_contact_pipeline/core/solver/residual_inputs.py` — include the new typed inputs in sequence factor input construction.
- `scripts/shared/generic_contact_pipeline/core/solver/factor_residuals.py` — implement differentiable, bounded semantic/audio residuals.
- `scripts/shared/generic_contact_pipeline/core/solver/problem_factory.py` — pass the new inputs into the unchanged generic sequence problem.
- `scripts/shared/generic_contact_pipeline/core/solver/residual_boundary.py` — map new residual references to production evaluators.
- `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py` — load evidence artifacts and prepare generic factors from asset/profile capabilities.
- `scripts/shared/generic_contact_pipeline/stages/main/stage4_contact_refine.py` — publish candidate evidence ledgers and enforce fair candidate selection.
- `scripts/shared/generic_contact_pipeline/configs/assets/suitcase_fixed_rigid.json` — declare face normals/vertices and semantic roles as asset data.
- `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml` — enable generic semantic/audio capabilities and uncertainty policy.
- `scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/ablation_registry.py` — register the four object-solver evidence variants without touching downstream human evaluation.

### Existing dirty files

Before each commit, stage only the paths named by that task. Do not reset, clean, or bulk-stage the worktree; existing solver changes and generated outputs belong to the current work.

## Task 1: Freeze the shared baseline and causal run contract

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/run_suitcase_evidence_ablations.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/ablation_registry.py`

- [ ] **Step 1: Add an explicit four-variant object evidence registry**

Add variants whose only differences are evidence switches:

```python
SUITCASE_EVIDENCE_VARIANTS = (
    MethodVariant("full", "suitcase_evidence_full", [], audio=True, vlm="qwen", llm="none"),
    MethodVariant("no_vlm", "suitcase_evidence_no_vlm", ["disable_vlm_semantic_evidence"], audio=True, vlm="none", llm="none"),
    MethodVariant("no_audio", "suitcase_evidence_no_audio", ["disable_audio_events"], audio=False, vlm="qwen", llm="none"),
    MethodVariant("vision_only", "suitcase_evidence_vision_only", ["disable_vlm_semantic_evidence", "disable_audio_events"], audio=False, vlm="none", llm="none"),
)
```

Do not alter `DEFAULT_VARIANTS`, because its downstream HOI evaluator is outside this object-only task.

- [ ] **Step 2: Implement a dry-run command manifest before execution**

The runner must resolve one base profile and emit `output/suitcase_evidence_ablations/run_matrix.json` containing, for every variant:

```python
{
    "case": "suitcase_drag",
    "result_name": variant.result_name,
    "vlm_mode": variant.vlm,
    "llm_mode": "none",
    "ablation_flags": sorted(variant.ablation_flags),
    "shared_config_sha256": shared_config_sha256,
    "shared_initializer_sha256": shared_initializer_sha256,
    "shared_geometry_sha256": shared_geometry_sha256,
    "canonical_write_allowed": False,
    "human_state_optimized": False,
}
```

The runner accepts `--execute`, `--variant`, `--from-stage`, and `--to-stage`; without `--execute` it only writes the matrix. It must call the existing pipeline entry point rather than duplicating solver logic.

- [ ] **Step 3: Verify the run matrix without running a solver**

Run:

```bash
python scripts/shared/generic_contact_pipeline/tools/run_suitcase_evidence_ablations.py
python - <<'PY'
import json
from pathlib import Path
p = Path('output/suitcase_evidence_ablations/run_matrix.json')
d = json.loads(p.read_text())
assert {r['variant'] for r in d['variants']} == {'full','no_vlm','no_audio','vision_only'}
assert len({r['shared_config_sha256'] for r in d['variants']}) == 1
assert all(not r['canonical_write_allowed'] for r in d['variants'])
assert all(not r['human_state_optimized'] for r in d['variants'])
print('run matrix causal contract: PASS')
PY
```

Expected: `run matrix causal contract: PASS`.

- [ ] **Step 4: Commit only the registry and runner**

```bash
git add scripts/shared/generic_contact_pipeline/core/evaluation/final_hoi/ablation_registry.py scripts/shared/generic_contact_pipeline/tools/run_suitcase_evidence_ablations.py
git commit -m "feat: define causal object evidence ablations"
```

## Task 2: Produce interval-aware AudioEventIR from the real soundtrack

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/audio_events/envelope.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/audio_events/types.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/audio_events/adapters.py`
- Modify: `scripts/shared/generic_contact_pipeline/tools/run_audio_event_extract.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/preprocess/registry.py`

- [ ] **Step 1: Extend the typed event contract without breaking v1 rows**

Add event kinds `MOTION_ONSET`, `MOTION_OFFSET`, `SHORT_TUG`, and `SEAM_CLICK`. Extend `AudioEvent` with optional `start_frame`, `end_frame`, `start_time_s`, `end_time_s`, `snr`, and `band_profile`; default interval bounds to the peak frame/time in `__post_init__`. Validate ordered finite bounds and preserve all old constructor call sites through defaults.

Use this interval predicate everywhere downstream:

```python
def contains_frame(self, frame: int) -> bool:
    start = self.start_frame if self.start_frame is not None else self.frame
    end = self.end_frame if self.end_frame is not None else self.frame
    return start <= frame <= end
```

- [ ] **Step 2: Implement generic envelope extraction**

In `envelope.py`, decode mono PCM, compute per-frame RMS, spectral flux, and high-frequency ratio, normalize by robust median/MAD, and use hysteresis plus a minimum duration to create intervals. Keep thresholds in a named `AudioEnvelopeConfig` loaded from the case profile; do not inspect `case_name`.

The public interface is:

```python
@dataclass(frozen=True)
class AudioEnvelopeConfig:
    window_ms: float
    hop_ms: float
    motion_on_z: float
    motion_off_z: float
    min_motion_ms: float
    min_silence_ms: float
    impulse_z: float

def extract_audio_evidence(audio_path: Path, fps: float, config: AudioEnvelopeConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return typed event rows and one envelope row per video frame."""
```

An event interval is labeled `sustained_motion` only when envelope energy remains above the off threshold for the configured duration. A high-flux isolated event inside a motion interval is `seam_click`; a high-energy short interval is `short_tug`; low-energy intervals are `silence`. Write confidence from distance to the hysteresis thresholds, not a hand-authored label.

- [ ] **Step 3: Write v2 artifacts atomically and keep peak provenance**

`run_audio_event_extract.py` writes:

- `results/events/audio_events.csv` with v2 interval columns and all old peak columns;
- `results/events/audio_envelope.csv` with `frame,time_s,rms_z,flux_z,hf_ratio,motion_probability,source`;
- `results/events/audio_event_manifest.json` with audio SHA-256, extractor config, FPS, row counts, and output hashes.

The adapter returns schema `audio_events_v2` when interval columns are present and `audio_peak_events_v1` otherwise.

- [ ] **Step 4: Regenerate the real suitcase audio evidence**

Run:

```bash
python scripts/shared/generic_contact_pipeline/tools/run_audio_event_extract.py \
  --sample-dir samples_known_object/15_suitcase_drag
python - <<'PY'
import csv, json
from pathlib import Path
root = Path('samples_known_object/15_suitcase_drag/results/events')
rows = list(csv.DictReader((root/'audio_events.csv').open()))
env = list(csv.DictReader((root/'audio_envelope.csv').open()))
m = json.loads((root/'audio_event_manifest.json').read_text())
assert rows and env
assert {'sustained_motion','silence'} <= {r['event_type'] for r in rows}
assert all(int(r['start_frame']) <= int(r['end_frame']) for r in rows)
assert m['schema_version'] == 2 and m['audio_sha256']
print('audio v2 real artifact: PASS', len(rows), len(env))
PY
```

Expected: a non-empty frame envelope and at least one sustained-motion and silence interval. If the real soundtrack cannot satisfy this, stop with the measured distribution and adjust only the profile-declared generic thresholds; never invent intervals.

- [ ] **Step 5: Commit typed audio production**

```bash
git add scripts/shared/generic_contact_pipeline/core/audio_events scripts/shared/generic_contact_pipeline/tools/run_audio_event_extract.py scripts/shared/generic_contact_pipeline/core/preprocess/registry.py
git commit -m "feat: produce interval audio evidence"
```

## Task 3: Produce uncertainty-triggered SemanticRelationIR with Qwen

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/gates/semantic_relations.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/gates/vlm.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/assets/suitcase_fixed_rigid.json`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [ ] **Step 1: Define the semantic record and strict vocabulary**

Implement:

```python
@dataclass(frozen=True)
class SemanticRelation:
    relation_id: str
    start_frame: int
    end_frame: int
    subject_entity: str
    predicate: str
    object_entity: str
    label: str
    confidence: float
    source_query_id: str
    evidence_sha256: str
    prompt_sha256: str
    response_sha256: str

ALLOWED_LABELS = {
    "visible_face": {"grasp_side_wide", "opposite_wide", "side_left", "side_right", "unclear"},
    "facing_relation": {"grasp_side_toward_human", "grasp_side_away", "side_on", "unclear"},
    "turn_direction_screen": {"counterclockwise", "clockwise", "stationary", "unclear"},
    "visibility": {"visible", "partial", "human_occluded", "absent", "unclear"},
    "grasp_state": {"active", "released", "unclear"},
}
```

Reject unknown labels, out-of-range confidence, missing hashes, and responses containing coordinate/quaternion/angle fields.

- [ ] **Step 2: Declare asset semantics as geometry data**

In `suitcase_fixed_rigid.json`, add named broad-face and side-face vertex sets/normals, handle/rail/wheel feature IDs, and the relation vocabulary. The two wide faces must be named by asset topology (`grasp_side_wide`, `opposite_wide`), not screen color or frame number.

- [ ] **Step 3: Select query windows from uncertainty, not a fixed case schedule**

Implement a generic scorer using only typed measurements and current pose hypotheses:

```python
score = (
    0.25 * mask_area_drop
    + 0.20 * rail_visibility_drop
    + 0.15 * wheel_visibility_drop
    + 0.20 * pose_hypothesis_disagreement
    + 0.10 * human_overlap
    + 0.10 * hand_handle_inconsistency
)
```

Select local maxima over the profile-declared threshold with non-maximum suppression. Each evidence package contains the original temporal strip, visible mask, projected asset face/rail/wheel overlays for the top pose hypotheses, and hand/handle marker. Do not include manual red/yellow drawings or manually supplied labels.

- [ ] **Step 4: Add forced-choice Qwen queries**

Generate one structured query per selected window asking all five predicates and requiring strict JSON. Reuse the existing Qwen provider and hashing. Persist:

- `vlm/stage4/semantic_queries.jsonl`;
- `vlm/stage4/semantic_raw_responses.jsonl`;
- `vlm/stage4/semantic_relations.jsonl`;
- `vlm/stage4/semantic_relation_manifest.json`.

`--vlm-mode none` and `disable_vlm_semantic_evidence` must produce an empty, valid relation stream with status `disabled_by_ablation`.

- [ ] **Step 5: Run real Qwen and validate that no pose is emitted**

Run the existing pipeline through the stage that materializes Stage 4 VLM queries with `--vlm-mode qwen --llm-mode none`. Then run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('samples_known_object/15_suitcase_drag/results/suitcase_evidence_full/vlm/stage4/semantic_relations.jsonl')
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
assert rows
for r in rows:
    blob = json.dumps(r).lower()
    assert not any(k in blob for k in ('quaternion','euler','pose_xyz','rotation_deg'))
    assert r['label'] != 'unclear' or 0.0 <= r['confidence'] <= 1.0
    assert r['evidence_sha256'] and r['response_sha256']
print('semantic relation IR: PASS', len(rows))
PY
```

Expected: non-empty hashed semantic relations and no continuous-pose fields.

- [ ] **Step 6: Commit semantic evidence production**

```bash
git add scripts/shared/generic_contact_pipeline/core/gates/semantic_relations.py scripts/shared/generic_contact_pipeline/core/gates/vlm.py scripts/shared/generic_contact_pipeline/configs/assets/suitcase_fixed_rigid.json scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml
git commit -m "feat: produce vlm semantic relation evidence"
```

## Task 4: Merge audio and VLM evidence into InteractionStateIR

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/interaction/estimator.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/interaction/timeline.py`

- [ ] **Step 1: Index interval evidence across every covered frame**

Replace peak-only audio lookup with interval membership. Add a semantic relation index keyed by frame. Resolve conflicts by evidence confidence and typed precedence only:

```python
if semantic.visibility == "human_occluded":
    visibility_state = VisibilityState.OCCLUDED
if grasp_state == "active" and visibility_state is VisibilityState.OCCLUDED:
    contact_state = ContactState.OCCLUDED_HOLD
if audio_motion and support_active:
    motion_mode = MotionMode.SUPPORTED_MOVING
elif audio_silence and support_active and visual_speed_is_low:
    motion_mode = MotionMode.SUPPORTED_STATIC
```

Audio silence may strengthen freeze only when visual speed is also low. Audio motion may prevent an erroneous freeze but must not create translation without visual/contact support.

- [ ] **Step 2: Preserve evidence IDs and conflict decisions**

Populate `audio_event_ids`, `semantic_relation_ids`, and provenance for every state. Write `interaction_state_metrics.json` counters for evidence-covered frames, conflicts, conflict winners, and ablation-disabled streams.

- [ ] **Step 3: Materialize and inspect the full timeline**

Run Stage 2/interaction materialization, then:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('samples_known_object/15_suitcase_drag/results/suitcase_evidence_full/interaction/interaction_timeline.jsonl')
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
assert rows
assert any(r['audio_event_ids'] for r in rows)
assert any(r['semantic_relation_ids'] for r in rows)
assert all(r['target_entity_id'] for r in rows)
print('interaction evidence merge: PASS', len(rows))
PY
```

Expected: both evidence streams cover real frame intervals and remain traceable by ID.

- [ ] **Step 4: Commit interaction production**

```bash
git add scripts/shared/generic_contact_pipeline/core/interaction/estimator.py scripts/shared/generic_contact_pipeline/core/interaction/timeline.py
git commit -m "feat: condition interaction state on semantic and audio evidence"
```

## Task 5: Add generic semantic and audio factors

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/core/solver/semantic_factor_inputs.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/factors/types.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/factors/activation.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/factors/compiler.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/residual_inputs.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/factor_residuals.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/problem_factory.py`
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/residual_boundary.py`

- [ ] **Step 1: Add factor kinds and typed inputs**

Add generic kinds:

```python
FACE_VISIBILITY_INEQUALITY = "face_visibility_inequality"
FACING_RELATION = "facing_relation"
HEADING_TOPOLOGY = "heading_topology"
AUDIO_MOTION_ENVELOPE = "audio_motion_envelope"
```

Define immutable inputs carrying frame intervals, asset-local face normals/points, camera parameters, target labels, confidence, evidence IDs, and per-frame active masks. They must not contain case names or target pose values.

- [ ] **Step 2: Implement bounded semantic residuals**

Use smooth hinge residuals:

```python
def smooth_hinge(x: np.ndarray, margin: float, beta: float = 20.0) -> np.ndarray:
    return np.logaddexp(0.0, beta * (margin - x)) / beta
```

- `face_visibility_inequality`: selected face normal/extent must rank ahead of incompatible face hypotheses under projection; `unclear` yields zero activation.
- `facing_relation`: the asset-declared grasp-side face direction must agree with the human-to-object bearing only during active/occluded grasp.
- `heading_topology`: penalize the wrong sign of signed yaw increments over a relation interval; do not prescribe absolute yaw or degrees.
- All semantic residuals are confidence-scaled and robust-loss bounded.

- [ ] **Step 3: Implement interval audio residuals**

`audio_motion_envelope` operates on tangential displacement magnitude:

```python
speed_t = np.linalg.norm((translation[t] - translation[t - 1]) - normal * np.dot(translation[t] - translation[t - 1], normal))
```

- sustained motion: weak lower-bound/rank residual so legitimate non-uniform movement is not flattened;
- silence plus low visual speed: freeze residual;
- onset/offset: permit a local speed transition;
- seam click: lower acceleration smoothing locally without imposing a position;
- no audio evidence: no audio residual.

- [ ] **Step 4: Compile factors solely from state/evidence/capability**

The compiler uses relation/audio IDs in `input_ids` and `gate_provenance`. Activation rules:

- visible + confident face relation: semantic orientation active;
- occluded + persistent grasp: facing and heading topology active, point/mask visual downweighted;
- released/unclear grasp: human-facing factor inactive;
- supported moving + sustained audio: audio envelope active and static freeze inactive;
- supported static + silence + low visual speed: static freeze active.

- [ ] **Step 5: Verify factor math with focused assertions**

Run a standalone Python assertion script that constructs synthetic generic rigid states and checks:

```python
assert facing_residual(correct_heading) < facing_residual(flipped_heading)
assert topology_residual(counterclockwise_sequence, "counterclockwise") < topology_residual(clockwise_sequence, "counterclockwise")
assert motion_residual(nonzero_tangential_motion, "sustained_motion") < motion_residual(frozen_motion, "sustained_motion")
assert motion_residual(frozen_motion, "silence") < motion_residual(drifting_motion, "silence")
```

Expected: all assertions pass and every residual is finite.

- [ ] **Step 6: Compile changed modules**

```bash
python -m py_compile \
  scripts/shared/generic_contact_pipeline/core/solver/semantic_factor_inputs.py \
  scripts/shared/generic_contact_pipeline/core/factors/types.py \
  scripts/shared/generic_contact_pipeline/core/factors/activation.py \
  scripts/shared/generic_contact_pipeline/core/factors/compiler.py \
  scripts/shared/generic_contact_pipeline/core/solver/residual_inputs.py \
  scripts/shared/generic_contact_pipeline/core/solver/factor_residuals.py \
  scripts/shared/generic_contact_pipeline/core/solver/problem_factory.py \
  scripts/shared/generic_contact_pipeline/core/solver/residual_boundary.py
```

Expected: exit code 0.

- [ ] **Step 7: Commit generic factor execution**

```bash
git add scripts/shared/generic_contact_pipeline/core/factors scripts/shared/generic_contact_pipeline/core/solver/semantic_factor_inputs.py scripts/shared/generic_contact_pipeline/core/solver/residual_inputs.py scripts/shared/generic_contact_pipeline/core/solver/factor_residuals.py scripts/shared/generic_contact_pipeline/core/solver/problem_factory.py scripts/shared/generic_contact_pipeline/core/solver/residual_boundary.py
git commit -m "feat: solve with semantic and audio evidence factors"
```

## Task 6: Wire evidence into the production generic Stage 4 path

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py`
- Modify: `scripts/shared/generic_contact_pipeline/stages/main/stage4_contact_refine.py`
- Modify: `scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml`

- [ ] **Step 1: Load typed evidence and build factor inputs**

In `prepare_capability_object_problem`, load audio/semantic artifacts from the current `result_dir`, build inputs from descriptor-declared capabilities, and attach them to `SequenceFactorInputs`. Missing optional evidence produces an empty input set; missing evidence in `required` mode is an explicit preparation error. No fallback may read canonical/final pose.

- [ ] **Step 2: Configure finite, auditable weights**

Add profile entries for the four generic factors with one fixed weight per factor and the existing activation tiers. Qwen may select labels/gates but cannot write these values. Keep the current visual, depth, contact, support, temporal, robust loss, bounds, and solver budget unchanged across all four variants.

- [ ] **Step 3: Persist a factor-to-evidence ledger**

Every candidate writes:

- `generic_stage4_candidate/factor_ledger.json`;
- `generic_stage4_candidate/semantic_factor_residuals.csv`;
- `generic_stage4_candidate/audio_factor_residuals.csv`;
- `generic_stage4_candidate/evidence_consumption.json`.

`evidence_consumption.json` records each evidence ID, factor IDs, frame interval, activation status, weight source, gate source, and residual summary. Disabled streams must be marked `disabled_by_ablation`, not silently missing.

- [ ] **Step 4: Enforce object-only and accepted-output boundaries**

Add runtime assertions before candidate publication:

```python
assert prepared.case_dispatch_used is False
assert prepared.baseline_pose_read is False
assert prepared.human_state_optimized is False
assert candidate_path.resolve() != profile.object_pose_csv.resolve()
```

The renderer may read GVHMR to display the skeleton, but Stage 4 must not modify any GVHMR/body artifact.

- [ ] **Step 5: Run one full isolated Stage 4 candidate**

```bash
python -m scripts.shared.generic_contact_pipeline.run_pipeline \
  --case suitcase_drag \
  --from-stage stage0 --to-stage stage4 \
  --result-name suitcase_evidence_full \
  --vlm-mode qwen --llm-mode none
```

Expected: solver success, non-empty semantic/audio residual ledgers, `case_dispatch_used=false`, `baseline_pose_read=false`, `human_state_optimized=false`, and no canonical pose hash change.

- [ ] **Step 6: Commit production wiring**

```bash
git add scripts/shared/generic_contact_pipeline/core/solver/capability_production_problem.py scripts/shared/generic_contact_pipeline/stages/main/stage4_contact_refine.py scripts/shared/generic_contact_pipeline/configs/cases/suitcase_drag.yaml
git commit -m "feat: consume multimodal evidence in generic stage4"
```

## Task 7: Materialize all four fair ablations

**Files:**
- Modify: `scripts/shared/generic_contact_pipeline/tools/run_suitcase_evidence_ablations.py`

- [ ] **Step 1: Execute the four variants through the same pipeline**

```bash
python scripts/shared/generic_contact_pipeline/tools/run_suitcase_evidence_ablations.py \
  --execute --from-stage stage0 --to-stage stage4
```

The runner must execute full, no-VLM, no-audio, and vision-only into distinct result directories. It records command, environment, start/end time, exit status, input hashes, factor config hash, initializer hash, geometry hash, and candidate pose hash.

- [ ] **Step 2: Prove causal parity before comparing results**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('output/suitcase_evidence_ablations/run_matrix.json').read_text())
done = [r for r in d['variants'] if r['status'] == 'complete']
assert len(done) == 4
for key in ('shared_config_sha256','shared_initializer_sha256','shared_geometry_sha256','solver_budget_sha256'):
    assert len({r[key] for r in done}) == 1, key
assert next(r for r in done if r['variant']=='no_vlm')['semantic_evidence_count'] == 0
assert next(r for r in done if r['variant']=='no_audio')['audio_evidence_count'] == 0
assert next(r for r in done if r['variant']=='full')['semantic_evidence_count'] > 0
assert next(r for r in done if r['variant']=='full')['audio_evidence_count'] > 0
print('causal ablation parity: PASS')
PY
```

Expected: `causal ablation parity: PASS`.

- [ ] **Step 3: Reject fake or hand-edited differences**

Search the production path:

```bash
rg -n "case_name\s*==|suitcase.*frame|frame.*suitcase|manual_pose|pose_override|inject_noise" \
  scripts/shared/generic_contact_pipeline/core/solver \
  scripts/shared/generic_contact_pipeline/core/factors \
  scripts/shared/generic_contact_pipeline/core/state \
  scripts/shared/generic_contact_pipeline/core/geometry
```

Expected: no new case/frame-specific solver branch, manual pose override, or injected noise. Asset/profile declarations are allowed; core solver branches are not.

- [ ] **Step 4: Commit runner completion**

```bash
git add scripts/shared/generic_contact_pipeline/tools/run_suitcase_evidence_ablations.py
git commit -m "feat: materialize fair suitcase evidence ablations"
```

## Task 8: Evaluate visible and numerical evidence gains

**Files:**
- Create: `scripts/shared/generic_contact_pipeline/tools/evaluate_evidence_gain.py`

- [ ] **Step 1: Implement metrics tied to each claim**

Compute, without VLM judging its own answer:

- VLM: face/topology consistency against held-out audit labels, rail/wheel reprojection, hand-handle gap only while grasp is active, occluded-interval orientation continuity, and wrong-face interval count.
- Audio: complete-stop drift, start/stop timing error, non-uniform motion preservation, seam-transition acceleration, and false movement during silence.
- Shared hard gates: finite pose, maximum translation/rotation step, support penetration, support gap, depth rank, heading reversal count, and canonical hash preservation.

The held-out audit labels may encode intervals and discrete face/turn labels only; they may not contain pose values or be loaded by Stage 0–4.

- [ ] **Step 2: Write auditable tables and report**

Write:

- `output/suitcase_evidence_ablations/evidence_gain_metrics.csv`;
- `output/suitcase_evidence_ablations/evidence_gain_intervals.csv`;
- `output/suitcase_evidence_ablations/evidence_gain_report.md`;
- `output/suitcase_evidence_ablations/evidence_gain_manifest.json`.

The manifest hashes every pose, evidence stream, ledger, metric table, and video.

- [ ] **Step 3: Enforce the approved quantitative exit conditions**

```bash
python scripts/shared/generic_contact_pipeline/tools/evaluate_evidence_gain.py \
  --root output/suitcase_evidence_ablations \
  --sample-dir samples_known_object/15_suitcase_drag
python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('output/suitcase_evidence_ablations/evidence_gain_manifest.json').read_text())
assert m['full_hard_gates_pass'] is True
assert m['canonical_pose_unchanged'] is True
assert m['vlm_wrong_face_or_topology_reduction'] >= 0.50
assert m['audio_complete_stop_drift_reduction'] >= 0.50
assert m['manual_pose_edits'] == 0
assert m['case_dispatch_used'] is False
assert m['human_state_optimized'] is False
print('evidence gain acceptance: PASS')
PY
```

If a threshold fails, report the measured failure and return to the responsible evidence/factor task. Do not alter the no-VLM/no-audio outputs to manufacture the delta.

- [ ] **Step 4: Commit evaluation tooling**

```bash
git add scripts/shared/generic_contact_pipeline/tools/evaluate_evidence_gain.py
git commit -m "feat: evaluate causal vlm and audio gains"
```

## Task 9: Render object results for visual approval

**Files:**
- Modify only if required by a generic renderer bug: `scripts/shared/generic_contact_pipeline/components/render/scenes/generic_urdf_scene.py`

- [ ] **Step 1: Render the full result and comparison results**

Run Stage 5/6 or the existing generic Articraft renderer for all four candidate pose CSV files. Required outputs per variant:

- `review/object_only/overlay.mp4`;
- `review/object_only/camera3d.mp4`;
- `review/with_human/overlay.mp4` where the skeleton is read-only context;
- a contact/semantic overlay showing query windows, relation labels, rails, wheels, handle, and support points.

Do not render no-VLM/no-audio if the user’s earlier “pose.csv only” restriction is still intended for the basketball baseline; for this suitcase evidence experiment, render all four because the approved B criterion is visible comparison.

- [ ] **Step 2: Inspect required intervals and whole-sequence smoothness**

Extract contact sheets for the entire sequence plus uncertainty/occlusion windows. Verify visually and numerically:

- broad/side face progression is coherent;
- rail and four-wheel topology remain rigid;
- support points remain on the floor within the declared rolling tolerance;
- no quaternion branch flip or adjacent-frame jitter;
- full preserves irregular movement and complete stops;
- no-VLM visibly loses semantic orientation during ambiguous/occluded intervals;
- no-audio visibly over-smooths starts/stops or drifts during silence while retaining identical observations.

- [ ] **Step 3: Verify no canonical overwrite and no human mutation**

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('output/suitcase_evidence_ablations')
m = json.loads((root/'evidence_gain_manifest.json').read_text())
assert m['canonical_pose_sha256_before'] == m['canonical_pose_sha256_after']
assert m['gvhmr_sha256_before'] == m['gvhmr_sha256_after']
for name in ('full','no_vlm','no_audio','vision_only'):
    assert (root/name/'review/object_only/overlay.mp4').is_file()
    assert (root/name/'review/object_only/camera3d.mp4').is_file()
print('publication boundary: PASS')
PY
```

Expected: `publication boundary: PASS`.

- [ ] **Step 4: Commit only a generic renderer fix if one was necessary**

```bash
git add scripts/shared/generic_contact_pipeline/components/render/scenes/generic_urdf_scene.py
git commit -m "fix: render generic semantic evidence overlays"
```

Skip this commit if no renderer source changed. Generated videos remain untracked evidence unless the repository’s existing artifact policy explicitly tracks their manifest.

## Task 10: Final audit and local checkpoint

**Files:**
- Modify: `docs/superpowers/specs/2026-08-04-vlm-audio-evidence-gain-design.md`

- [ ] **Step 1: Record measured results, not intended behavior**

Append an implementation status section with commit IDs, result directories, factor/evidence counts, numerical gains, failed gates if any, and direct paths to the four videos. Do not claim zero-shot completion; this task proves generic multimodal evidence consumption on the suitcase held-out scenario.

- [ ] **Step 2: Run the complete focused verification**

```bash
python -m py_compile $(git diff --name-only --diff-filter=ACM HEAD~8..HEAD -- '*.py')
python scripts/shared/generic_contact_pipeline/tools/run_suitcase_evidence_ablations.py
python scripts/shared/generic_contact_pipeline/tools/evaluate_evidence_gain.py \
  --root output/suitcase_evidence_ablations \
  --sample-dir samples_known_object/15_suitcase_drag
git diff --check
git status --short
```

Expected: compilation and artifact assertions pass, `git diff --check` is clean, and `git status` shows only known pre-existing/generated paths.

- [ ] **Step 3: Commit the measured design status locally**

```bash
git add docs/superpowers/specs/2026-08-04-vlm-audio-evidence-gain-design.md
git commit -m "docs: record multimodal evidence gain results"
```

- [ ] **Step 4: Stop before promotion or remote push**

Present the full/no-VLM/no-audio/vision-only pose hashes, metrics, and local video paths for user approval. Do not copy a candidate to canonical output, merge branches, push a remote, or modify downstream human/ablation-evaluation code.

## Self-review against the approved design

- Spec coverage: uncertainty-triggered forced-choice VLM, interval audio, typed IR, interaction-state gates, one generic solver, fair four-way ablation, object-only boundary, provenance, numerical metrics, visible videos, and no canonical overwrite each map to Tasks 1–10.
- No hidden labels: manual orientation hints appear only as held-out evaluation labels in Task 8 and never enter Stage 0–4.
- No fake gains: Task 7 proves shared solver/config hashes and explicitly rejects injected noise or pose overrides.
- Type consistency: `SemanticRelation`, `AudioEvent`, factor-kind names, evidence IDs, and artifact paths are introduced before their consumers and are used consistently.
- Scope consistency: downstream human refinement and existing final HOI ablation evaluation are not modified; GVHMR is read-only context.
- Placeholder scan: the implementation and verification behavior is explicit; there are no unresolved or deferred implementation markers.
