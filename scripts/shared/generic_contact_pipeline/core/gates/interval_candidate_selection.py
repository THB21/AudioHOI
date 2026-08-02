from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


QUERY_TYPE = "interval_candidate_selection_check"
INTERVAL_SELECTION_LABELS = frozenset(
    {"keep_stable", "use_occlusion_challenger", "reject_both", "unclear"}
)


@dataclass(frozen=True)
class IntervalCandidateDecision:
    query_id: str
    start_frame: int
    end_frame: int
    normalized_label: str
    evidence_sha256: str
    response_sha256: str
    provider: str
    model: str
    evidence_valid: bool


@dataclass(frozen=True)
class IntervalCandidateSelectionLedger:
    status: str
    blocking: bool
    decisions: tuple[IntervalCandidateDecision, ...] = ()


@dataclass(frozen=True)
class IntervalCompositionOutcome:
    result: Any | None
    provenance: dict[str, object]


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def load_interval_candidate_selection(
    *,
    result_dir: Path,
) -> IntervalCandidateSelectionLedger:
    """Load hash-verified forced-choice interval decisions."""

    stage_dir = result_dir / "vlm" / "stage4"
    queries = [
        row for row in _rows(stage_dir / "vlm_queries.csv")
        if row.get("query_type") == QUERY_TYPE
    ]
    raw_path = stage_dir / "qwen_raw_results.json"
    if not queries or not raw_path.is_file():
        return IntervalCandidateSelectionLedger(status="not_evaluated", blocking=False)
    payload = json.loads(raw_path.read_text())
    if not isinstance(payload, list):
        return IntervalCandidateSelectionLedger(status="invalid_results", blocking=True)
    raw_by_id = {
        str(row.get("query_id", "")): row
        for row in payload
        if isinstance(row, Mapping) and row.get("query_type") == QUERY_TYPE
    }
    decisions: list[IntervalCandidateDecision] = []
    blocking = False
    for query in queries:
        query_id = str(query.get("query_id", ""))
        raw = raw_by_id.get(query_id)
        evidence_path = Path(
            str(query.get("input_render_path") or query.get("input_image_path") or "")
        )
        expected_hash = str(query.get("evidence_sha256", ""))
        evidence_valid = (
            evidence_path.is_file()
            and bool(expected_hash)
            and _sha256_bytes(evidence_path.read_bytes()) == expected_hash
        )
        label = str((raw or {}).get("label", "reject_both"))
        provider = str((raw or {}).get("provider", "missing_provider"))
        model = str((raw or {}).get("model", "missing_model"))
        evaluated = (
            raw is not None
            and evidence_valid
            and label in INTERVAL_SELECTION_LABELS
            and provider not in {"", "missing_provider"}
            and model not in {"", "missing_model"}
        )
        if not evaluated:
            label = "reject_both"
            blocking = True
        if label == "reject_both":
            blocking = True
        response_payload = raw if raw is not None else {
            "query_id": query_id,
            "status": "missing_result",
        }
        decisions.append(
            IntervalCandidateDecision(
                query_id=query_id,
                start_frame=int(query.get("start_frame") or query.get("frame") or 0),
                end_frame=int(query.get("end_frame") or query.get("frame") or 0),
                normalized_label=label,
                evidence_sha256=expected_hash,
                response_sha256=_canonical_hash(response_payload),
                provider=provider,
                model=model,
                evidence_valid=evidence_valid,
            )
        )
    return IntervalCandidateSelectionLedger(
        status="evaluated" if decisions else "not_evaluated",
        blocking=blocking,
        decisions=tuple(decisions),
    )


def _normalized_shortest_path_blend(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    left = np.asarray(q0, dtype=float).copy()
    right = np.asarray(q1, dtype=float).copy()
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    dot = float(np.clip(left @ right, -1.0, 1.0))
    if dot < 0.0:
        right *= -1.0
        dot = -dot
    if dot > 0.9995:
        blended = (1.0 - alpha) * left + alpha * right
        return blended / np.linalg.norm(blended)
    angle = float(np.arccos(dot))
    sine = float(np.sin(angle))
    blended = (
        np.sin((1.0 - alpha) * angle) / sine * left
        + np.sin(alpha * angle) / sine * right
    )
    return blended / np.linalg.norm(blended)


def compose_interval_selected_result(
    stable: Any,
    challenger: Any,
    ledger: IntervalCandidateSelectionLedger,
    *,
    transition_frames: int,
    quaternion_groups: Sequence[tuple[int, int, int, int]] = ((3, 4, 5, 6),),
) -> IntervalCompositionOutcome:
    """Compose selected challenger intervals without modifying stable frames elsewhere."""

    if stable.frames != challenger.frames or len(stable.states) != len(challenger.states):
        raise ValueError("stable and challenger results must share frame alignment")
    if any(len(a) != len(b) for a, b in zip(stable.states, challenger.states)):
        raise ValueError("stable and challenger results must share state width")
    if transition_frames < 0:
        raise ValueError("transition_frames must be nonnegative")
    if ledger.blocking:
        return IntervalCompositionOutcome(
            result=None,
            provenance={
                "schema_version": 1,
                "status": "blocked",
                "stable_attempt_id": stable.solve_attempt_id,
                "challenger_attempt_id": challenger.solve_attempt_id,
                "ledger": asdict(ledger),
            },
        )
    frames = tuple(int(frame) for frame in stable.frames)
    frame_to_index = {frame: index for index, frame in enumerate(frames)}
    stable_states = np.asarray(stable.states, dtype=float)
    challenger_states = np.asarray(challenger.states, dtype=float)
    composed = stable_states.copy()
    quaternion_indices = {index for group in quaternion_groups for index in group}
    frame_sources = {frame: "stable" for frame in frames}
    selected_intervals: list[dict[str, object]] = []
    for decision in ledger.decisions:
        if decision.normalized_label != "use_occlusion_challenger":
            continue
        selected = [
            frame for frame in frames
            if decision.start_frame <= frame <= decision.end_frame
        ]
        if not selected:
            continue
        count = len(selected)
        edge = min(transition_frames, max(0, count // 2))
        for local_index, frame in enumerate(selected):
            blend = 1.0
            source = "challenger"
            if edge:
                if local_index < edge:
                    blend = (local_index + 1) / float(edge + 1)
                    source = "transition"
                elif local_index >= count - edge:
                    blend = (count - local_index) / float(edge + 1)
                    source = "transition"
            index = frame_to_index[frame]
            for state_index in range(composed.shape[1]):
                if state_index not in quaternion_indices:
                    composed[index, state_index] = (
                        (1.0 - blend) * stable_states[index, state_index]
                        + blend * challenger_states[index, state_index]
                    )
            for group in quaternion_groups:
                indices = list(group)
                composed[index, indices] = _normalized_shortest_path_blend(
                    stable_states[index, indices],
                    challenger_states[index, indices],
                    blend,
                )
            frame_sources[frame] = source
        selected_intervals.append(
            {
                "query_id": decision.query_id,
                "start_frame": selected[0],
                "end_frame": selected[-1],
                "transition_frames": edge,
                "evidence_sha256": decision.evidence_sha256,
                "response_sha256": decision.response_sha256,
            }
        )
    changed = np.max(np.abs(composed - stable_states), axis=1) > 1e-12
    changed_frames = [frame for frame, value in zip(frames, changed) if value]
    quaternion_norm_errors = [
        abs(float(np.linalg.norm(composed[:, list(group)], axis=1).max()) - 1.0)
        for group in quaternion_groups
    ] + [
        abs(float(np.linalg.norm(composed[:, list(group)], axis=1).min()) - 1.0)
        for group in quaternion_groups
    ]
    translation_steps = np.linalg.norm(np.diff(composed[:, :3], axis=0), axis=1)
    provenance: dict[str, object] = {
        "schema_version": 1,
        "status": "composed" if changed_frames else "stable_unchanged",
        "stable_attempt_id": stable.solve_attempt_id,
        "challenger_attempt_id": challenger.solve_attempt_id,
        "selected_intervals": selected_intervals,
        "changed_frames": changed_frames,
        "frame_sources": {str(frame): source for frame, source in frame_sources.items()},
        "maximum_quaternion_norm_error": max(quaternion_norm_errors, default=0.0),
        "maximum_translation_step_m": float(translation_steps.max(initial=0.0)),
        "ledger_sha256": _canonical_hash(asdict(ledger)),
    }
    if not changed_frames:
        return IntervalCompositionOutcome(result=stable, provenance=provenance)
    payload = {
        "parent": stable.solve_attempt_id,
        "challenger": challenger.solve_attempt_id,
        "ledger_sha256": provenance["ledger_sha256"],
        "states": composed.tolist(),
    }
    result = replace(
        stable,
        solve_attempt_id="generic-solve-" + _canonical_hash(payload)[:12],
        parent_solve_attempt_id=stable.solve_attempt_id,
        states=tuple(tuple(float(value) for value in row) for row in composed),
        message=stable.message + "; VLM interval candidate selection applied",
        canonical_sha256="",
    )
    result_payload = asdict(result)
    result_payload.pop("canonical_sha256")
    result = replace(result, canonical_sha256=_canonical_hash(result_payload))
    return IntervalCompositionOutcome(result=result, provenance=provenance)

