from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping


ALLOWED_LABELS: dict[str, frozenset[str]] = {
    "visible_face": frozenset(
        {"grasp_side_wide", "opposite_wide", "side_left", "side_right", "unclear"}
    ),
    "facing_relation": frozenset(
        {"grasp_side_toward_human", "grasp_side_away", "side_on", "unclear"}
    ),
    "turn_direction_screen": frozenset(
        {"counterclockwise", "clockwise", "stationary", "unclear"}
    ),
    "visibility": frozenset(
        {"visible", "partial", "human_occluded", "absent", "unclear"}
    ),
    "grasp_state": frozenset({"active", "released", "unclear"}),
}

FORBIDDEN_CONTINUOUS_FIELDS = frozenset(
    {"xyz", "pose_xyz", "quaternion", "euler", "rotation_deg", "yaw", "pitch", "roll", "loss_weight"}
)


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

    def __post_init__(self) -> None:
        if not self.relation_id or self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ValueError("semantic relation identity/interval is invalid")
        if not self.subject_entity or not self.object_entity or not self.source_query_id:
            raise ValueError("semantic relation requires subject, object, and source query")
        labels = ALLOWED_LABELS.get(self.predicate)
        if labels is None or self.label not in labels:
            raise ValueError(f"invalid semantic relation label {self.predicate}:{self.label}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("semantic relation confidence must be in [0, 1]")
        for value in (self.evidence_sha256, self.prompt_sha256, self.response_sha256):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("semantic relation requires lowercase SHA-256 provenance")

    def contains_frame(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


@dataclass(frozen=True)
class FrameSemanticUncertainty:
    frame: int
    time_s: float
    mask_area_drop: float
    rail_visibility_drop: float
    wheel_visibility_drop: float
    pose_hypothesis_disagreement: float
    human_overlap: float
    hand_handle_inconsistency: float

    def __post_init__(self) -> None:
        if self.frame < 1 or self.time_s < 0.0:
            raise ValueError("semantic uncertainty frame/time is invalid")
        values = (
            self.mask_area_drop,
            self.rail_visibility_drop,
            self.wheel_visibility_drop,
            self.pose_hypothesis_disagreement,
            self.human_overlap,
            self.hand_handle_inconsistency,
        )
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("semantic uncertainty components must be in [0, 1]")

    @property
    def score(self) -> float:
        return (
            0.25 * self.mask_area_drop
            + 0.20 * self.rail_visibility_drop
            + 0.15 * self.wheel_visibility_drop
            + 0.20 * self.pose_hypothesis_disagreement
            + 0.10 * self.human_overlap
            + 0.10 * self.hand_handle_inconsistency
        )


def select_uncertain_windows(
    frames: Iterable[FrameSemanticUncertainty],
    *,
    threshold: float,
    radius: int,
    suppression_radius: int,
    maximum_queries: int,
) -> list[tuple[int, int, int, float]]:
    """Return center/start/end/time windows at score maxima, independent of object identity."""

    if not 0.0 <= threshold <= 1.0 or radius < 0 or suppression_radius < 0 or maximum_queries < 1:
        raise ValueError("invalid semantic uncertainty selection policy")
    ordered = sorted(frames, key=lambda item: item.frame)
    candidates = sorted(
        (item for item in ordered if item.score >= threshold),
        key=lambda item: (-item.score, item.frame),
    )
    selected: list[FrameSemanticUncertainty] = []
    for candidate in candidates:
        if any(abs(candidate.frame - other.frame) <= suppression_radius for other in selected):
            continue
        selected.append(candidate)
        if len(selected) >= maximum_queries:
            break
    if not ordered:
        return []
    first, last = ordered[0].frame, ordered[-1].frame
    return [
        (item.frame, max(first, item.frame - radius), min(last, item.frame + radius), item.time_s)
        for item in sorted(selected, key=lambda value: value.frame)
    ]


def forced_choice_questions(target: str) -> tuple[dict[str, object], ...]:
    questions = {
        "visible_face": f"Which asset-declared face of the {target} is most visible?",
        "facing_relation": f"How is the grasp-side wide face of the {target} oriented relative to the human?",
        "turn_direction_screen": f"Across this temporal strip, which screen-space turn direction does the rigid {target} follow?",
        "visibility": f"What is the visibility state of the same rigid {target} in the center frame?",
        "grasp_state": f"Is the hand-handle grasp on the rigid {target} active in the center frame?",
    }
    return tuple(
        {
            "predicate": predicate,
            "question": question,
            "choices": "|".join(sorted(ALLOWED_LABELS[predicate])),
            "gate_policy": "forced_choice_semantics_only_no_pose_or_weight",
        }
        for predicate, question in questions.items()
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def parse_semantic_response(
    *,
    query: Mapping[str, object],
    response: Mapping[str, object],
    raw_response: str,
) -> SemanticRelation:
    lowered_keys = {str(key).lower() for key in response}
    forbidden = sorted(lowered_keys & FORBIDDEN_CONTINUOUS_FIELDS)
    if forbidden:
        raise ValueError(f"semantic response contains forbidden continuous fields: {forbidden}")
    predicate = str(query["predicate"])
    label = str(response.get("label", "unclear")).strip().lower()
    if label not in ALLOWED_LABELS.get(predicate, frozenset()):
        label = "unclear"
    try:
        confidence = max(0.0, min(1.0, float(response.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    query_id = str(query["query_id"])
    evidence_sha256 = str(query["evidence_sha256"])
    prompt = str(query["question"]) + "\n" + str(query["choices"])
    return SemanticRelation(
        relation_id=f"relation:{query_id}",
        start_frame=int(query["start_frame"]),
        end_frame=int(query["end_frame"]),
        subject_entity=str(query.get("subject_entity", "target_object")),
        predicate=predicate,
        object_entity=str(query.get("object_entity", "human")),
        label=label,
        confidence=confidence,
        source_query_id=query_id,
        evidence_sha256=evidence_sha256,
        prompt_sha256=_sha256_text(prompt),
        response_sha256=_sha256_text(raw_response),
    )


def relation_record(relation: SemanticRelation) -> dict[str, object]:
    return asdict(relation)


def write_semantic_relations(path: Path, relations: Iterable[SemanticRelation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            for relation in relations:
                handle.write(json.dumps(relation_record(relation), ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_semantic_relations(path: Path) -> tuple[SemanticRelation, ...]:
    if not path.is_file():
        return ()
    return tuple(
        SemanticRelation(**json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    )
