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
        {
            "grasp_side_wide",
            "grasp_side_wide_left_oblique",
            "grasp_side_wide_right_oblique",
            "opposite_wide",
            "side_left",
            "side_right",
            "unclear",
        }
    ),
    "facing_relation": frozenset(
        {"grasp_side_toward_human", "grasp_side_away", "side_on", "unclear"}
    ),
    "side_exposure": frozenset(
        {"left_exposed", "right_exposed", "none", "unclear"}
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
    lowered_raw = raw_response.lower()
    forbidden.extend(
        field
        for field in FORBIDDEN_CONTINUOUS_FIELDS
        if f'"{field}"' in lowered_raw and field not in forbidden
    )
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
