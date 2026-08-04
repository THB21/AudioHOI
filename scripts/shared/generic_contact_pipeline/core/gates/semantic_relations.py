from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ..base.config import CaseProfile
from ..semantics.relations import (
    ALLOWED_LABELS,
    SemanticRelation,
    load_semantic_relations,
    parse_semantic_response,
    write_semantic_relations,
)

QUERY_TYPE_BY_PREDICATE = {
    predicate: f"semantic_{predicate}_check" for predicate in ALLOWED_LABELS
}
PREDICATE_BY_QUERY_TYPE = {value: key for key, value in QUERY_TYPE_BY_PREDICATE.items()}
SEMANTIC_QUERY_TYPES = tuple(PREDICATE_BY_QUERY_TYPE)


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
        "visible_face": f"Which physical face of the {target} is most visible in the center panel? grasp_side_wide is the broad face on the same side as the two telescoping rails and handle mount; opposite_wide is the other broad face; side_left/side_right are the narrow side faces as seen in the image. Use the real body edges, rails, and wheels, not agreement with the dark render.",
        "facing_relation": f"How is the grasp-side wide face of the {target} oriented relative to the human in the center panel? Infer the grasp-side from the two physical rails and handle mount. Use the physical object pixels and hand-handle relation, not the dark render.",
        "turn_direction_screen": f"Across the left-to-right temporal strip, which apparent yaw rotation does the same {target} follow around its upright axis as viewed by the camera? Ignore translation of the object center to the left or right. Decide only from the changing exposure/order of physical broad and narrow faces, the two rails, and the rigid wheel constellation. Choose unclear if those orientation cues do not establish a rotation direction; do not infer direction from the dark render alone.",
        "visibility": f"What is the visibility state of the same {target} in the center frame? The cyan mask may be partial; distinguish human occlusion from true absence using rigid rails, wheels, and neighboring panels.",
        "grasp_state": f"Is a physical hand-handle grasp on the {target} active in the center frame? Require a hand at the physical handle with temporal continuity; do not treat mere overlap or a rendered marker as grasp.",
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


def question_for_query_type(target: str, query_type: str) -> dict[str, object]:
    predicate = PREDICATE_BY_QUERY_TYPE.get(query_type)
    if predicate is None:
        raise ValueError(f"unknown semantic query type: {query_type}")
    return next(item for item in forced_choice_questions(target) if item["predicate"] == predicate)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _frame_count(sample_dir: Path) -> int:
    return len(list((sample_dir / "frames").glob("*.png"))) or len(
        list((sample_dir / "frames").glob("*.jpg"))
    )


def _mask_areas(sample_dir: Path, frame_count: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception:
        return np.ones(frame_count, dtype=float)
    areas = np.zeros(frame_count, dtype=float)
    root = sample_dir / "results/segmentation/masks"
    for index in range(frame_count):
        path = root / f"{index + 1:05d}_mask.png"
        if path.is_file():
            areas[index] = float(np.count_nonzero(np.asarray(Image.open(path).convert("L"))))
    nonzero = areas[areas > 0]
    if not len(nonzero):
        return np.ones(frame_count, dtype=float)
    areas[areas <= 0] = float(np.median(nonzero))
    return areas


def _rolling_reference(values: np.ndarray, radius: int = 12) -> np.ndarray:
    reference = np.empty_like(values)
    for index in range(len(values)):
        start, end = max(0, index - radius), min(len(values), index + radius + 1)
        reference[index] = max(float(np.quantile(values[start:end], 0.75)), 1e-9)
    return reference


def _rail_drop(result_dir: Path, frame_count: int) -> np.ndarray:
    values = np.ones(frame_count, dtype=float)
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in _csv_rows(result_dir / "line_observations.csv"):
        grouped.setdefault(int(float(row["frame"])), []).append(row)
    for frame, rows in grouped.items():
        if not 1 <= frame <= frame_count:
            continue
        trusted = [row for row in rows if row.get("line_observation_trusted", "1") in {"1", "1.0", "true"}]
        paired = sum(row.get("visibility") == "visible_pair" for row in trusted)
        single = sum(row.get("visibility") == "visible_single" for row in trusted)
        values[frame - 1] = 0.0 if paired >= 2 else 0.2 if paired else 0.55 if single else 1.0
    return values


def _bottom_feature_drop(sample_dir: Path, frame_count: int) -> np.ndarray:
    rows = _csv_rows(sample_dir / "results/tracking/rigid_point_tracks.csv")
    if not rows:
        return np.zeros(frame_count, dtype=float)
    first = [row for row in rows if int(float(row["frame"])) == 1]
    if not first:
        return np.zeros(frame_count, dtype=float)
    ys = np.asarray([float(row["y"]) for row in first], dtype=float)
    threshold = float(np.quantile(ys, 0.72))
    bottom_ids = {row["track_id"] for row in first if float(row["y"]) >= threshold}
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("track_id") in bottom_ids:
            grouped.setdefault(int(float(row["frame"])), []).append(row)
    values = np.ones(frame_count, dtype=float)
    for frame, frame_rows in grouped.items():
        if 1 <= frame <= frame_count:
            visible = sum(float(row.get("visible", 0.0)) >= 0.5 for row in frame_rows)
            values[frame - 1] = 1.0 - visible / max(len(bottom_ids), 1)
    return np.clip(values, 0.0, 1.0)


def _pose_disagreement(sample_dir: Path, frame_count: int) -> np.ndarray:
    path = sample_dir / "results/megapose/rigid_pose_hypotheses.jsonl"
    if not path.is_file():
        return np.zeros(frame_count, dtype=float)
    grouped: dict[int, list[dict[str, object]]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(int(row["frame"]), []).append(row)
    anchors = np.zeros(frame_count, dtype=float)
    for frame, rows in grouped.items():
        ranked = sorted(rows, key=lambda row: int(row.get("visual_geometry_rank", 999)))[:2]
        if len(ranked) < 2 or not 1 <= frame <= frame_count:
            continue
        qa = np.asarray([ranked[0][key] for key in ("qw", "qx", "qy", "qz")], dtype=float)
        qb = np.asarray([ranked[1][key] for key in ("qw", "qx", "qy", "qz")], dtype=float)
        qa /= max(float(np.linalg.norm(qa)), 1e-12)
        qb /= max(float(np.linalg.norm(qb)), 1e-12)
        angle = 2.0 * np.arccos(np.clip(abs(float(np.dot(qa, qb))), 0.0, 1.0))
        iou_gap = abs(float(ranked[0].get("official_render_mask_iou", 0.0)) - float(ranked[1].get("official_render_mask_iou", 0.0)))
        anchors[frame - 1] = float(np.clip((angle / np.pi) * (1.0 - min(iou_gap, 1.0)), 0.0, 1.0))
    if not np.any(anchors):
        return anchors
    kernel_x = np.arange(-6, 7, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / 2.5) ** 2)
    return np.clip(np.convolve(anchors, kernel, mode="same"), 0.0, 1.0)


def build_profile_uncertainty(profile: CaseProfile) -> list[FrameSemanticUncertainty]:
    """Build generic uncertainty from mask, rigid features, pose hypotheses, and contact evidence."""

    count = _frame_count(profile.sample_dir)
    if count <= 0:
        return []
    areas = _mask_areas(profile.sample_dir, count)
    area_drop = np.clip(1.0 - areas / _rolling_reference(areas), 0.0, 1.0)
    rails = _rail_drop(profile.result_dir, count)
    wheels = _bottom_feature_drop(profile.sample_dir, count)
    pose = _pose_disagreement(profile.sample_dir, count)
    overlap = np.zeros(count, dtype=float)
    hand_gap = np.zeros(count, dtype=float)
    for row in _csv_rows(profile.result_dir / "contact_candidates.csv"):
        frame = int(float(row["frame"]))
        if not 1 <= frame <= count:
            continue
        visibility = str(row.get("visibility", "")).lower()
        if visibility in {"hidden", "occluded", "occluded_by_human"}:
            overlap[frame - 1] = 1.0
        active = str(row.get("contact_active", "0")).lower() in {"1", "1.0", "true", "active"}
        confidence = float(row.get("contact_conf", 0.0) or 0.0)
        if active:
            hand_gap[frame - 1] = max(hand_gap[frame - 1], 1.0 - np.clip(confidence, 0.0, 1.0))
    fps = float(dict(profile.data.get("preprocess", {})).get("fps", 30.0))
    return [
        FrameSemanticUncertainty(
            frame=index + 1,
            time_s=index / fps,
            mask_area_drop=float(area_drop[index]),
            rail_visibility_drop=float(rails[index]),
            wheel_visibility_drop=float(wheels[index]),
            pose_hypothesis_disagreement=float(pose[index]),
            human_overlap=float(overlap[index]),
            hand_handle_inconsistency=float(hand_gap[index]),
        )
        for index in range(count)
    ]


def profile_uncertain_windows(profile: CaseProfile) -> list[tuple[int, int, int, float]]:
    raw = dict(dict(profile.data.get("vlm", {})).get("semantic_orientation", {}))
    if not raw.get("enabled", False):
        return []
    if "disable_vlm_semantic_evidence" in set(profile.data.get("ablation_flags", ())):
        return []
    return select_uncertain_windows(
        build_profile_uncertainty(profile),
        threshold=float(raw.get("uncertainty_threshold", 0.42)),
        radius=int(raw.get("temporal_radius_frames", 3)),
        suppression_radius=int(raw.get("suppression_radius_frames", 8)),
        maximum_queries=int(raw.get("maximum_queries", 12)),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_semantic_relation_artifacts(
    out_dir: Path,
    raw_rows: Iterable[Mapping[str, object]],
    *,
    status: str,
) -> dict[str, object]:
    semantic_rows = [row for row in raw_rows if str(row.get("query_type", "")) in PREDICATE_BY_QUERY_TYPE]
    relations: list[SemanticRelation] = []
    for row in semantic_rows:
        predicate = PREDICATE_BY_QUERY_TYPE[str(row["query_type"])]
        query = {**row, "predicate": predicate}
        response = {"label": row.get("label", "unclear"), "confidence": row.get("confidence", 0.0)}
        relations.append(parse_semantic_response(query=query, response=response, raw_response=str(row.get("raw_text", ""))))
    query_path = out_dir / "semantic_queries.jsonl"
    raw_path = out_dir / "semantic_raw_responses.jsonl"
    relation_path = out_dir / "semantic_relations.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    query_path.write_text("".join(json.dumps({key: row.get(key) for key in ("query_id", "frame", "start_frame", "end_frame", "query_type", "question", "choices", "evidence_sha256")}, ensure_ascii=False, sort_keys=True) + "\n" for row in semantic_rows))
    raw_path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in semantic_rows))
    write_semantic_relations(relation_path, relations)
    manifest = {
        "schema_version": 1,
        "status": status,
        "query_count": len(semantic_rows),
        "relation_count": len(relations),
        "continuous_pose_fields_allowed": False,
        "queries_sha256": _sha256_file(query_path),
        "raw_responses_sha256": _sha256_file(raw_path),
        "relations_sha256": _sha256_file(relation_path),
    }
    manifest_path = out_dir / "semantic_relation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {**manifest, "manifest": str(manifest_path), "relations": str(relation_path)}
