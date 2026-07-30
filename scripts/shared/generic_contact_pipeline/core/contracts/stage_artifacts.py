from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ..base.schema import stage_paths


class ContractValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Se3Pose:
    frame: int
    time: float
    translation: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class PointLocalCoordinate:
    x: float | None
    y: float | None
    z: float | None


@dataclass(frozen=True)
class LineLocalCoordinate:
    s: float | None


@dataclass(frozen=True)
class ContactCandidate:
    frame: int
    time: float
    contact_active: bool
    human_part: str
    human_side: str | None
    object_part: str
    local_coordinate: PointLocalCoordinate | LineLocalCoordinate | None


@dataclass(frozen=True)
class PointAnchorState:
    frame: int
    time: float
    contact_id: str
    human_part: str
    human_side: str | None
    object_part: str
    stable_local: PointLocalCoordinate


@dataclass(frozen=True)
class LineAnchorState:
    frame: int
    time: float
    human_side: str
    observed_local_s: float | None
    stable_local_s: float | None


@dataclass(frozen=True)
class FrameTimeRecord:
    frame: int
    time: float


@dataclass(frozen=True)
class FrameRecord:
    frame: int


@dataclass(frozen=True)
class CsvContract:
    artifact_key: str
    adapter: str
    required_columns: tuple[str, ...]
    row_adapter: Callable[[dict[str, str]], object] | None = None
    required: bool = False


def _required_text(row: dict[str, str], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"{key} is empty")
    return value


def _optional_text(row: dict[str, str], key: str) -> str | None:
    value = str(row.get(key, "")).strip()
    return value or None


def _float(row: dict[str, str], key: str, *, optional: bool = False) -> float | None:
    raw = str(row.get(key, "")).strip()
    if optional and not raw:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def _int(row: dict[str, str], key: str) -> int:
    value = _float(row, key)
    assert value is not None
    integer = int(value)
    if value != integer:
        raise ValueError(f"{key} is not an integer")
    return integer


def _bool(row: dict[str, str], key: str) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(f"{key} is not boolean-like")


def adapt_se3_pose(row: dict[str, str]) -> Se3Pose:
    quaternion = tuple(float(_float(row, key)) for key in ("qw", "qx", "qy", "qz"))
    if not any(value != 0.0 for value in quaternion):
        raise ValueError("quaternion is all zero")
    return Se3Pose(
        frame=_int(row, "frame"),
        time=float(_float(row, "time")),
        translation=tuple(float(_float(row, key)) for key in ("tx", "ty", "tz")),
        quaternion_wxyz=quaternion,
    )


def adapt_frame_time(row: dict[str, str]) -> FrameTimeRecord:
    return FrameTimeRecord(frame=_int(row, "frame"), time=float(_float(row, "time")))


def adapt_frame(row: dict[str, str]) -> FrameRecord:
    return FrameRecord(frame=_int(row, "frame"))


def adapt_contact_candidate(row: dict[str, str]) -> ContactCandidate:
    if "object_local_s" in row or "stable_object_local_s" in row:
        local: PointLocalCoordinate | LineLocalCoordinate | None = LineLocalCoordinate(
            _float(
                row,
                "object_local_s" if "object_local_s" in row else "stable_object_local_s",
                optional=True,
            )
        )
    elif any(key in row for key in ("stable_local_x", "object_local_x")):
        prefix = "stable_local_" if "stable_local_x" in row else "object_local_"
        local = PointLocalCoordinate(
            _float(row, prefix + "x", optional=True),
            _float(row, prefix + "y", optional=True),
            _float(row, prefix + "z", optional=True),
        )
    else:
        local = None
    return ContactCandidate(
        frame=_int(row, "frame"),
        time=float(_float(row, "time")),
        contact_active=_bool(row, "contact_active"),
        human_part=_required_text(row, "human_part"),
        human_side=_optional_text(row, "human_side"),
        object_part=_required_text(row, "object_part"),
        local_coordinate=local,
    )


def adapt_point_anchor(row: dict[str, str]) -> PointAnchorState:
    return PointAnchorState(
        frame=_int(row, "frame"),
        time=float(_float(row, "time")),
        contact_id=_required_text(row, "contact_id"),
        human_part=_required_text(row, "human_part"),
        human_side=_optional_text(row, "human_side"),
        object_part=_required_text(row, "object_part"),
        stable_local=PointLocalCoordinate(
            _float(row, "stable_local_x", optional=True),
            _float(row, "stable_local_y", optional=True),
            _float(row, "stable_local_z", optional=True),
        ),
    )


def adapt_line_anchor(row: dict[str, str]) -> LineAnchorState:
    return LineAnchorState(
        frame=_int(row, "frame"),
        time=float(_float(row, "time")),
        human_side=_required_text(row, "human_side"),
        observed_local_s=_float(row, "observed_object_local_s", optional=True),
        stable_local_s=_float(row, "stable_object_local_s", optional=True),
    )


def _observation_contract(case_name: str) -> CsvContract:
    if case_name == "mug":
        return CsvContract(
            "object_observations",
            "mug_rigid_parts_observation_v1",
            ("frame", "time", "center_x", "center_y", "handle_side", "observation_conf"),
            adapt_frame_time,
            required=True,
        )
    if case_name == "chair":
        return CsvContract(
            "object_observations",
            "chair_semantic_graph_observation_v1",
            (
                "frame", "time", "top_rail_left_u", "top_rail_left_v",
                "top_rail_right_u", "top_rail_right_v", "seat_center_u",
                "seat_center_v", "semantic_conf",
            ),
            adapt_frame_time,
            required=True,
        )
    return CsvContract(
        "object_observations",
        "point_reference_observation_v1",
        ("frame", "time", "ref_u", "ref_v", "support_u", "support_v", "observation_conf"),
        adapt_frame_time,
        required=True,
    )


def _semantic_points_contract(case_name: str) -> CsvContract:
    if case_name == "chair":
        return CsvContract(
            "object_semantic_points",
            "semantic_segment_v1",
            (
                "segment_id", "part", "role", "start_point_id", "end_point_id",
                "start_local_x", "start_local_y", "start_local_z",
                "end_local_x", "end_local_y", "end_local_z", "source",
            ),
        )
    return CsvContract(
        "object_semantic_points",
        "semantic_point_v1",
        ("point_id", "part", "role", "local_x", "local_y", "local_z", "source"),
    )


def contracts_for_stage(case_name: str, stage_name: str) -> list[CsvContract]:
    correspondence = CsvContract(
        "object_correspondence",
        "line_object_correspondence_v1" if case_name == "stick" else "tracked_point_correspondence_v1",
        (
            "frame", "time", "correspondence_id", "u", "v",
            "track_confidence", "visible_fraction", "occlusion_state", "source",
        ),
        adapt_frame if case_name == "stick" else adapt_frame_time,
        required=True,
    )
    if stage_name == "stage1":
        contracts = [_observation_contract(case_name), correspondence, _semantic_points_contract(case_name)]
        if case_name == "stick":
            contracts.append(
                CsvContract(
                    "line_observations",
                    "line_observation_v1",
                    (
                        "frame", "visible_x1", "visible_y1", "visible_x2", "visible_y2",
                        "visible_len_px", "visible_angle_rad", "physical_length_m",
                        "visible_fraction", "occlusion_state", "line_observation_trusted",
                    ),
                    adapt_frame,
                    required=True,
                )
            )
        return contracts
    if stage_name == "stage2":
        candidates = CsvContract(
            "contact_candidates",
            "contact_candidate_v1",
            (
                "frame", "time", "contact_active", "human_part", "human_side",
                "object_part", "object_local_id", "contact_u", "contact_v",
                "contact_depth_offset_m", "anchor_score", "source",
            ),
            adapt_contact_candidate,
            required=True,
        )
        if case_name == "stick":
            anchor = CsvContract(
                "anchor_state",
                "line_anchor_state_v1",
                (
                    "frame", "time", "human_side", "contact_observed",
                    "contact_persistent", "anchor_update_allowed", "pose_anchor_allowed",
                    "observed_object_local_s", "stable_object_local_s", "anchor_action",
                ),
                adapt_line_anchor,
                required=True,
            )
        else:
            anchor = CsvContract(
                "anchor_state",
                "point_anchor_state_v1",
                (
                    "frame", "time", "contact_id", "human_part", "human_side",
                    "object_part", "contact_observed", "contact_persistent",
                    "anchor_update_allowed", "pose_anchor_allowed", "stable_local_x",
                    "stable_local_y", "stable_local_z", "anchor_action", "source",
                ),
                adapt_point_anchor,
                required=True,
            )
        return [candidates, anchor]
    if stage_name == "stage3":
        return [
            CsvContract(
                "object_pose_init",
                "se3_pose_v1",
                ("frame", "time", "tx", "ty", "tz", "qw", "qx", "qy", "qz"),
                adapt_se3_pose,
                required=True,
            )
        ]
    if stage_name == "stage4":
        return [
            CsvContract(
                "object_pose",
                "se3_pose_v1",
                ("frame", "time", "tx", "ty", "tz", "qw", "qx", "qy", "qz"),
                adapt_se3_pose,
                required=True,
            ),
            CsvContract(
                "motion_regime",
                "motion_regime_v1",
                (
                    "frame", "time", "motion_regime", "motion_speed_score",
                    "contact_observed", "contact_persistent", "pose_anchor_allowed", "source",
                ),
                adapt_frame_time,
            ),
            CsvContract(
                "physical_smooth_residuals",
                "physical_smooth_residual_v1",
                (
                    "frame", "se3_velocity_spike", "se3_accel_spike",
                    "delta_translation_m", "delta_rotation_rad", "trust_weight",
                    "anchor_residual_enabled", "anchor_residual_count", "anchor_residual_m",
                ),
                adapt_frame,
            ),
            CsvContract(
                "optimizer_decisions",
                "optimizer_decision_v1",
                (
                    "frame", "motion_regime", "sequence_se3_optimizer_enabled",
                    "visual_residual_enabled", "contact_anchor_residual_enabled",
                    "depth_residual_enabled", "geometry_residual_enabled",
                    "velocity_residual_enabled", "acceleration_residual_enabled",
                    "trust_weight", "smooth_weight", "accel_weight", "decision_source",
                ),
                adapt_frame,
            ),
        ]
    return []


def _validate_csv(path: Path, contract: CsvContract) -> dict[str, object]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    missing_columns = [column for column in contract.required_columns if column not in columns]
    errors = [f"missing required column {column}" for column in missing_columns]
    adapted_examples: list[dict[str, Any]] = []
    if not missing_columns and contract.row_adapter is not None:
        for index, row in enumerate(rows):
            try:
                adapted = contract.row_adapter(row)
                if len(adapted_examples) < 2:
                    adapted_examples.append(asdict(adapted))
            except (AssertionError, TypeError, ValueError) as exc:
                errors.append(f"row {index + 2}: {exc}")
                if len(errors) >= 20:
                    errors.append("additional row errors truncated")
                    break
    return {
        "artifact_key": contract.artifact_key,
        "path": str(path),
        "adapter": contract.adapter,
        "required": contract.required,
        "columns": columns,
        "required_columns": list(contract.required_columns),
        "extension_columns": [column for column in columns if column not in contract.required_columns],
        "row_count": len(rows),
        "adapted_examples": adapted_examples,
        "status": "pass" if not errors else "fail",
        "errors": errors,
    }


def validate_stage_contracts(profile: Any, stage_name: str) -> dict[str, object]:
    paths = stage_paths(profile)
    artifacts: list[dict[str, object]] = []
    errors: list[str] = []
    for contract in contracts_for_stage(profile.case_name, stage_name):
        path = paths[contract.artifact_key]
        if not path.exists():
            if contract.required:
                errors.append(f"{contract.artifact_key}: required artifact is missing: {path}")
            artifacts.append(
                {
                    "artifact_key": contract.artifact_key,
                    "path": str(path),
                    "adapter": contract.adapter,
                    "required": contract.required,
                    "status": "missing_required" if contract.required else "not_materialized",
                    "errors": [],
                }
            )
            continue
        audit = _validate_csv(path, contract)
        artifacts.append(audit)
        errors.extend(f"{contract.artifact_key}: {error}" for error in audit["errors"])
    return {
        "schema_version": 1,
        "case_name": profile.case_name,
        "result_name": profile.result_name,
        "stage": stage_name,
        "status": "pass" if not errors else "fail",
        "artifacts": artifacts,
        "errors": errors,
    }
