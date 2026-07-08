#!/usr/bin/env python3
"""Export semantic 3D local anchors from the Articraft chair model.

Chair-side counterpart of the mug-local mesh coordinates. No image pose fitting
here, just stable local 3D points/segments that later 2D/6D fitting projects
onto the video observations.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


POINT_FIELDS = [
    "point_id",
    "part",
    "role",
    "local_x",
    "local_y",
    "local_z",
    "weight",
    "source",
]

SEGMENT_FIELDS = [
    "segment_id",
    "part",
    "role",
    "start_point_id",
    "end_point_id",
    "start_local_x",
    "start_local_y",
    "start_local_z",
    "end_local_x",
    "end_local_y",
    "end_local_z",
    "weight",
    "source",
]


def parse_model_constants(model_py: Path) -> dict[str, Any]:
    text = model_py.read_text()
    tree = ast.parse(text)
    constants: dict[str, Any] = {}
    wanted = {
        "WIDTH",
        "DEPTH",
        "HEIGHT",
        "SEAT_TOP_Z",
        "FRONT_FOOT_Y",
        "REAR_FOOT_Y",
        "FOOT_Z",
        "PIVOT_Y",
        "PIVOT_Z",
        "FRONT_TOP_Y",
        "FRONT_TOP_Z",
        "UPPER_HALF_WIDTH",
        "FOOT_HALF_WIDTH",
        "REAR_SUPPORT_TOP_HALF_WIDTH",
        "BACKREST_WIDTH",
        "BACKREST_HEIGHT",
        "BACKREST_THICKNESS",
        "BACKREST_CENTER",
        "SEAT_HINGE_Y",
        "SEAT_LENGTH",
        "SEAT_THICKNESS",
        "SEAT_SIDE_X",
        "CENTER_SLAT_XS",
        "LOWER_STRETCHER_RADIUS",
    }

    def eval_const(node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            if node.id not in constants:
                raise ValueError(f"Constant {node.id!r} referenced before definition")
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -float(eval_const(node.operand))
        if isinstance(node, ast.Tuple):
            return tuple(eval_const(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [eval_const(elt) for elt in node.elts]
        return ast.literal_eval(node)

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        constants[target.id] = eval_const(node.value)
    missing = sorted(wanted - constants.keys())
    if missing:
        raise ValueError(f"Missing constants in {model_py}: {missing}")
    return constants


def v(a: tuple[float, float, float] | list[float]) -> np.ndarray:
    return np.asarray(a, dtype=float)


def add_point(
    rows: list[dict[str, str]],
    point_map: dict[str, np.ndarray],
    point_id: str,
    part: str,
    role: str,
    xyz: np.ndarray,
    weight: float,
    source: str,
) -> None:
    point_map[point_id] = xyz.astype(float)
    rows.append(
        {
            "point_id": point_id,
            "part": part,
            "role": role,
            "local_x": f"{float(xyz[0]):.6f}",
            "local_y": f"{float(xyz[1]):.6f}",
            "local_z": f"{float(xyz[2]):.6f}",
            "weight": f"{weight:.3f}",
            "source": source,
        }
    )


def add_segment(
    rows: list[dict[str, str]],
    point_map: dict[str, np.ndarray],
    segment_id: str,
    part: str,
    role: str,
    start_point_id: str,
    end_point_id: str,
    weight: float,
    source: str,
) -> None:
    a = point_map[start_point_id]
    b = point_map[end_point_id]
    rows.append(
        {
            "segment_id": segment_id,
            "part": part,
            "role": role,
            "start_point_id": start_point_id,
            "end_point_id": end_point_id,
            "start_local_x": f"{float(a[0]):.6f}",
            "start_local_y": f"{float(a[1]):.6f}",
            "start_local_z": f"{float(a[2]):.6f}",
            "end_local_x": f"{float(b[0]):.6f}",
            "end_local_y": f"{float(b[1]):.6f}",
            "end_local_z": f"{float(b[2]):.6f}",
            "weight": f"{weight:.3f}",
            "source": source,
        }
    )


def interp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def sanitize_record_id(path: Path) -> str:
    match = re.search(r"(rec_[^/]+)", str(path))
    return match.group(1) if match else path.parent.name


def build(model_py: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    c = parse_model_constants(model_py)
    source = sanitize_record_id(model_py)
    model_text = model_py.read_text()

    def optional_float_constant(name: str, default: float) -> float:
        match = re.search(rf"^{name}\s*=\s*([-0-9.]+)", model_text, re.M)
        return float(match.group(1)) if match else default

    front_lower_x = optional_float_constant("FRONT_LOWER_STRETCHER_X", 0.202)
    front_lower_y = optional_float_constant("FRONT_LOWER_STRETCHER_Y", -0.179)
    front_lower_z = optional_float_constant("FRONT_LOWER_STRETCHER_Z", 0.138)
    rear_lower_local_y = optional_float_constant("REAR_LOWER_STRETCHER_LOCAL_Y", 0.164)
    rear_lower_local_z = optional_float_constant("REAR_LOWER_STRETCHER_LOCAL_Z", -0.366)

    front_foot_y = float(c["FRONT_FOOT_Y"])
    rear_foot_y = float(c["REAR_FOOT_Y"])
    foot_z = float(c["FOOT_Z"])
    pivot_y = float(c["PIVOT_Y"])
    pivot_z = float(c["PIVOT_Z"])
    front_top_y = float(c["FRONT_TOP_Y"])
    front_top_z = float(c["FRONT_TOP_Z"])
    upper_half_width = float(c["UPPER_HALF_WIDTH"])
    foot_half_width = float(c["FOOT_HALF_WIDTH"])
    rear_top_half_width = float(c["REAR_SUPPORT_TOP_HALF_WIDTH"])
    backrest_width = float(c["BACKREST_WIDTH"])
    backrest_height = float(c["BACKREST_HEIGHT"])
    backrest_thickness = float(c["BACKREST_THICKNESS"])
    backrest_center = v(c["BACKREST_CENTER"])
    seat_hinge_y = float(c["SEAT_HINGE_Y"])
    seat_length = float(c["SEAT_LENGTH"])
    seat_top_z = float(c["SEAT_TOP_Z"])
    seat_side_x = float(c["SEAT_SIDE_X"])
    # Match the Articraft seat visual, not a widened 2D proxy. The widest
    # modeled seat member is front_slat_tie with x size 0.365 in model.py.
    visible_seat_half_width = max(seat_side_x + 0.015, 0.365 * 0.5)
    center_slat_xs = [float(x) for x in c["CENTER_SLAT_XS"]]

    points: list[dict[str, str]] = []
    segments: list[dict[str, str]] = []
    point_map: dict[str, np.ndarray] = {}

    def point(pid: str, part: str, role: str, xyz: tuple[float, float, float] | np.ndarray, weight: float = 1.0) -> None:
        add_point(points, point_map, pid, part, role, v(xyz), weight, source)

    def segment(sid: str, part: str, role: str, start: str, end: str, weight: float = 1.0) -> None:
        add_segment(segments, point_map, sid, part, role, start, end, weight, source)

    # Front frame: the continuous orange legs in the video.
    for side, name in [(-1.0, "left"), (1.0, "right")]:
        foot = np.array([side * foot_half_width, front_foot_y, foot_z])
        top = np.array([side * upper_half_width, front_top_y, front_top_z])
        hinge_t = (pivot_z - foot_z) / (front_top_z - foot_z)
        board_t = (backrest_center[2] - foot_z) / (front_top_z - foot_z)
        point(f"front_leg_{name}_bottom", "front_leg", "floor_foot", foot, 1.0)
        point(f"front_leg_{name}_top", "front_leg", "top_post", top, 1.0)
        point(f"front_leg_{name}_hinge", "front_leg", "side_hinge", interp(foot, top, hinge_t), 0.8)
        point(f"front_leg_{name}_backrest_mount", "front_leg", "backrest_mount", interp(foot, top, board_t), 1.0)
        segment(f"front_leg_{name}", "front_leg", "orange_front_leg", f"front_leg_{name}_top", f"front_leg_{name}_bottom", 1.0)

    # Backrest board and top rail-like visible board edge.
    board_y_front = float(backrest_center[1])
    board_y_back = float(backrest_center[1] - backrest_thickness)
    board_z_top = float(backrest_center[2] + backrest_height * 0.5)
    board_z_bottom = float(backrest_center[2] - backrest_height * 0.5)
    for x_name, x in [("left", -backrest_width * 0.5), ("right", backrest_width * 0.5)]:
        point(f"backrest_top_{x_name}", "backrest_board", "top_edge", (x, board_y_front, board_z_top), 1.0)
        point(f"backrest_bottom_{x_name}", "backrest_board", "bottom_edge", (x, board_y_front, board_z_bottom), 0.8)
    point("backrest_center", "backrest_board", "board_center", (backrest_center[0], board_y_front, backrest_center[2]), 1.0)
    point("backrest_hole_center", "backrest_board", "round_hole_center", (backrest_center[0], board_y_front, backrest_center[2]), 0.8)
    point("backrest_board_back_center", "backrest_board", "board_back_face_center", (backrest_center[0], board_y_back, backrest_center[2]), 0.5)
    segment("backrest_top_edge", "backrest_board", "top_rail_observation", "backrest_top_left", "backrest_top_right", 1.0)
    segment("backrest_bottom_edge", "backrest_board", "board_bottom_edge", "backrest_bottom_left", "backrest_bottom_right", 0.8)

    # Rear semantic legs must follow the Articraft rear_support link, not the
    # backrest board edge. Earlier proxy wires connected rear_leg_top to the
    # board bottom, which made the overlay look plausible in some frames but
    # broke the real chair proportions and the front/back hinge geometry.
    for side, name in [(-1.0, "left"), (1.0, "right")]:
        point(f"rear_leg_{name}_top", "rear_leg", "rear_support_hinge", (side * rear_top_half_width, pivot_y, pivot_z), 1.0)
        point(f"rear_leg_{name}_bottom", "rear_leg", "floor_foot", (side * foot_half_width, rear_foot_y, foot_z), 1.0)
        segment(f"rear_leg_{name}", "rear_leg", "blue_rear_leg", f"rear_leg_{name}_top", f"rear_leg_{name}_bottom", 1.0)

    # Lower horizontal stretchers from the Articraft model.
    point("front_lower_stretcher_left", "front_stretcher", "lower_stretcher_endpoint", (-front_lower_x, front_lower_y, front_lower_z), 0.8)
    point("front_lower_stretcher_right", "front_stretcher", "lower_stretcher_endpoint", (front_lower_x, front_lower_y, front_lower_z), 0.8)
    segment("front_lower_stretcher", "front_stretcher", "front_lower_stretcher", "front_lower_stretcher_left", "front_lower_stretcher_right", 0.8)
    point("rear_lower_stretcher_left", "rear_stretcher", "lower_stretcher_endpoint", (-front_lower_x, pivot_y + rear_lower_local_y, pivot_z + rear_lower_local_z), 0.8)
    point("rear_lower_stretcher_right", "rear_stretcher", "lower_stretcher_endpoint", (front_lower_x, pivot_y + rear_lower_local_y, pivot_z + rear_lower_local_z), 0.8)
    segment("rear_lower_stretcher", "rear_stretcher", "rear_lower_stretcher", "rear_lower_stretcher_left", "rear_lower_stretcher_right", 0.8)
    segment("left_side_lower_stretcher", "side_stretcher", "side_lower_stretcher", "front_lower_stretcher_left", "rear_lower_stretcher_left", 0.8)
    segment("right_side_lower_stretcher", "side_stretcher", "side_lower_stretcher", "front_lower_stretcher_right", "rear_lower_stretcher_right", 0.8)

    # Seat is a child link at the hinge. Its visible front edge in the video is
    # a stable secondary body anchor.
    seat_joint = np.array([0.0, seat_hinge_y, seat_top_z])
    seat_front_y = seat_hinge_y - seat_length
    seat_center_y = seat_hinge_y - seat_length * 0.5
    point("seat_front_left", "seat", "front_edge", (-visible_seat_half_width, seat_front_y, seat_top_z), 1.0)
    point("seat_front_right", "seat", "front_edge", (visible_seat_half_width, seat_front_y, seat_top_z), 1.0)
    point("seat_center", "seat", "seat_plane_center", (0.0, seat_center_y, seat_top_z), 1.0)
    point("seat_hinge_left", "seat", "hinge_edge", (-visible_seat_half_width, seat_hinge_y, seat_top_z), 0.8)
    point("seat_hinge_right", "seat", "hinge_edge", (visible_seat_half_width, seat_hinge_y, seat_top_z), 0.8)
    segment("seat_front_edge", "seat", "seat_front_observation", "seat_front_left", "seat_front_right", 1.0)
    segment("seat_hinge_edge", "seat", "seat_hinge_observation", "seat_hinge_left", "seat_hinge_right", 0.6)
    for idx, x in enumerate(center_slat_xs):
        pid_front = f"seat_slat_{idx}_front"
        pid_back = f"seat_slat_{idx}_back"
        point(pid_front, "seat_slat", "slat_front", (x, seat_front_y, seat_top_z), 0.4)
        point(pid_back, "seat_slat", "slat_back", (x, seat_hinge_y, seat_top_z), 0.4)
        segment(f"seat_slat_{idx}", "seat_slat", "seat_depth_line", pid_back, pid_front, 0.4)

    # Feet/floor support anchors.
    segment("front_feet_line", "feet", "front_floor_support", "front_leg_left_bottom", "front_leg_right_bottom", 0.7)
    segment("rear_feet_line", "feet", "rear_floor_support", "rear_leg_left_bottom", "rear_leg_right_bottom", 0.7)
    for point_id in ["front_leg_left_bottom", "front_leg_right_bottom", "rear_leg_left_bottom", "rear_leg_right_bottom"]:
        xyz = point_map[point_id]
        point(f"{point_id}_floor_contact", "feet", "floor_contact", (xyz[0], xyz[1], 0.0), 0.6)

    out_dir.mkdir(parents=True, exist_ok=True)
    point_csv = out_dir / "chair_semantic_local_points.csv"
    segment_csv = out_dir / "chair_semantic_local_segments.csv"
    with point_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POINT_FIELDS)
        writer.writeheader()
        writer.writerows(points)
    with segment_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        writer.writerows(segments)

    xyzs = np.vstack([np.asarray([float(r["local_x"]), float(r["local_y"]), float(r["local_z"])]) for r in points])
    summary = {
        "point_csv": str(point_csv),
        "segment_csv": str(segment_csv),
        "points": len(points),
        "segments": len(segments),
        "source_model_py": str(model_py),
        "record_id": source,
        "bbox_min": xyzs.min(axis=0).round(6).tolist(),
        "bbox_max": xyzs.max(axis=0).round(6).tolist(),
        "axis_note": "local x=chair width, local y=depth/front-back, local z=up; root frame is front_frame with rear/seat joints at default open pose.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return point_csv, segment_csv, summary_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples_known_object/05_chair"))
    ap.add_argument("--model-py", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    sample = args.sample_dir
    model_py = args.model_py or (
        sample
        / "articraft/generated_record/rec_continue-editing-the-current-topology-fixed-chai_20260615_113825_272909_6911e934/revisions/rev_000001/model.py"
    )
    out_dir = args.out_dir or sample / "results/chair_semantic_local_points"
    build(model_py, out_dir)


if __name__ == "__main__":
    main()
