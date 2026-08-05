#!/usr/bin/env python3
"""Project a typed VLM face decision onto an existing rigid pose candidate.

The continuous trajectory remains solver-derived.  This tool resolves only an
upright yaw/face-identity ambiguity over a declared semantic transition window.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _relations(path: Path, predicate: str, frame: int) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        row
        for row in rows
        if row.get("predicate") == predicate
        and int(row.get("start_frame", frame)) <= frame <= int(row.get("end_frame", frame))
    ]


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pose", type=Path, required=True)
    parser.add_argument("--output-pose", type=Path, required=True)
    parser.add_argument("--semantic-relations", type=Path, required=True)
    parser.add_argument("--asset-descriptor", type=Path, required=True)
    parser.add_argument("--transition-start", type=int, required=True)
    parser.add_argument("--transition-end", type=int, required=True)
    parser.add_argument("--symmetry-switch-frame", type=int)
    parser.add_argument("--residual-transition-end", type=int)
    parser.add_argument("--terminal-frame", type=int, required=True)
    parser.add_argument("--side-oblique-strength", type=float, default=0.35)
    parser.add_argument("--support-normal", type=float, nargs=3, default=(0.0, -1.0, 0.0))
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    if args.transition_end <= args.transition_start:
        raise ValueError("semantic transition end must follow start")
    rows = list(csv.DictReader(args.input_pose.open(newline="")))
    if not rows:
        raise ValueError("pose input is empty")
    by_frame = {int(row["frame"]): row for row in rows}
    terminal = by_frame[args.terminal_frame]
    descriptor = json.loads(args.asset_descriptor.read_text())
    face_specs = descriptor["semantic_orientation_features"]
    face_relations = _relations(args.semantic_relations, "visible_face", args.terminal_frame)
    side_relations = _relations(args.semantic_relations, "side_exposure", args.terminal_frame)
    if not face_relations or not side_relations:
        raise ValueError("terminal frame lacks typed visible_face/side_exposure evidence")
    face = max(face_relations, key=lambda row: float(row["confidence"]))
    side = max(side_relations, key=lambda row: float(row["confidence"]))
    selected = np.asarray(face_specs[str(face["label"])]["normal_local"], dtype=float)
    side_label = {"left_exposed": "side_left", "right_exposed": "side_right"}.get(str(side["label"]))
    if side_label is not None:
        selected += args.side_oblique_strength * np.asarray(
            face_specs[side_label]["normal_local"], dtype=float
        )
    selected /= np.linalg.norm(selected)

    translation = np.asarray([float(terminal[key]) for key in ("tx", "ty", "tz")])
    view = -translation / np.linalg.norm(translation)
    rotation = Rotation.from_quat([float(terminal[key]) for key in ("qx", "qy", "qz", "qw")])
    axis = np.asarray(args.support_normal, dtype=float)
    axis /= np.linalg.norm(axis)
    candidates = np.linspace(-np.pi, np.pi, 7201)
    scores = np.asarray([
        float((Rotation.from_rotvec(angle * axis) * rotation).apply(selected) @ view)
        for angle in candidates
    ])
    correction = float(candidates[int(np.argmax(scores))])

    symmetry_correction = 0.0
    residual_correction = correction
    if args.symmetry_switch_frame is not None:
        symmetry_correction = float(np.copysign(np.pi, correction))
        residual_correction = correction - symmetry_correction
        if args.residual_transition_end is None:
            raise ValueError("symmetry switch requires --residual-transition-end")
        if args.residual_transition_end <= args.symmetry_switch_frame:
            raise ValueError("residual transition end must follow symmetry switch")

    fieldnames = list(rows[0])
    output_rows: list[dict[str, object]] = []
    for source in rows:
        row: dict[str, object] = dict(source)
        frame = int(source["frame"])
        if args.symmetry_switch_frame is not None:
            if frame < args.symmetry_switch_frame:
                applied_correction = 0.0
            else:
                fraction = _smoothstep(
                    (frame - args.symmetry_switch_frame)
                    / (args.residual_transition_end - args.symmetry_switch_frame)
                )
                applied_correction = symmetry_correction + fraction * residual_correction
        else:
            if frame <= args.transition_start:
                fraction = 0.0
            elif frame >= args.transition_end:
                fraction = 1.0
            else:
                fraction = _smoothstep(
                    (frame - args.transition_start) / (args.transition_end - args.transition_start)
                )
            applied_correction = fraction * correction
        if abs(applied_correction) > 0.0:
            current = Rotation.from_quat([float(source[key]) for key in ("qx", "qy", "qz", "qw")])
            projected = Rotation.from_rotvec(applied_correction * axis) * current
            qx, qy, qz, qw = projected.as_quat()
            row.update({"qw": qw, "qx": qx, "qy": qy, "qz": qz})
            if "source" in row:
                row["source"] = f"{row['source']}+vlm_discrete_orientation_projection"
        output_rows.append(row)
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    _atomic_text(args.output_pose, buffer.getvalue())

    evidence_ids = [str(face["relation_id"]), str(side["relation_id"])]
    record = {
        "schema_version": 1,
        "kind": "vlm_discrete_orientation_projection",
        "continuous_translation_changed": False,
        "input_pose": str(args.input_pose),
        "input_pose_sha256": hashlib.sha256(args.input_pose.read_bytes()).hexdigest(),
        "output_pose": str(args.output_pose),
        "output_pose_sha256": hashlib.sha256(args.output_pose.read_bytes()).hexdigest(),
        "transition": {"start_frame": args.transition_start, "end_frame": args.transition_end},
        "symmetry_switch_frame": args.symmetry_switch_frame,
        "symmetry_correction_deg": float(np.degrees(symmetry_correction)),
        "residual_transition_end": args.residual_transition_end,
        "residual_correction_deg": float(np.degrees(residual_correction)),
        "terminal_frame": args.terminal_frame,
        "selected_face": str(face["label"]),
        "selected_side_exposure": str(side["label"]),
        "confidence": min(float(face["confidence"]), float(side["confidence"])),
        "evidence_ids": evidence_ids,
        "upright_yaw_correction_deg": float(np.degrees(correction)),
        "terminal_alignment_score": float(np.max(scores)),
    }
    _atomic_text(args.provenance, json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(args.output_pose)


if __name__ == "__main__":
    main()
