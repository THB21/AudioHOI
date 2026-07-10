from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...base.io import REPO, repo_path
from ...base.config import load_case_profile


DEFAULT_FINAL_RESULT_MANIFEST = REPO / "final_result" / "evaluation_manifest.json"


def _path(value: object) -> Path | None:
    if value in {None, ""}:
        return None
    return repo_path(str(value))


def _csv_rows(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _video_frames(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        value = subprocess.check_output(command, text=True).strip()
        return int(value) if value and value != "N/A" else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


@dataclass(frozen=True)
class FinalResultProfile:
    entry: dict[str, Any]
    output_root: Path

    @property
    def case_name(self) -> str:
        return str(self.entry["case"])

    @property
    def sample_dir(self) -> Path:
        return _path(self.entry.get("sample_dir")) or REPO

    @property
    def result_name(self) -> str:
        return "canonical_final_result"

    @property
    def result_dir(self) -> Path:
        return self.output_root / self.case_name

    @property
    def render_dir(self) -> Path:
        return REPO / "final_result" / "videos"

    @property
    def data(self) -> dict[str, Any]:
        base = dict(load_case_profile(self.case_name).data)
        base["result_name"] = self.result_name
        base["evaluation_source"] = dict(self.entry)
        return base

    @property
    def camera(self) -> dict[str, float]:
        raw = self.data.get("camera", {})
        return {
            "fx": float(raw.get("fx", 1468.604736328125)),
            "fy": float(raw.get("fy", 1468.604736328125)),
            "cx": float(raw.get("cx", 640.0)),
            "cy": float(raw.get("cy", 360.0)),
        }

    def component(self, key: str) -> str:
        return str(self.data.get(key, ""))


def load_final_result_profiles(
    manifest_path: Path = DEFAULT_FINAL_RESULT_MANIFEST,
    *,
    output_root: Path | None = None,
    cases: list[str] | None = None,
) -> list[FinalResultProfile]:
    payload = json.loads(manifest_path.read_text())
    selected = set(cases or [])
    root = output_root or (REPO / "final_result" / "evaluation")
    return [
        FinalResultProfile(dict(entry), root)
        for entry in payload.get("entries", [])
        if (selected and str(entry.get("case")) in selected)
        or (not selected and bool(entry.get("evaluate", True)))
    ]


def validate_final_result_profile(profile: FinalResultProfile) -> dict[str, Any]:
    entry = profile.entry
    final_video = _path(entry.get("final_video"))
    source_video = _path(entry.get("source_video"))
    pose_csv = _path(entry.get("object_pose_csv"))
    human_params = _path(entry.get("human_params"))
    contact_points = _path(entry.get("contact_points_csv"))
    gate_trace_dir = _path(entry.get("gate_trace_result_dir"))
    final_frames = _video_frames(final_video)
    source_frames = _video_frames(source_video)
    pose_frames = _csv_rows(pose_csv)
    final_pose_aligned = final_frames is not None and pose_frames is not None and final_frames == pose_frames
    source_pose_aligned = source_frames is not None and pose_frames is not None and source_frames == pose_frames
    hard_metrics_ready = bool(
        final_pose_aligned
        and source_pose_aligned
        and human_params is not None
        and human_params.exists()
        and contact_points is not None
        and contact_points.exists()
    )
    missing = []
    for label, path in (
        ("final_video", final_video),
        ("source_video", source_video),
        ("object_pose_csv", pose_csv),
        ("human_params", human_params),
        ("contact_points_csv", contact_points),
        ("gate_trace_result_dir", gate_trace_dir),
    ):
        if path is None or not path.exists():
            missing.append(label)
    return {
        "case": profile.case_name,
        "declared_status": entry.get("status", ""),
        "final_video": str(final_video) if final_video else "",
        "source_video": str(source_video) if source_video else "",
        "object_pose_csv": str(pose_csv) if pose_csv else "",
        "human_params": str(human_params) if human_params else "",
        "contact_points_csv": str(contact_points) if contact_points else "",
        "gate_trace_result_dir": str(gate_trace_dir) if gate_trace_dir else "",
        "gate_trace_ready": bool(
            gate_trace_dir
            and (gate_trace_dir / "vlm_trace" / "04_gating" / "gate_timeline.csv").exists()
            and (gate_trace_dir / "optimizer_decisions.csv").exists()
            and (gate_trace_dir / "object_pose_pre_smooth.csv").exists()
        ),
        "final_video_frames": final_frames,
        "source_video_frames": source_frames,
        "object_pose_frames": pose_frames,
        "final_pose_frame_aligned": final_pose_aligned,
        "source_pose_frame_aligned": source_pose_aligned,
        "hard_metrics_ready": hard_metrics_ready,
        "missing_artifacts": missing,
        "note": entry.get("note", ""),
    }
