from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...base.config import CaseProfile


@dataclass(frozen=True)
class EvaluationPaths:
    sample_dir: Path
    result_dir: Path
    gate_trace_dir: Path
    render_dir: Path
    evaluation_dir: Path
    hoi_eval_json: Path
    final_video: Path | None
    source_video: Path | None
    object_pose_csv: Path
    contact_points_csv: Path
    audio_contact_csv: Path
    anchor_state_csv: Path
    observed_mask_dirs: tuple[Path, ...]
    camera: dict[str, float]

    @classmethod
    def from_profile(cls, profile: CaseProfile) -> "EvaluationPaths":
        source = dict(profile.data.get("evaluation_source", {}))
        def source_path(key: str, fallback: Path) -> Path:
            value = source.get(key)
            return Path(value) if value else fallback

        # Object-stage ablations may intentionally carry different event
        # streams (for example, the no-audio variant).  Prefer the event
        # contract materialized beside that variant before falling back to the
        # legacy sample-level human/audio result.
        local_audio_contacts = profile.result_dir / "audio_contact_events.csv"
        default_audio_contacts = (
            local_audio_contacts
            if local_audio_contacts.exists()
            else profile.sample_dir / "results" / "human_audio_semantics" / "contact_records.csv"
        )

        return cls(
            sample_dir=profile.sample_dir,
            result_dir=profile.result_dir,
            gate_trace_dir=source_path("gate_trace_result_dir", profile.result_dir),
            render_dir=profile.render_dir,
            evaluation_dir=profile.result_dir / "evaluation",
            hoi_eval_json=source_path("hoi_metrics_json", profile.sample_dir / "results" / "hoi_eval" / "hoi_interaction_metrics.json"),
            final_video=Path(source["final_video"]) if source.get("final_video") else None,
            source_video=Path(source["source_video"]) if source.get("source_video") else None,
            object_pose_csv=source_path("object_pose_csv", profile.result_dir / "object_pose.csv"),
            contact_points_csv=source_path("contact_points_csv", profile.result_dir / "object_contact_points.csv"),
            audio_contact_csv=source_path(
                "audio_contact_csv",
                default_audio_contacts,
            ),
            anchor_state_csv=source_path("anchor_state_csv", profile.result_dir / "anchor_state.csv"),
            observed_mask_dirs=tuple(
                [Path(source["observed_mask_dir"])] if source.get("observed_mask_dir") else [
                    profile.sample_dir / "results" / "segmentation" / "masks",
                    profile.result_dir / "segmentation" / "masks",
                    profile.result_dir / "object_masks",
                ]
            ),
            camera=getattr(
                profile,
                "camera",
                {"fx": 1468.604736328125, "fy": 1468.604736328125, "cx": 640.0, "cy": 360.0},
            ),
        )


@dataclass
class MetricBlock:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
