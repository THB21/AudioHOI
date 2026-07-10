from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...base.config import CaseProfile


@dataclass(frozen=True)
class EvaluationPaths:
    sample_dir: Path
    result_dir: Path
    render_dir: Path
    evaluation_dir: Path
    hoi_eval_json: Path

    @classmethod
    def from_profile(cls, profile: CaseProfile) -> "EvaluationPaths":
        return cls(
            sample_dir=profile.sample_dir,
            result_dir=profile.result_dir,
            render_dir=profile.render_dir,
            evaluation_dir=profile.result_dir / "evaluation",
            hoi_eval_json=profile.sample_dir / "results" / "hoi_eval" / "hoi_interaction_metrics.json",
        )


@dataclass
class MetricBlock:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
