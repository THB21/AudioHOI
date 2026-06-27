from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..base.io import REPO, repo_path


CONFIG_DIR = REPO / "scripts/shared/generic_contact_pipeline/configs/cases"


@dataclass(frozen=True)
class CaseProfile:
    data: dict[str, Any]

    @property
    def case_name(self) -> str:
        return str(self.data["case_name"])

    @property
    def sample_dir(self) -> Path:
        return repo_path(self.data["sample_dir"])

    @property
    def result_name(self) -> str:
        return str(self.data.get("result_name", "generic_pipeline_v1"))

    @property
    def result_dir(self) -> Path:
        return self.sample_dir / "results" / self.result_name

    @property
    def render_dir(self) -> Path:
        return self.sample_dir / "results" / "renders" / self.result_name

    @property
    def baseline(self) -> dict[str, str]:
        return dict(self.data.get("baseline", {}))

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
        value = self.data.get(key)
        if value is None:
            raise KeyError(f"Missing component key {key!r} in case profile {self.case_name}")
        return str(value)

    def refinement_policies(self) -> list[str]:
        value = self.data.get("refinement_policy", [])
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value]


def load_case_profile(case_name: str) -> CaseProfile:
    path = CONFIG_DIR / f"{case_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown case {case_name!r}; missing {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return CaseProfile(data)


def with_runtime_overrides(profile: CaseProfile, *, result_name: str | None = None, ablation_flags: list[str] | None = None) -> CaseProfile:
    data = deepcopy(profile.data)
    if result_name:
        data["result_name"] = result_name
    if ablation_flags:
        data["ablation_flags"] = list(ablation_flags)
    return CaseProfile(data)


def available_cases() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))
