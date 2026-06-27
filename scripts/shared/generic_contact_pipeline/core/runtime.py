from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import REPO


RUNTIME_CONFIG = REPO / "scripts/shared/generic_contact_pipeline/configs/runtime_envs.yaml"


@dataclass(frozen=True)
class RuntimeEnv:
    name: str
    data: dict[str, Any]

    @property
    def python(self) -> Path:
        return Path(str(self.data["python"])).expanduser()

    @property
    def purpose(self) -> str:
        return str(self.data.get("purpose", ""))

    def doctor(self) -> dict[str, object]:
        return {
            "name": self.name,
            "python": str(self.python),
            "exists": self.python.exists(),
            "purpose": self.purpose,
        }


def load_runtime_env(name: str) -> RuntimeEnv:
    if not RUNTIME_CONFIG.exists():
        raise FileNotFoundError(f"Missing runtime config: {RUNTIME_CONFIG}")
    with RUNTIME_CONFIG.open() as f:
        data = yaml.safe_load(f) or {}
    envs = dict(data.get("envs", {}))
    if name not in envs:
        raise KeyError(f"Unknown runtime env {name!r}; available: {sorted(envs)}")
    return RuntimeEnv(name, dict(envs[name]))


def runtime_python(name: str, *, override_env: str | None = None) -> str:
    if override_env:
        value = os.environ.get(override_env)
        if value:
            return value
    return str(load_runtime_env(name).python)
