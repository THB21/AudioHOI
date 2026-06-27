from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import REPO, repo_path


PROVIDER_CONFIG = REPO / "scripts/shared/generic_contact_pipeline/configs/vlm_provider.yaml"


@dataclass(frozen=True)
class VLMProvider:
    name: str
    data: dict[str, Any]

    @property
    def kind(self) -> str:
        return str(self.data.get("kind", "qwen_vl"))

    @property
    def python(self) -> Path:
        return Path(str(self.data.get("python", sys.executable))).expanduser()

    @property
    def model_id(self) -> str:
        return str(self.data.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"))

    @property
    def local_dir(self) -> Path | None:
        value = self.data.get("local_dir")
        if not value:
            return None
        return repo_path(value)

    @property
    def device_map(self) -> str:
        return str(self.data.get("device_map", "auto"))

    @property
    def load_4bit(self) -> bool:
        return bool(self.data.get("load_4bit", False))

    @property
    def resize_max(self) -> int:
        return int(self.data.get("resize_max", 960))

    @property
    def max_new_tokens(self) -> int:
        return int(self.data.get("max_new_tokens", 80))

    @property
    def env(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in dict(self.data.get("env", {})).items()}

    def is_current_python(self) -> bool:
        try:
            return self.python.resolve() == Path(sys.executable).resolve()
        except Exception:
            return str(self.python) == sys.executable

    def apply_env_defaults(self) -> None:
        for key, value in self.env.items():
            os.environ.setdefault(key, value)

    def doctor(self) -> dict[str, object]:
        local_dir = self.local_dir
        return {
            "name": self.name,
            "kind": self.kind,
            "configured_python": str(self.python),
            "current_python": sys.executable,
            "is_current_python": self.is_current_python(),
            "python_exists": self.python.exists(),
            "model_id": self.model_id,
            "local_dir": str(local_dir) if local_dir else "",
            "local_dir_exists": bool(local_dir and local_dir.exists()),
            "load_4bit": self.load_4bit,
            "resize_max": self.resize_max,
            "max_new_tokens": self.max_new_tokens,
            "env": self.env,
        }


def load_vlm_provider(name: str | None = None) -> VLMProvider:
    if not PROVIDER_CONFIG.exists():
        raise FileNotFoundError(f"Missing VLM provider config: {PROVIDER_CONFIG}")
    with PROVIDER_CONFIG.open() as f:
        data = yaml.safe_load(f) or {}
    provider_name = name or str(data.get("default_provider", "qwen_vl_local"))
    providers = dict(data.get("providers", {}))
    if provider_name not in providers:
        raise KeyError(f"Unknown VLM provider {provider_name!r}; available: {sorted(providers)}")
    return VLMProvider(provider_name, dict(providers[provider_name]))
