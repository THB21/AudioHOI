from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import REPO


PROVIDER_CONFIG = REPO / "scripts/shared/generic_contact_pipeline/configs/llm_provider.yaml"


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (REPO / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class LLMProvider:
    name: str
    data: dict[str, Any]

    @property
    def kind(self) -> str:
        return str(self.data.get("kind", "mistral_chat_completions"))

    @property
    def api_key_env(self) -> str:
        return str(self.data.get("api_key_env", "MISTRAL_API_KEY"))

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @property
    def model_id(self) -> str:
        env_key = str(self.data.get("model_id_env", "") or "")
        if env_key and os.environ.get(env_key):
            return str(os.environ[env_key])
        return str(self.data.get("model_id", "mistral-small-latest"))

    @property
    def base_url(self) -> str:
        env_key = str(self.data.get("base_url_env", "") or "")
        if env_key and os.environ.get(env_key):
            return str(os.environ[env_key]).rstrip("/")
        return str(self.data.get("base_url", "https://api.mistral.ai/v1")).rstrip("/")

    @property
    def temperature(self) -> float:
        return float(self.data.get("temperature", 0.0))

    @property
    def max_new_tokens(self) -> int:
        return int(self.data.get("max_new_tokens", 900))

    @property
    def timeout_seconds(self) -> int:
        return int(self.data.get("timeout_seconds", 120))

    def doctor(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(self.api_key),
            "model_id": self.model_id,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "timeout_seconds": self.timeout_seconds,
        }


def load_llm_provider(name: str | None = None) -> LLMProvider:
    load_dotenv()
    if not PROVIDER_CONFIG.exists():
        raise FileNotFoundError(f"Missing LLM provider config: {PROVIDER_CONFIG}")
    with PROVIDER_CONFIG.open() as f:
        data = yaml.safe_load(f) or {}
    provider_name = name or str(data.get("default_provider", "mistral_api"))
    providers = dict(data.get("providers", {}))
    if provider_name not in providers:
        raise KeyError(f"Unknown LLM provider {provider_name!r}; available: {sorted(providers)}")
    return LLMProvider(provider_name, dict(providers[provider_name]))
