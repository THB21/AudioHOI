from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..base.io import REPO


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

    def _complete_claude_cli(self, system: str, user: str) -> tuple[str | None, str]:
        """Route through the local ``claude`` CLI in headless print mode --- uses the user's
        Claude subscription, no API key. Runs from a neutral cwd (``/tmp``) so NO project
        CLAUDE.md, memory, or conversation context is loaded: a fresh Claude with no background
        knowledge that judges only the data provided in the prompt."""
        import subprocess

        prompt = f"{system}\n\n{user}" if system else user
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
        if self.model_id and self.model_id != "mistral-small-latest":
            cmd += ["--model", self.model_id]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_seconds, cwd="/tmp",
            )
            if proc.returncode != 0:
                return None, f"claude cli exit {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
            out = proc.stdout.strip()
            return (out, "") if out else (None, "empty claude cli output")
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    def complete(self, system: str, user: str) -> tuple[str | None, str]:
        """Return (assistant_text, error_string). Dispatches on ``kind``: local ``claude`` CLI
        (subscription, no key) for kind in {claude_cli, claude_code}; Anthropic Claude Messages
        API for {anthropic_messages, claude, anthropic}; otherwise the OpenAI/Mistral
        chat-completions format. Text-only; used for the discrete data-audit."""
        if self.kind in ("claude_cli", "claude_code"):
            return self._complete_claude_cli(system, user)

        import requests

        if not self.api_key:
            return None, f"missing API key {self.api_key_env}"
        try:
            if self.kind in ("anthropic_messages", "claude", "anthropic"):
                resp = requests.post(
                    self.base_url.rstrip("/") + "/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model_id,
                        "max_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                        "system": system,
                        "messages": [{"role": "user", "content": user}],
                    },
                    timeout=self.timeout_seconds,
                )
                if resp.status_code >= 400:
                    return None, f"anthropic status {resp.status_code}: {resp.text[:500]}"
                parts = resp.json().get("content", [])
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                return text, ""
            # OpenAI / Mistral chat-completions
            resp = requests.post(
                self.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_new_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout_seconds,
            )
            if resp.status_code >= 400:
                return None, f"{self.kind} status {resp.status_code}: {resp.text[:500]}"
            text = str(resp.json().get("choices", [{}])[0].get("message", {}).get("content", ""))
            return text, ""
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

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
