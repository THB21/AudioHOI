"""Typed contracts for deterministic case-ingestion tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence


CommandBuilder = Callable[[], Sequence[str]]


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    path: Path
    kind: str = "file"
    required: bool = True

    def __post_init__(self) -> None:
        if not self.artifact_id or self.kind not in {"file", "directory"}:
            raise ValueError("preprocess artifacts require an id and file/directory kind")
        if ".." in self.path.parts:
            raise ValueError(f"unsafe preprocess artifact path: {self.path}")


@dataclass(frozen=True)
class PreprocessTask:
    task_id: str
    runtime_env: str
    dependencies: tuple[str, ...]
    inputs: tuple[ArtifactSpec, ...]
    outputs: tuple[ArtifactSpec, ...]
    command_builder: CommandBuilder
    config_fingerprint: Mapping[str, object] = field(default_factory=dict)
    model_identity: Mapping[str, object] = field(default_factory=dict)
    required: bool = True
    human_state_role: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.runtime_env:
            raise ValueError("preprocess tasks require task_id and runtime_env")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"duplicate dependencies for task {self.task_id}")
        if self.task_id in self.dependencies:
            raise ValueError(f"task {self.task_id} cannot depend on itself")
        if self.required and not self.outputs:
            raise ValueError(f"required preprocess task {self.task_id} has no outputs")
        output_ids = [artifact.artifact_id for artifact in self.outputs]
        if len(set(output_ids)) != len(output_ids):
            raise ValueError(f"duplicate output artifacts for task {self.task_id}")
        if self.human_state_role not in {None, "read_only_observed"}:
            raise ValueError("preprocess human state can only be read_only_observed")


@dataclass(frozen=True)
class TaskExecutionRecord:
    task_id: str
    status: str
    required: bool
    runtime_env: str
    command: tuple[str, ...]
    cache_key: str
    input_hashes: Mapping[str, str]
    output_paths: Mapping[str, str]
    output_kinds: Mapping[str, str]
    output_hashes: Mapping[str, str]
    started_at: str
    completed_at: str
    return_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""
    human_state_role: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"generated", "reused", "disabled", "failed", "blocked"}:
            raise ValueError(f"unknown preprocess task status: {self.status}")
        if self.status == "failed" and not self.error:
            raise ValueError("failed preprocess task requires an error")
        if self.status in {"generated", "reused"} and self.return_code != 0:
            raise ValueError("successful preprocess task must have return code zero")

    def as_record(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "required": self.required,
            "runtime_env": self.runtime_env,
            "command": list(self.command),
            "cache_key": self.cache_key,
            "input_hashes": dict(self.input_hashes),
            "output_paths": dict(self.output_paths),
            "output_kinds": dict(self.output_kinds),
            "output_hashes": dict(self.output_hashes),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "return_code": self.return_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
            "human_state_role": self.human_state_role,
        }


@dataclass(frozen=True)
class CaseIngestionResult:
    status: str
    manifest_path: Path
    frame_count: int
    fps: float
    tasks: tuple[TaskExecutionRecord, ...]

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "failed"}:
            raise ValueError("case ingestion status must be accepted or failed")
        if self.frame_count <= 0 or self.fps <= 0.0:
            raise ValueError("case ingestion requires positive frame count and FPS")
