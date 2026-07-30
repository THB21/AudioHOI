"""Hashing and atomic manifest I/O for case ingestion."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from .types import ArtifactSpec, PreprocessTask, TaskExecutionRecord


INGESTION_SCHEMA_VERSION = 1


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(artifact: ArtifactSpec) -> str:
    path = artifact.path
    if artifact.kind == "file":
        if not path.is_file():
            raise FileNotFoundError(path)
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    records = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        records.append((child.relative_to(path).as_posix(), file_sha256(child)))
    if not records:
        raise ValueError(f"preprocess output directory is empty: {path}")
    return canonical_sha256(records)


def existing_artifact_hashes(artifacts: Sequence[ArtifactSpec]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for artifact in artifacts:
        if artifact.path.exists():
            hashes[artifact.artifact_id] = artifact_sha256(artifact)
        elif artifact.required:
            raise FileNotFoundError(artifact.path)
    return hashes


def task_cache_key(
    task: PreprocessTask,
    *,
    command: Sequence[str],
    input_hashes: Mapping[str, str],
    runtime_python: str,
    runner_source_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": INGESTION_SCHEMA_VERSION,
            "task_id": task.task_id,
            "inputs": dict(sorted(input_hashes.items())),
            "config": dict(task.config_fingerprint),
            "command": list(command),
            "runtime_env": task.runtime_env,
            "runtime_python": runtime_python,
            "runner_source_sha256": runner_source_sha256,
            "model_identity": dict(task.model_identity),
            "human_state_role": task.human_state_role,
        }
    )


def write_ingestion_manifest_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def ingestion_manifest_record(
    *,
    case_name: str,
    sample_dir: Path,
    result_dir: Path,
    status: str,
    frame_count: int,
    fps: float,
    tasks: Sequence[TaskExecutionRecord],
) -> dict[str, object]:
    records = [task.as_record() for task in tasks]
    return {
        "schema_version": INGESTION_SCHEMA_VERSION,
        "case_name": case_name,
        "sample_dir": str(sample_dir),
        "result_dir": str(result_dir),
        "status": status,
        "frame_count": frame_count,
        "fps": fps,
        "tasks": records,
        "canonical_sha256": canonical_sha256(records),
    }


def validate_ingestion_manifest(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != INGESTION_SCHEMA_VERSION:
        raise ValueError("unsupported case-ingestion manifest schema")
    if payload.get("status") != "accepted":
        raise ValueError("case-ingestion manifest is not accepted")
    if int(payload.get("frame_count", 0)) <= 0 or float(payload.get("fps", 0.0)) <= 0.0:
        raise ValueError("case-ingestion manifest has invalid frame count or FPS")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("case-ingestion manifest has no tasks")
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            raise ValueError("case-ingestion task record must be an object")
        status = raw.get("status")
        if raw.get("required", True) and status not in {"generated", "reused"}:
            raise ValueError(f"required ingestion task did not complete: {raw.get('task_id')}")
        hashes = raw.get("output_hashes")
        if status in {"generated", "reused"} and (not isinstance(hashes, dict) or not hashes):
            raise ValueError(f"ingestion task has no output hashes: {raw.get('task_id')}")
        paths = raw.get("output_paths")
        kinds = raw.get("output_kinds")
        if status in {"generated", "reused"}:
            if not isinstance(paths, dict) or not isinstance(kinds, dict):
                raise ValueError(f"ingestion task has no output artifact ledger: {raw.get('task_id')}")
            if set(paths) != set(hashes) or set(kinds) != set(hashes):
                raise ValueError(f"ingestion task output ledger keys disagree: {raw.get('task_id')}")
            for artifact_id, expected in hashes.items():
                actual = artifact_sha256(
                    ArtifactSpec(
                        str(artifact_id),
                        Path(str(paths[artifact_id])),
                        kind=str(kinds[artifact_id]),
                    )
                )
                if actual != expected:
                    raise ValueError(
                        f"ingestion output hash mismatch: {raw.get('task_id')}:{artifact_id}"
                    )
        if raw.get("task_id") == "gvhmr" and raw.get("human_state_role") != "read_only_observed":
            raise ValueError("GVHMR ingestion task must remain read-only observed state")


def load_and_validate_ingestion_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("case-ingestion manifest root must be an object")
    validate_ingestion_manifest(payload)
    return payload
