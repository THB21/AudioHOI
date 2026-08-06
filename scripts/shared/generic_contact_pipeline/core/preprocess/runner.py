"""Execute the fixed case-ingestion DAG with hash-valid reuse."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import cv2

from ..base.config import CaseProfile
from ..base.io import REPO
from ..base.runtime import runtime_python
from .manifest import (
    existing_artifact_hashes,
    file_sha256,
    ingestion_manifest_record,
    load_and_validate_ingestion_manifest,
    task_cache_key,
    write_ingestion_manifest_atomic,
)
from .registry import build_preprocess_tasks, validate_task_graph
from .types import CaseIngestionResult, PreprocessTask, TaskExecutionRecord


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _video_metadata(video: Path) -> tuple[int, float]:
    if not video.is_file():
        raise FileNotFoundError(f"case ingestion requires {video}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"could not decode case video: {video}")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or fps <= 0.0:
        raise ValueError(f"invalid case video metadata: frames={frame_count}, fps={fps}")
    return frame_count, fps


def _runner_hash(command: tuple[str, ...]) -> str:
    for value in command[1:]:
        path = Path(value)
        if path.is_file() and path.suffix == ".py":
            return file_sha256(path)
    return "external_command"


def _output_paths(
    task: PreprocessTask,
    output_ids: set[str] | None = None,
) -> dict[str, str]:
    return {
        artifact.artifact_id: str(artifact.path)
        for artifact in task.outputs
        if output_ids is None or artifact.artifact_id in output_ids
    }


def _output_kinds(
    task: PreprocessTask,
    output_ids: set[str] | None = None,
) -> dict[str, str]:
    return {
        artifact.artifact_id: artifact.kind
        for artifact in task.outputs
        if output_ids is None or artifact.artifact_id in output_ids
    }


def _safe_output_hashes(task: PreprocessTask) -> dict[str, str]:
    try:
        return existing_artifact_hashes(task.outputs)
    except (FileNotFoundError, ValueError):
        return {}


def _validate_task_outputs(task: PreprocessTask, frame_count: int) -> dict[str, str]:
    hashes = existing_artifact_hashes(task.outputs)
    if task.output_validator is not None:
        task.output_validator(frame_count)
    return hashes


def _record(
    task: PreprocessTask,
    *,
    status: str,
    command: tuple[str, ...],
    cache_key: str,
    input_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
    started_at: str,
    completed_at: str,
    return_code: int,
    stdout_tail: str = "",
    stderr_tail: str = "",
    error: str = "",
) -> TaskExecutionRecord:
    recorded_output_ids = (
        set(output_hashes)
        if status in {"generated", "reused"}
        else None
    )
    return TaskExecutionRecord(
        task_id=task.task_id,
        status=status,
        required=task.required,
        runtime_env=task.runtime_env,
        command=command,
        cache_key=cache_key,
        input_hashes=dict(input_hashes),
        output_paths=_output_paths(task, recorded_output_ids),
        output_kinds=_output_kinds(task, recorded_output_ids),
        output_hashes=dict(output_hashes),
        started_at=started_at,
        completed_at=completed_at,
        return_code=return_code,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        error=error,
        human_state_role=task.human_state_role,
    )


def _ensure_outputs_within_sample(profile: CaseProfile, tasks: tuple[PreprocessTask, ...]) -> None:
    sample = profile.sample_dir.resolve()
    for task in tasks:
        for artifact in task.outputs:
            try:
                artifact.path.resolve().relative_to(sample)
            except ValueError as exc:
                raise ValueError(
                    f"preprocess output escapes sample directory: {task.task_id}:{artifact.path}"
                ) from exc


def run_case_ingestion(profile: CaseProfile) -> CaseIngestionResult:
    frame_count, fps = _video_metadata(profile.sample_dir / "video.mp4")
    tasks = validate_task_graph(build_preprocess_tasks(profile))
    _ensure_outputs_within_sample(profile, tasks)
    accepted_path = profile.result_dir / "case_ingestion_manifest.json"
    attempt_path = profile.result_dir / "case_ingestion_last_attempt.json"
    previous: dict[str, object] | None = None
    previous_invalid = False
    if accepted_path.is_file():
        try:
            previous = load_and_validate_ingestion_manifest(accepted_path)
        except Exception:
            previous_invalid = True
    previous_tasks = {
        str(record.get("task_id")): record
        for record in (previous or {}).get("tasks", [])
        if isinstance(record, dict)
    }
    records: list[TaskExecutionRecord] = []
    completed_ids: set[str] = set()
    for task in tasks:
        started = _now()
        command = tuple(str(value) for value in task.command_builder())
        input_hashes: dict[str, str] = {}
        cache_key = "unavailable"
        if not task.required:
            records.append(
                _record(
                    task,
                    status="disabled",
                    command=command,
                    cache_key="disabled_by_ablation",
                    input_hashes={},
                    output_hashes={},
                    started_at=started,
                    completed_at=_now(),
                    return_code=0,
                    stdout_tail="disabled by disable_audio_events",
                )
            )
            completed_ids.add(task.task_id)
            continue
        unresolved = set(task.dependencies) - completed_ids
        if unresolved:
            error = f"blocked by incomplete dependencies: {sorted(unresolved)}"
            records.append(
                _record(
                    task,
                    status="blocked",
                    command=command,
                    cache_key="blocked",
                    input_hashes={},
                    output_hashes={},
                    started_at=started,
                    completed_at=_now(),
                    return_code=1,
                    error=error,
                )
            )
            break
        try:
            input_hashes = existing_artifact_hashes(task.inputs)
            python = runtime_python(task.runtime_env)
            if not Path(python).is_file():
                raise FileNotFoundError(f"runtime Python is missing for {task.runtime_env}: {python}")
            cache_key = task_cache_key(
                task,
                command=command,
                input_hashes=input_hashes,
                runtime_python=python,
                runner_source_sha256=_runner_hash(command),
            )
            output_hashes: dict[str, str] = {}
            previous_task = previous_tasks.get(task.task_id)
            if not previous_invalid and previous_task and previous_task.get("cache_key") == cache_key:
                output_hashes = _validate_task_outputs(task, frame_count)
                if output_hashes != previous_task.get("output_hashes"):
                    output_hashes = {}
            elif previous is None and not previous_invalid:
                try:
                    output_hashes = _validate_task_outputs(task, frame_count)
                except (FileNotFoundError, ValueError, KeyError):
                    output_hashes = {}
            if output_hashes:
                records.append(
                    _record(
                        task,
                        status="reused",
                        command=command,
                        cache_key=cache_key,
                        input_hashes=input_hashes,
                        output_hashes=output_hashes,
                        started_at=started,
                        completed_at=_now(),
                        return_code=0,
                        stdout_tail="hash-valid cache hit" if previous_task else "validated existing artifact bootstrap",
                    )
                )
                completed_ids.add(task.task_id)
                continue
            result = subprocess.run(
                list(command),
                cwd=REPO,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    (result.stderr or result.stdout or f"return code {result.returncode}")[-4000:]
                )
            output_hashes = _validate_task_outputs(task, frame_count)
            records.append(
                _record(
                    task,
                    status="generated",
                    command=command,
                    cache_key=cache_key,
                    input_hashes=input_hashes,
                    output_hashes=output_hashes,
                    started_at=started,
                    completed_at=_now(),
                    return_code=0,
                    stdout_tail=result.stdout[-4000:],
                    stderr_tail=result.stderr[-4000:],
                )
            )
            completed_ids.add(task.task_id)
        except Exception as exc:
            records.append(
                _record(
                    task,
                    status="failed",
                    command=command,
                    cache_key=cache_key,
                    input_hashes=input_hashes,
                    output_hashes=_safe_output_hashes(task),
                    started_at=started,
                    completed_at=_now(),
                    return_code=1,
                    error=str(exc),
                )
            )
            break
    accepted = len(records) == len(tasks) and all(
        record.status in {"generated", "reused", "disabled"} for record in records
    )
    manifest = ingestion_manifest_record(
        case_name=profile.case_name,
        sample_dir=profile.sample_dir,
        result_dir=profile.result_dir,
        status="accepted" if accepted else "failed",
        frame_count=frame_count,
        fps=fps,
        tasks=records,
    )
    write_ingestion_manifest_atomic(attempt_path, manifest)
    if accepted:
        write_ingestion_manifest_atomic(accepted_path, manifest)
    return CaseIngestionResult(
        status="accepted" if accepted else "failed",
        manifest_path=accepted_path if accepted else attempt_path,
        frame_count=frame_count,
        fps=fps,
        tasks=tuple(records),
    )


def validate_case_ingestion_current(profile: CaseProfile) -> dict[str, object]:
    """Validate both output hashes and current task/input/cache identities."""

    manifest_path = profile.result_dir / "case_ingestion_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"case ingestion manifest is missing: {manifest_path}")
    payload = load_and_validate_ingestion_manifest(manifest_path)
    current_tasks = validate_task_graph(build_preprocess_tasks(profile))
    recorded = {
        str(item.get("task_id")): item
        for item in payload.get("tasks", [])
        if isinstance(item, dict)
    }
    for task in current_tasks:
        item = recorded.get(task.task_id)
        if item is None:
            raise ValueError(f"case ingestion manifest is missing task: {task.task_id}")
        if not task.required:
            if item.get("status") != "disabled":
                raise ValueError(f"disabled ingestion task has stale status: {task.task_id}")
            continue
        command = tuple(str(value) for value in task.command_builder())
        inputs = existing_artifact_hashes(task.inputs)
        python = runtime_python(task.runtime_env)
        current_key = task_cache_key(
            task,
            command=command,
            input_hashes=inputs,
            runtime_python=python,
            runner_source_sha256=_runner_hash(command),
        )
        if item.get("cache_key") != current_key:
            raise ValueError(f"case ingestion task cache is stale: {task.task_id}")
    return payload
