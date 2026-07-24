from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from ..base.config import CONFIG_DIR, load_case_profile, with_runtime_overrides
from ..base.io import REPO, repo_relative_value
from ..base.schema import REQUIRED_RENDER_FILES, stage_paths
from .attempts import STAGE_ARTIFACT_KEYS


CANONICAL_CASES = ("basketball", "football", "mug", "chair", "stick")
CANONICAL_RESULT_NAME = "benchmark_vlm_qwen"
DEFAULT_GOLDEN_MANIFEST = REPO / "tests" / "golden" / "pipeline_v1_five_cases.json"
DEFAULT_RUNTIME_INPUT_MANIFEST = REPO / "tests" / "golden" / "pipeline_v1_runtime_inputs.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        rel = item.relative_to(path).as_posix()
        item_hash = _sha256_file(item)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
        count += 1
        size += item.stat().st_size
    return digest.hexdigest(), count, size


def artifact_record(
    path: Path,
    *,
    logical_path: str | None = None,
    source_scope: str = "repository",
) -> dict[str, object]:
    rel = logical_path or str(repo_relative_value(path))
    if path.is_file():
        record: dict[str, object] = {
            "path": rel,
            "source_scope": source_scope,
            "kind": "file",
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if path.suffix.lower() == ".csv":
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                record["columns"] = list(reader.fieldnames or [])
                record["rows"] = sum(1 for _ in reader)
        return record
    if path.is_dir():
        digest, count, size = _sha256_directory(path)
        return {
            "path": rel,
            "source_scope": source_scope,
            "kind": "directory",
            "exists": True,
            "file_count": count,
            "size_bytes": size,
            "sha256": digest,
        }
    return {
        "path": rel,
        "source_scope": source_scope,
        "kind": "missing",
        "exists": False,
    }


def _repo_path_from_recorded(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return REPO / path
    parts = path.parts
    for marker in ("samples_known_object", "video_sample", "final_result"):
        if marker in parts:
            return REPO.joinpath(*parts[parts.index(marker) :])
    return path


def _input_records(profile: Any, *, input_root: Path | None = None) -> list[dict[str, object]]:
    paths: list[Path] = [
        CONFIG_DIR / f"{profile.case_name}.yaml",
        profile.sample_dir / "metadata.json",
        profile.sample_dir / "video.mp4",
        profile.sample_dir / "audio.wav",
    ]
    for value in profile.baseline.values():
        paths.append(_repo_path_from_recorded(str(value)))
    for key in ("articraft_model_py", "articraft_urdf"):
        if profile.data.get(key):
            paths.append(_repo_path_from_recorded(str(profile.data[key])))
    stage0_manifest = stage_paths(profile)["stage0_inputs_manifest"]
    if stage0_manifest.exists():
        payload = json.loads(stage0_manifest.read_text())
        for item in payload.get("prepared_inputs", {}).values():
            if isinstance(item, dict) and item.get("path"):
                paths.append(_repo_path_from_recorded(str(item["path"])))
    unique = {str(repo_relative_value(path)): path for path in paths}
    records: list[dict[str, object]] = []
    for logical_path in sorted(unique):
        source_path = unique[logical_path]
        source_scope = "repository"
        if not source_path.exists() and input_root is not None and not Path(logical_path).is_absolute():
            fallback = input_root / logical_path
            if fallback.exists():
                source_path = fallback
                source_scope = "input_root"
        records.append(
            artifact_record(
                source_path,
                logical_path=logical_path,
                source_scope=source_scope,
            )
        )
    return records


def _stage_records(profile: Any, stage_name: str) -> list[dict[str, object]]:
    paths = stage_paths(profile)
    candidates = [paths[key] for key in STAGE_ARTIFACT_KEYS.get(stage_name, ())]
    if stage_name in {"stage0", "stage1", "stage2", "stage3", "stage4", "stage5"}:
        candidates.extend(
            [
                paths["vlm_dir"] / stage_name / "vlm_gates.csv",
                paths["stage_audit_dir"] / stage_name / "stage_audit_gates.csv",
            ]
        )
    return [artifact_record(path) for path in candidates if path.exists()]


def decoded_video_record(path: Path) -> dict[str, object]:
    probe = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,pix_fmt",
            "-of", "json", str(path),
        ],
        text=True,
    )
    stream = (json.loads(probe).get("streams") or [{}])[0]
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    digest = hashlib.sha256()
    decoded_bytes = 0
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
        decoded_bytes += len(chunk)
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return {
        **artifact_record(path),
        "decoded": {
            "sha256_rgb24": digest.hexdigest(),
            "decoded_bytes": decoded_bytes,
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
            "source_pix_fmt": stream.get("pix_fmt", ""),
            "hash_pix_fmt": "rgb24",
        },
    }


def capture_golden_manifest(
    *,
    cases: Iterable[str] = CANONICAL_CASES,
    result_name: str = CANONICAL_RESULT_NAME,
    base_commit: str = "53ae3a3470cbef0df47f5d4d8427328c85566627",
    input_root: Path | None = None,
) -> dict[str, object]:
    case_records: dict[str, object] = {}
    for case_name in cases:
        profile = with_runtime_overrides(load_case_profile(case_name), result_name=result_name)
        paths = stage_paths(profile)
        renders = [profile.render_dir / rel for rel in REQUIRED_RENDER_FILES]
        missing = [str(repo_relative_value(path)) for path in renders if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{case_name}: missing canonical renders: {missing}")
        stages = {
            stage_name: _stage_records(profile, stage_name)
            for stage_name in ("stage-1", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5")
        }
        pipeline_manifest_path = paths["pipeline_manifest"]
        pipeline_manifest = json.loads(pipeline_manifest_path.read_text())
        manifest_profile = pipeline_manifest.get("profile", {})
        case_records[case_name] = {
            "result_name": result_name,
            "result_dir": repo_relative_value(profile.result_dir),
            "render_dir": repo_relative_value(profile.render_dir),
            "inputs": _input_records(profile, input_root=input_root),
            "recorded_execution": {
                "pipeline_manifest": artifact_record(pipeline_manifest_path),
                "vlm_mode": pipeline_manifest.get("vlm_mode", ""),
                "llm_mode": pipeline_manifest.get("llm_mode", ""),
                "ablation_flags": manifest_profile.get("ablation_flags", [])
                if isinstance(manifest_profile, dict)
                else [],
                "note": "Observed historical metadata; result-directory names are not treated as proof of an enabled mechanism.",
            },
            "stages": stages,
            "contact_and_gate_state": [
                artifact_record(path)
                for path in (
                    paths["contact_candidates"], paths["anchor_state"], paths["contact_state"],
                    paths["vlm_dir"] / "vlm_gates.csv",
                    paths["stage_audit_dir"] / "stage_audit_gates.csv",
                )
                if path.exists()
            ],
            "outputs": {
                "pose": artifact_record(paths["object_pose"]),
                "phase": artifact_record(paths["object_phase"]),
                "decoded_renders": [decoded_video_record(path) for path in renders],
            },
        }
    return {
        "schema_version": 1,
        "purpose": "Phase 0 regression lock; hashes describe existing results and do not endorse solver semantics.",
        "base_commit": base_commit,
        "canonical_result_name": result_name,
        "input_root_policy": "Records with source_scope=input_root require a read-only external data root at verification time.",
        "cases": case_records,
    }


def _iter_artifact_records(value: Any):
    if isinstance(value, dict):
        if "path" in value and "exists" in value and "kind" in value:
            yield value
        for item in value.values():
            yield from _iter_artifact_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_artifact_records(item)


def verify_golden_manifest(
    manifest: dict[str, object],
    *,
    verify_decoded_renders: bool = True,
    input_root: Path | None = None,
    exclude_paths: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append(f"schema_version expected 1, got {manifest.get('schema_version')!r}")
    cases = manifest.get("cases", {})
    if not isinstance(cases, dict) or tuple(cases) != CANONICAL_CASES:
        actual_cases = tuple(cases) if isinstance(cases, dict) else type(cases).__name__
        errors.append(f"canonical cases expected {CANONICAL_CASES!r}, got {actual_cases!r}")
    seen: set[str] = set()
    for record in _iter_artifact_records(cases):
        path_value = str(record["path"])
        if exclude_paths and path_value in exclude_paths:
            continue
        source_scope = str(record.get("source_scope", "repository"))
        identity = f"{source_scope}:{path_value}:{record.get('sha256')}"
        if identity in seen:
            continue
        seen.add(identity)
        if source_scope == "input_root":
            if input_root is None:
                errors.append(f"{path_value}: verification requires --input-root")
                continue
            path = input_root / path_value
        else:
            path = _repo_path_from_recorded(path_value)
        actual = artifact_record(path, logical_path=path_value, source_scope=source_scope)
        for key in ("exists", "kind", "sha256", "size_bytes", "file_count", "columns", "rows"):
            if actual.get(key) != record.get(key):
                errors.append(f"{path_value}: {key} expected {record.get(key)!r}, got {actual.get(key)!r}")
        if verify_decoded_renders and record.get("decoded") and path.is_file():
            decoded = decoded_video_record(path).get("decoded", {})
            expected_decoded = record["decoded"]
            if decoded != expected_decoded:
                errors.append(f"{path_value}: decoded render hash/metadata changed")
    return errors


def verify_runtime_input_manifest(manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append(f"runtime input schema_version expected 1, got {manifest.get('schema_version')!r}")
    cases = manifest.get("cases", {})
    if not isinstance(cases, dict):
        return errors + ["runtime input cases must be an object"]
    for record in _iter_artifact_records(cases):
        path_value = str(record["path"])
        actual = artifact_record(_repo_path_from_recorded(path_value), logical_path=path_value)
        for key in ("exists", "kind", "sha256", "size_bytes", "file_count", "columns", "rows"):
            if actual.get(key) != record.get(key):
                errors.append(f"{path_value}: {key} expected {record.get(key)!r}, got {actual.get(key)!r}")
    return errors


def manifest_artifact_paths(manifest: dict[str, object]) -> set[str]:
    """Return logical paths owned by a manifest.

    Supplemental runtime-input manifests intentionally override stale
    missing/empty input records in the frozen Phase 0 result manifest.
    """
    return {str(record["path"]) for record in _iter_artifact_records(manifest.get("cases", {}))}


def sync_golden_inputs(
    manifest: dict[str, object],
    *,
    source_root: Path,
    destination_root: Path = REPO,
    apply: bool = False,
    exclude_paths: set[str] | None = None,
) -> dict[str, object]:
    copied: list[str] = []
    would_copy: list[str] = []
    verified: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    cases = manifest.get("cases", {})
    if not isinstance(cases, dict):
        return {"copied": [], "would_copy": [], "verified": [], "errors": ["manifest cases must be an object"]}
    for case in cases.values():
        if not isinstance(case, dict):
            continue
        inputs = case.get("inputs", [])
        if not isinstance(inputs, list):
            continue
        for record in inputs:
            if not isinstance(record, dict) or not record.get("exists"):
                continue
            logical_path = str(record.get("path", ""))
            if exclude_paths and logical_path in exclude_paths:
                continue
            if logical_path in seen:
                continue
            seen.add(logical_path)
            relative = Path(logical_path)
            if not logical_path or relative.is_absolute() or ".." in relative.parts:
                errors.append(f"unsafe input path {logical_path!r}")
                continue
            destination = destination_root / relative
            if destination.exists():
                current = artifact_record(destination, logical_path=logical_path)
                if current.get("sha256") == record.get("sha256"):
                    verified.append(logical_path)
                    continue
                source = source_root / relative
                if record.get("kind") == "directory" and destination.is_dir() and source.is_dir():
                    source_record = artifact_record(source, logical_path=logical_path)
                    if source_record.get("sha256") != record.get("sha256"):
                        errors.append(f"{logical_path}: source hash does not match golden manifest")
                        continue
                    conflicts = []
                    for existing in sorted(item for item in destination.rglob("*") if item.is_file()):
                        source_item = source / existing.relative_to(destination)
                        if not source_item.is_file() or _sha256_file(existing) != _sha256_file(source_item):
                            conflicts.append(existing.relative_to(destination).as_posix())
                    if conflicts:
                        errors.append(
                            f"{logical_path}: partial destination has conflicting files: {conflicts[:5]}"
                        )
                        continue
                    if not apply:
                        would_copy.append(logical_path)
                        continue
                    for source_item in sorted(item for item in source.rglob("*") if item.is_file()):
                        target = destination / source_item.relative_to(source)
                        if target.exists():
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_item, target)
                    hydrated = artifact_record(destination, logical_path=logical_path)
                    if hydrated.get("sha256") != record.get("sha256"):
                        errors.append(f"{logical_path}: hydrated directory does not match golden manifest")
                    else:
                        copied.append(logical_path)
                    continue
                errors.append(f"{logical_path}: destination exists with a non-golden hash")
                continue
            source = source_root / relative
            if not source.exists():
                errors.append(f"{logical_path}: missing from source root")
                continue
            source_record = artifact_record(source, logical_path=logical_path)
            if source_record.get("sha256") != record.get("sha256"):
                errors.append(f"{logical_path}: source hash does not match golden manifest")
                continue
            if not apply:
                would_copy.append(logical_path)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, copy_function=shutil.copy2)
            else:
                shutil.copy2(source, destination)
            copied.append(logical_path)
    return {
        "copied": copied,
        "would_copy": would_copy,
        "verified": verified,
        "errors": errors,
    }
