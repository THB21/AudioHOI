from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..base.io import repo_path, repo_relative_value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    """Immutable per-result content-addressed storage for provenance snapshots."""

    def __init__(self, result_dir: Path):
        self.root = result_dir / "provenance" / "artifact_store"
        self.blob_root = self.root / "sha256"

    def blob_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid SHA-256 digest {digest!r}")
        return self.blob_root / digest[:2] / digest

    def put(self, source: Path, *, canonical_path: str | None = None) -> dict[str, object]:
        if not source.is_file():
            raise FileNotFoundError(f"Artifact source is not a file: {source}")
        digest = sha256_file(source)
        size = source.stat().st_size
        destination = self.blob_path(digest)
        if destination.exists():
            if destination.stat().st_size != size or sha256_file(destination) != digest:
                raise RuntimeError(f"Artifact-store blob is corrupt: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{digest}.{os.getpid()}.tmp")
            if temporary.exists():
                raise FileExistsError(f"Refusing to reuse artifact-store temporary file {temporary}")
            try:
                shutil.copy2(source, temporary)
                if temporary.stat().st_size != size or sha256_file(temporary) != digest:
                    raise RuntimeError(f"Artifact-store copy verification failed for {source}")
                temporary.chmod(0o444)
                if destination.exists():
                    if sha256_file(destination) != digest:
                        raise RuntimeError(f"Concurrent artifact-store blob is corrupt: {destination}")
                    temporary.unlink()
                else:
                    os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return {
            "algorithm": "sha256",
            "sha256": digest,
            "size_bytes": size,
            "canonical_path": canonical_path or str(repo_relative_value(source)),
            "blob_path": str(repo_relative_value(destination)),
        }

    def verify_reference(self, reference: dict[str, object]) -> list[str]:
        errors: list[str] = []
        digest = str(reference.get("sha256", ""))
        try:
            blob = self.blob_path(digest)
        except ValueError as exc:
            return [str(exc)]
        if not blob.is_file():
            return [f"missing artifact-store blob for {reference.get('canonical_path', '')}: {blob}"]
        expected_size = reference.get("size_bytes")
        if blob.stat().st_size != expected_size:
            errors.append(
                f"{blob}: size expected {expected_size!r}, got {blob.stat().st_size!r}"
            )
        actual_digest = sha256_file(blob)
        if actual_digest != digest:
            errors.append(f"{blob}: sha256 expected {digest}, got {actual_digest}")
        return errors


def store_stage_artifacts(
    result_dir: Path,
    artifacts: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    store = ArtifactStore(result_dir)
    stored: dict[str, dict[str, object]] = {}
    for canonical_path in sorted(artifacts):
        source = repo_path(canonical_path)
        if not source.is_file():
            raise FileNotFoundError(
                f"Stage artifact disappeared before provenance storage: {canonical_path}"
            )
        expected = artifacts[canonical_path]
        reference = store.put(source, canonical_path=canonical_path)
        if reference["sha256"] != expected.get("sha256"):
            raise RuntimeError(
                f"Stage artifact changed during provenance storage: {canonical_path}"
            )
        stored[canonical_path] = reference
    return stored


def verify_attempt_artifacts(result_dir: Path) -> list[str]:
    errors: list[str] = []
    store = ArtifactStore(result_dir)
    attempts_root = result_dir / "provenance" / "stages"
    if not attempts_root.exists():
        return [f"missing stage provenance directory: {attempts_root}"]
    for attempt_path in sorted(attempts_root.glob("*/attempts/*.json")):
        try:
            payload: Any = json.loads(attempt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{attempt_path}: invalid attempt record: {exc}")
            continue
        stored = payload.get("stored_artifacts", {}) if isinstance(payload, dict) else {}
        artifacts_after = payload.get("artifacts_after", {}) if isinstance(payload, dict) else {}
        if not isinstance(stored, dict):
            errors.append(f"{attempt_path}: stored_artifacts must be an object")
            continue
        if not isinstance(artifacts_after, dict):
            errors.append(f"{attempt_path}: artifacts_after must be an object")
            continue
        missing_references = sorted(set(artifacts_after) - set(stored))
        extra_references = sorted(set(stored) - set(artifacts_after))
        if missing_references:
            errors.append(
                f"{attempt_path}: artifacts missing store references: {missing_references}"
            )
        if extra_references:
            errors.append(
                f"{attempt_path}: store references lack canonical artifacts: {extra_references}"
            )
        for canonical_path, reference in stored.items():
            if not isinstance(reference, dict):
                errors.append(f"{attempt_path}: invalid reference for {canonical_path}")
                continue
            expected = artifacts_after.get(canonical_path, {})
            if reference.get("canonical_path") != canonical_path:
                errors.append(f"{attempt_path}: canonical path mismatch for {canonical_path}")
            if isinstance(expected, dict):
                for key in ("sha256", "size_bytes"):
                    if reference.get(key) != expected.get(key):
                        errors.append(
                            f"{attempt_path}: {canonical_path} {key} does not match artifacts_after"
                        )
            for error in store.verify_reference(reference):
                errors.append(f"{attempt_path}: {error}")
    return errors
