from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Any

from ...base.io import REPO, repo_relative_value


def repo_rel(path: str | Path) -> str:
    """Return a stable repo-relative artifact path when possible."""
    original = str(path)
    candidate = Path(original)
    try:
        resolved = candidate.resolve() if candidate.is_absolute() else (REPO / candidate).resolve()
        return resolved.relative_to(REPO).as_posix()
    except Exception:
        repo_prefix = REPO.as_posix().rstrip("/") + "/"
        if original.startswith(repo_prefix):
            return original[len(repo_prefix) :]
        return original


def normalize_artifact_value(value: Any) -> Any:
    """Normalize artifact payloads so shareable outputs avoid local absolute paths."""
    return repo_relative_value(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    fields = list(fields)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_artifact_value(row.get(key, "")) for key in fields})
    return path


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_artifact_value(data), indent=2, ensure_ascii=False) + "\n")
    return path


def f(value: object, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        out = float(value)  # type: ignore[arg-type]
        return out if math.isfinite(out) else default
    except Exception:
        return default


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def max_or_none(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return max(vals) if vals else None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
