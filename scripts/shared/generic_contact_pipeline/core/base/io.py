from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable, Any


REPO = Path(__file__).resolve().parents[5]


def repo_relative_value(value: Any) -> Any:
    """Convert repo-local absolute paths in artifact payloads to repo-relative strings."""
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(REPO).as_posix()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {key: repo_relative_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [repo_relative_value(item) for item in value]
    if isinstance(value, tuple):
        return [repo_relative_value(item) for item in value]
    if isinstance(value, str):
        prefix = REPO.as_posix().rstrip("/") + "/"
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def ensure_dir(path: str | Path) -> Path:
    out = repo_path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = repo_path(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_csv_header(path: str | Path) -> list[str]:
    path = repo_path(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f).fieldnames or [])


def write_csv(path: str | Path, rows: list[dict[str, object]], fields: Iterable[str] | None = None) -> Path:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        seen: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fields = seen
    fields = list(fields)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: repo_relative_value(row.get(key, "")) for key in fields})
    return path


def write_json(path: str | Path, data: object) -> Path:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(repo_relative_value(data), indent=2, ensure_ascii=False) + "\n")
    return path


def copy_file(src: str | Path, dst: str | Path) -> Path:
    src = repo_path(src)
    dst = repo_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def copy_tree_files(src_dir: str | Path, dst_dir: str | Path, suffixes: tuple[str, ...] = (".mp4", ".png", ".json")) -> list[Path]:
    src_dir = repo_path(src_dir)
    dst_dir = repo_path(dst_dir)
    copied: list[Path] = []
    for src in src_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in suffixes:
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def count_rows(path: str | Path) -> int:
    return len(read_csv(path))


def float_or_none(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None
