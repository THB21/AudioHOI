#!/usr/bin/env python3
"""Run Depth-Anything-3 and normalize dense frame depth into the sample contract."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[4]
DEFAULT_DA3_ROOT = REPO / "third-party/Depth-Anything-3"
DEFAULT_MODEL = "depth-anything/DA3METRIC-LARGE"


def _frame_files(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))


def _install_directory_atomic(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.parent / f".{target.name}.previous"
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except Exception:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def generate_scene_depth(
    sample_dir: Path,
    *,
    da3_root: Path,
    model_dir: str,
    process_res: int,
    chunk_size: int,
) -> dict[str, object]:
    sample_dir = sample_dir.resolve()
    da3_root = da3_root.resolve()
    cli = da3_root / "src/depth_anything_3/cli.py"
    if not cli.is_file():
        raise FileNotFoundError(f"Depth-Anything-3 checkout is missing: expected {cli}")
    frames_dir = sample_dir / "frames"
    frames = _frame_files(frames_dir)
    if not frames:
        raise FileNotFoundError(f"no input frames found in {frames_dir}")

    results_dir = sample_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="audiohoi-da3-", dir=str(results_dir)))
    normalized = work / "scene_depth"
    if chunk_size <= 0:
        raise ValueError("DA3 chunk size must be positive")
    sys.path.insert(0, str(da3_root / "src"))
    try:
        import torch
        from depth_anything_3.api import DepthAnything3

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DepthAnything3.from_pretrained(model_dir).to(device).eval()
        chunks: list[np.ndarray] = []
        for start in range(0, len(frames), chunk_size):
            selected = frames[start : start + chunk_size]
            prediction = model.inference(
                image=[str(path) for path in selected],
                process_res=process_res,
                export_dir=None,
            )
            values = np.asarray(prediction.depth, dtype=np.float32)
            if values.ndim != 3 or values.shape[0] != len(selected):
                raise ValueError(
                    f"DA3 chunk depth shape {values.shape} does not match {len(selected)} frames"
                )
            chunks.append(values)
            del prediction
            if device == "cuda":
                torch.cuda.empty_cache()
        depth = np.concatenate(chunks, axis=0)
        if depth.ndim != 3 or depth.shape[0] != len(frames):
            raise ValueError(
                f"DA3 depth shape {depth.shape} does not match {len(frames)} input frames"
            )
        if not np.isfinite(depth).all():
            raise ValueError("DA3 depth contains non-finite values")
        normalized.mkdir(parents=True)
        digest = hashlib.sha256()
        digest.update(model_dir.encode())
        digest.update(str(process_res).encode())
        for values in depth:
            digest.update(np.ascontiguousarray(values).tobytes())
        source_hash = digest.hexdigest()
        index_rows: list[dict[str, object]] = []
        for frame, values in enumerate(depth, start=1):
            filename = f"{frame:05d}.npy"
            np.save(normalized / filename, values, allow_pickle=False)
            index_rows.append(
                {
                    "frame": frame,
                    "file": filename,
                    "source_file": f"da3_chunked_metric:{source_hash}:depth[{frame - 1}]",
                    "storage": "unpacked",
                }
            )
        with (normalized / "index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("frame", "file", "source_file", "storage"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(index_rows)
        target = sample_dir / "results/da3/scene_depth"
        _install_directory_atomic(normalized, target)
        summary = {
            "schema_version": 1,
            "status": "generated",
            "sample_dir": str(sample_dir),
            "output": str(target),
            "frame_count": len(frames),
            "depth_shape": list(depth.shape),
            "normalized_depth_sha256": source_hash,
            "da3_root": str(da3_root),
            "model_dir": model_dir,
            "process_res": process_res,
            "chunk_size": chunk_size,
            "device": device,
        }
        (target / "generation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return summary
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--da3-root", type=Path, default=DEFAULT_DA3_ROOT)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_scene_depth(
                args.sample_dir,
                da3_root=args.da3_root,
                model_dir=args.model_dir,
                process_res=args.process_res,
                chunk_size=args.chunk_size,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
