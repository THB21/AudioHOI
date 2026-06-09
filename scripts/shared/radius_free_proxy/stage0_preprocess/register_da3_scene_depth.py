#!/usr/bin/env python3
"""Register precomputed Depth Anything 3 outputs into the sample results tree.

This script does not run DA3 inference itself. It organizes externally produced
depth maps into:

    results/da3/scene_depth/

and writes a small index/metadata manifest for downstream processing.

It supports two source layouts:

1. A directory of per-frame depth files (`.npy/.npz/.png/.tif`)
2. A DA3 `mini_npz` export containing `results.npz` with a stacked `depth` array
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


SUPPORTED_EXTS = (".npy", ".npz", ".png", ".tiff", ".tif")


def list_depth_files(source_dir: Path) -> list[Path]:
    files = [p for p in sorted(source_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        raise RuntimeError(
            f"No supported depth files found in {source_dir}. "
            f"Expected one of: {', '.join(SUPPORTED_EXTS)}"
        )
    return files


def infer_frame_id(path: Path, fallback_idx: int) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if digits:
        return int(digits)
    return fallback_idx


def resolve_npz_source(source_path: Path) -> Path | None:
    if source_path.is_file() and source_path.suffix.lower() == ".npz":
        return source_path
    candidate = source_path / "results.npz"
    if candidate.exists():
        return candidate
    return None


def unpack_da3_stack(npz_path: Path, out_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    data = np.load(npz_path)
    if "depth" not in data.files:
        raise RuntimeError(f"{npz_path} does not contain a 'depth' array")
    depth = np.asarray(data["depth"], dtype=np.float32)
    if depth.ndim != 3:
        raise RuntimeError(f"Expected stacked depth array with shape [T,H,W], got {depth.shape}")

    rows: list[dict[str, object]] = []
    for idx in range(depth.shape[0]):
        frame = idx + 1
        dst_name = f"{frame:05d}.npy"
        dst = out_dir / dst_name
        np.save(dst, depth[idx])
        rows.append(
            {
                "frame": frame,
                "file": dst.name,
                "source_file": f"{npz_path.resolve()}::depth[{idx}]",
                "storage": "unpacked",
            }
        )

    meta = {
        "source_npz": str(npz_path.resolve()),
        "num_frames": int(depth.shape[0]),
        "depth_shape": list(depth.shape),
        "keys": sorted(data.files),
    }
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Register precomputed DA3 depth outputs for a sample.")
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--source-depth-dir", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy files into results/da3/scene_depth instead of symlinking.")
    parser.add_argument("--description", type=str, default="Precomputed Depth Anything 3 outputs")
    args = parser.parse_args()

    sample_dir = args.sample_dir
    source_dir = args.source_depth_dir
    out_dir = sample_dir / "results" / "da3" / "scene_depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_csv = out_dir / "index.csv"
    meta_json = out_dir / "meta.json"

    npz_source = resolve_npz_source(source_dir)
    if npz_source is not None:
        rows, source_meta = unpack_da3_stack(npz_source, out_dir)
        storage_mode = "unpacked"
    else:
        if not source_dir.exists():
            raise RuntimeError(f"Missing source depth directory: {source_dir}")
        depth_files = list_depth_files(source_dir)
        rows = []
        for idx, src in enumerate(depth_files, start=1):
            frame = infer_frame_id(src, idx)
            dst_name = f"{frame:05d}{src.suffix.lower()}"
            dst = out_dir / dst_name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.copy:
                shutil.copy2(src, dst)
                storage = "copied"
            else:
                dst.symlink_to(src.resolve())
                storage = "symlinked"
            rows.append(
                {
                    "frame": frame,
                    "file": dst.name,
                    "source_file": str(src.resolve()),
                    "storage": storage,
                }
            )
        source_meta = {}
        storage_mode = "copy" if args.copy else "symlink"

    with index_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "file", "source_file", "storage"])
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "description": args.description,
        "source_depth_dir": str(source_dir.resolve()),
        "num_frames": len(rows),
        "storage_mode": storage_mode,
        "supported_exts": list(SUPPORTED_EXTS),
    }
    meta.update(source_meta)
    meta_json.write_text(json.dumps(meta, indent=2))

    print(f"da3_scene_depth_dir: {out_dir}")
    print(f"da3_index_csv: {index_csv}")
    print(f"da3_meta_json: {meta_json}")
    print(f"registered_frames: {len(rows)}")


if __name__ == "__main__":
    main()
