#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def sample_contour_uniform(contour: np.ndarray, n: int) -> np.ndarray:
    contour = contour.reshape(-1, 2).astype(np.float32)
    if len(contour) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(contour) <= n:
        return contour
    idx = np.linspace(0, len(contour) - 1, n, dtype=np.int32)
    return contour[idx]


def sample_interior_grid(binary: np.ndarray, n: int) -> np.ndarray:
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(coords) <= n:
        return coords
    idx = np.linspace(0, len(coords) - 1, n, dtype=np.int32)
    return coords[idx]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Experimental generic object mesh tracks from SAM2 masks.')
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=24.0)
    parser.add_argument('--n-boundary', type=int, default=64)
    parser.add_argument('--n-interior', type=int, default=64)
    parser.add_argument('--out-name', type=str, default='object_mesh_tracks_test.csv')
    args = parser.parse_args()

    sample_dir = args.sample_dir
    masks_dir = sample_dir / 'results' / 'segmentation' / 'masks'
    out_csv = sample_dir / 'results' / 'tracking' / args.out_name
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    mask_paths = sorted(masks_dir.glob('*_mask.png'))
    if not mask_paths:
        raise RuntimeError(f'No masks found in {masks_dir}')

    for frame_idx, mask_path in enumerate(mask_paths, start=1):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f'Could not read mask {mask_path}')
        binary = mask > 0
        if not np.any(binary):
            continue
        contours, _ = cv2.findContours((binary.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        boundary = sample_contour_uniform(contour, args.n_boundary)
        interior = sample_interior_grid(binary, args.n_interior)
        t = (frame_idx - 1) / args.fps
        for i, pt in enumerate(boundary):
            rows.append({'frame': frame_idx, 'time': f'{t:.6f}', 'point_id': f'boundary_{i:03d}', 'point_name': f'boundary_{i:03d}', 'point_type': 'boundary', 'x': f'{float(pt[0]):.3f}', 'y': f'{float(pt[1]):.3f}', 'visible': '1.000000', 'source_chunk': 0})
        for i, pt in enumerate(interior):
            rows.append({'frame': frame_idx, 'time': f'{t:.6f}', 'point_id': f'interior_{i:03d}', 'point_name': f'interior_{i:03d}', 'point_type': 'interior', 'x': f'{float(pt[0]):.3f}', 'y': f'{float(pt[1]):.3f}', 'visible': '1.000000', 'source_chunk': 0})

    write_csv(out_csv, rows, ['frame', 'time', 'point_id', 'point_name', 'point_type', 'x', 'y', 'visible', 'source_chunk'])
    print(f'object_mesh_tracks_csv: {out_csv}')


if __name__ == '__main__':
    main()
