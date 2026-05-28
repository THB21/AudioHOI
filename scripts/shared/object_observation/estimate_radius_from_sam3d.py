#!/usr/bin/env python3
"""Estimate sphere radius from SAM 3D Objects reconstruction.

Uses SAM 3D Objects to reconstruct 3D object geometry from single-image mask,
then fits a sphere to the point cloud to estimate the object radius.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Add SAM 3D Objects to path
SAM3D_ROOT = Path(__file__).parent.parent.parent.parent / "third-party" / "sam-3d-objects"
if SAM3D_ROOT.exists():
    sys.path.insert(0, str(SAM3D_ROOT / "notebook"))
    sys.path.insert(0, str(SAM3D_ROOT))


def load_image(image_path: str) -> np.ndarray:
    """Load and prepare image for SAM 3D Objects."""
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_single_mask(mask_dir: Path, frame: int) -> np.ndarray:
    """Load mask for a single frame."""
    mask_path = mask_dir / f"{frame:05d}_mask.png"
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8) * 255


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a sphere to 3D point cloud.
    
    Args:
        points: (N, 3) array of 3D points
        
    Returns:
        center: (3,) sphere center
        radius: scalar sphere radius
    """
    if points.shape[0] < 4:
        return np.zeros(3), 0.0
    
    # Simple least-squares sphere fitting
    # Minimize: sum((||p - c||^2 - r^2)^2)
    # Use initial center as mean of points
    center = points.mean(axis=0)
    
    # Iterative refinement (Gauss-Newton style)
    for _ in range(10):
        dists = np.linalg.norm(points - center, axis=1)
        radius = dists.mean()
        
        # Jacobian: d/dc of ||p - c||^2 = -2(p - c)
        residuals = dists - radius
        if np.abs(residuals).max() < 1e-6:
            break
            
        # Gradient for center
        grad = np.zeros(3)
        for i, p in enumerate(points):
            if dists[i] > 1e-6:
                grad += 2 * (center - p) * residuals[i] / dists[i]
        
        # Simple gradient step
        step_size = 0.01
        center = center - step_size * grad / max(len(points), 1)
    
    radius = np.linalg.norm(points - center, axis=1).mean()
    return center, radius


def estimate_radius_from_sam3d(
    image_path: Path,
    mask_path: Path,
    sam3d_inference,
    config_path: Path,
) -> dict[str, float]:
    """Estimate sphere radius from single image + mask using SAM 3D Objects.
    
    Args:
        image_path: Path to input image
        mask_path: Path to object mask
        sam3d_inference: Inference object (lazy loaded if available)
        config_path: Path to SAM 3D config
        
    Returns:
        Dictionary with estimated_radius_m, center_xyz, etc.
    """
    try:
        from inference import Inference
    except ImportError:
        print("Warning: SAM 3D Objects not properly installed, skipping radius estimation")
        return {"estimated_radius_m": None, "center_x": None, "center_y": None, "center_z": None}
    
    try:
        # Load image and mask
        image = load_image(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return {"estimated_radius_m": None, "center_x": None, "center_y": None, "center_z": None}
        
        mask = (mask > 0).astype(np.uint8) * 255
        
        # Initialize SAM 3D if needed
        if sam3d_inference is None:
            sam3d_inference = Inference(str(config_path), compile=False)
        
        # Run SAM 3D Objects inference
        output = sam3d_inference(image, mask, seed=42)
        
        # Extract 3D geometry
        # output contains "mesh" or "gs" (gaussian splatting) or point cloud
        if "mesh" in output:
            # If mesh, sample points from vertices
            vertices = output["mesh"].vertices
            points_3d = np.asarray(vertices, dtype=np.float32)
        elif "gs" in output:
            # Gaussian splatting: extract means as point cloud
            gs = output["gs"]
            if hasattr(gs, "mean_xyz"):
                points_3d = np.asarray(gs.mean_xyz, dtype=np.float32)
            else:
                points_3d = None
        else:
            points_3d = None
        
        if points_3d is None or len(points_3d) < 4:
            return {"estimated_radius_m": None, "center_x": None, "center_y": None, "center_z": None}
        
        # Fit sphere to 3D points
        center, radius = fit_sphere(points_3d)
        
        return {
            "estimated_radius_m": float(radius),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
        }
    
    except Exception as e:
        print(f"Error in SAM 3D estimation: {e}")
        return {"estimated_radius_m": None, "center_x": None, "center_y": None, "center_z": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate object radius using SAM 3D Objects reconstruction."
    )
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--object-observations-csv", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames to process (for testing)")
    parser.add_argument(
        "--sam3d-config",
        type=Path,
        default=SAM3D_ROOT / "checkpoints" / "hf" / "pipeline.yaml" if SAM3D_ROOT.exists() else None,
        help="Path to SAM 3D Objects config"
    )
    
    args = parser.parse_args()
    
    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    mask_dir = args.mask_dir or (results_dir / "segmentation" / "masks")
    video_dir = args.video_dir or (sample_dir / "video")
    out_csv = args.out_csv or (results_dir / "object_observations" / "radius_estimates.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Load object observations to get frame list
    obs_csv = args.object_observations_csv or (results_dir / "object_observations" / "object_observations.csv")
    frames = []
    with open(obs_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
    
    if args.max_frames:
        frames = frames[:args.max_frames]
    
    # Check if SAM 3D is available
    sam3d_inference = None
    config_path = args.sam3d_config
    if not config_path or not config_path.exists():
        print("Warning: SAM 3D Objects config not found, skipping radius estimation")
        print(f"Expected at: {args.sam3d_config}")
        # Still write empty CSV for compatibility
        rows = [
            {
                "frame": frame,
                "estimated_radius_m": "",
                "center_x": "",
                "center_y": "",
                "center_z": "",
            }
            for frame in frames
        ]
    else:
        rows = []
        for frame_idx, frame in enumerate(frames):
            print(f"Processing frame {frame_idx + 1}/{len(frames)}: {frame}")
            
            # Find video frame
            video_frames = sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.png"))
            frame_paths = [p for p in video_frames if f"{frame:05d}" in p.stem or f"_{frame:06d}" in p.stem]
            
            if not frame_paths:
                print(f"  Warning: No video frame found for frame {frame}")
                rows.append({
                    "frame": frame,
                    "estimated_radius_m": "",
                    "center_x": "",
                    "center_y": "",
                    "center_z": "",
                })
                continue
            
            image_path = frame_paths[0]
            mask_path = mask_dir / f"{frame:05d}_mask.png"
            
            if not mask_path.exists():
                print(f"  Warning: No mask found for frame {frame}")
                rows.append({
                    "frame": frame,
                    "estimated_radius_m": "",
                    "center_x": "",
                    "center_y": "",
                    "center_z": "",
                })
                continue
            
            # Estimate radius
            result = estimate_radius_from_sam3d(image_path, mask_path, sam3d_inference, config_path)
            rows.append({
                "frame": frame,
                "estimated_radius_m": f"{result['estimated_radius_m']:.6f}" if result["estimated_radius_m"] is not None else "",
                "center_x": f"{result['center_x']:.6f}" if result["center_x"] is not None else "",
                "center_y": f"{result['center_y']:.6f}" if result["center_y"] is not None else "",
                "center_z": f"{result['center_z']:.6f}" if result["center_z"] is not None else "",
            })
    
    # Write output
    with out_csv.open("w", newline="") as f:
        fieldnames = ["frame", "estimated_radius_m", "center_x", "center_y", "center_z"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote radius estimates to {out_csv}")


if __name__ == "__main__":
    main()
