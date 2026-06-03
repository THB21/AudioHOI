#!/usr/bin/env python3
"""Run HaMeR hand mesh recovery per frame, stitched to GVHMR wrist anchors.

Uses GVHMR wrist joint projections as crop hints — no separate hand detector
required. Each crop is fed to HaMeR to get MANO finger articulation. The
resulting hand mesh is re-rooted at the GVHMR wrist to align with the body.

Works on both real and video-diffusion-generated footage because:
  - Crop hints come from GVHMR (robust to synthetic appearance)
  - Only relative finger configuration is taken from HaMeR (not absolute scale)
  - Optional temporal smoothing compensates for per-frame jitter

Outputs (in <sample-dir>/results/hands/):
  hand_keypoints_3d.csv   fingertip + palm positions in GVHMR camera frame
  hand_mano_params.pkl    MANO params for downstream stitch_hands_to_body.py
  hand_detection_summary.txt

Install HaMeR before running:
  pip install git+https://github.com/geopavlakos/hamer.git
  # Download checkpoints per HaMeR README (fetch_demo_data.sh)
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

try:
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False

try:
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    HAMER_AVAILABLE = True
except ImportError:
    HAMER_AVAILABLE = False

# ---------------------------------------------------------------------------
# MANO 21-joint convention (thumb-first ordering used by HaMeR):
#   0: Wrist
#   1-4:  Thumb  (CMC, MCP, IP, TIP)
#   5-8:  Index  (MCP, PIP, DIP, TIP)
#   9-12: Middle (MCP, PIP, DIP, TIP)
#  13-16: Ring   (MCP, PIP, DIP, TIP)
#  17-20: Pinky  (MCP, PIP, DIP, TIP)
# ---------------------------------------------------------------------------
MANO_WRIST = 0
MANO_TIPS: dict[str, int] = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
# Palm centre: wrist + index/middle/ring/pinky MCPs
MANO_PALM_JOINTS = [0, 5, 9, 13, 17]

# GVHMR SMPL-X wrist joint indices
GVHMR_LEFT_WRIST = 20
GVHMR_RIGHT_WRIST = 21

HAND_NAMES = ("left", "right")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_gvhmr(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    params = data["smpl_params_incam"]
    return {
        "body_pose": np.asarray(params["body_pose"], dtype=np.float32),
        "betas": np.asarray(params["betas"], dtype=np.float32),
        "global_orient": np.asarray(params["global_orient"], dtype=np.float32),
        "transl": np.asarray(params["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(data["K_fullimg"], dtype=np.float32),
    }


def build_gvhmr_wrists(gvhmr: dict[str, np.ndarray], body_models_root: Path) -> np.ndarray:
    """Return wrist positions [N, 2, 3]: axis-1 is (left=0, right=1), in camera frame."""
    if not SMPLX_AVAILABLE:
        raise RuntimeError("smplx not installed — run: pip install smplx")
    n_frames = gvhmr["transl"].shape[0]
    model = smplx.create(
        str(body_models_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=n_frames,
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(gvhmr["body_pose"]),
            betas=torch.from_numpy(gvhmr["betas"]),
            global_orient=torch.from_numpy(gvhmr["global_orient"]),
            transl=torch.from_numpy(gvhmr["transl"]),
            return_verts=False,
        )
    joints = output.joints.detach().cpu().numpy().astype(np.float64)
    return np.stack([joints[:, GVHMR_LEFT_WRIST, :], joints[:, GVHMR_RIGHT_WRIST, :]], axis=1)


def list_frames(frames_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (frame_index, path) pairs for PNG frames."""
    result = []
    for p in sorted(frames_dir.glob("*.png")):
        try:
            result.append((int(p.stem), p))
        except ValueError:
            pass
    return result


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------

def project_to_2d(xyz: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    z = max(float(xyz[2]), 1e-6)
    u = float(K[0, 0]) * float(xyz[0]) / z + float(K[0, 2])
    v = float(K[1, 1]) * float(xyz[1]) / z + float(K[1, 2])
    return u, v


def make_bbox(cx: float, cy: float, size: float, img_w: int, img_h: int) -> Optional[np.ndarray]:
    """Square bbox [x1,y1,x2,y2,score] clamped to image bounds. None if degenerate."""
    half = size / 2.0
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(float(img_w), cx + half)
    y2 = min(float(img_h), cy + half)
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return None
    return np.array([x1, y1, x2, y2, 1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-hand result container
# ---------------------------------------------------------------------------

@dataclass
class HandResult:
    detected: bool = False
    wrist_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    palm_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tips_xyz: dict[str, np.ndarray] = field(default_factory=dict)
    mano_global_orient: np.ndarray = field(default_factory=lambda: np.eye(3))
    mano_hand_pose: np.ndarray = field(default_factory=lambda: np.zeros((15, 3, 3)))
    mano_betas: np.ndarray = field(default_factory=lambda: np.zeros(10))


# ---------------------------------------------------------------------------
# HaMeR inference for a single hand crop
# ---------------------------------------------------------------------------

def run_hamer_single(
    img_bgr: np.ndarray,
    bbox: np.ndarray,
    is_right: bool,
    model,
    model_cfg,
    device: torch.device,
    rescale_factor: float,
) -> Optional[dict]:
    """Run HaMeR on one hand crop. Returns raw outputs dict or None on failure."""
    bboxes = bbox[np.newaxis, :]  # [1, 5]
    try:
        dataset = ViTDetDataset(
            model_cfg,
            img_bgr,
            bboxes,
            right=np.array([float(is_right)]),
            rescale_factor=rescale_factor,
        )
        if len(dataset) == 0:
            return None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        item = next(iter(dataloader))
        item = recursive_to(item, device)
        with torch.no_grad():
            out = model(item)
        return {
            "vertices": out["pred_vertices"][0].cpu().numpy().astype(np.float64),
            "keypoints": out["pred_keypoints_3d"][0].cpu().numpy().astype(np.float64),
            "global_orient": out["pred_mano_params"]["global_orient"][0, 0].cpu().numpy().astype(np.float64),
            "hand_pose": out["pred_mano_params"]["hand_pose"][0].cpu().numpy().astype(np.float64),
            "betas": out["pred_mano_params"]["betas"][0].cpu().numpy().astype(np.float64),
        }
    except Exception:
        return None


def stitch_to_gvhmr(raw: dict, gvhmr_wrist: np.ndarray) -> HandResult:
    """Center HaMeR hand on its wrist root, then shift to GVHMR wrist position."""
    joints = raw["keypoints"]  # [21, 3] in HaMeR canonical space
    wrist_local = joints[MANO_WRIST].copy()

    joints_cam = joints - wrist_local + gvhmr_wrist

    tips: dict[str, np.ndarray] = {
        name: joints_cam[idx].copy() for name, idx in MANO_TIPS.items()
    }
    palm = joints_cam[MANO_PALM_JOINTS].mean(axis=0)

    return HandResult(
        detected=True,
        wrist_xyz=gvhmr_wrist.copy(),
        palm_xyz=palm,
        tips_xyz=tips,
        mano_global_orient=raw["global_orient"].copy(),
        mano_hand_pose=raw["hand_pose"].copy(),
        mano_betas=raw["betas"].copy(),
    )


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def smooth_positions(
    positions: np.ndarray,
    valid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Apply per-axis Gaussian smoothing only over detected frames."""
    if sigma <= 0 or valid.sum() < 3:
        return positions
    out = positions.copy()
    for axis in range(positions.shape[1]):
        series = positions[:, axis].copy()
        series[~valid] = np.interp(
            np.where(~valid)[0],
            np.where(valid)[0],
            series[valid],
        ) if valid.sum() >= 2 else 0.0
        smoothed = gaussian_filter1d(series, sigma=sigma)
        out[valid, axis] = smoothed[valid]
    return out


# ---------------------------------------------------------------------------
# CSV / PKL writers
# ---------------------------------------------------------------------------

TIP_NAMES = list(MANO_TIPS.keys())
HAND_CSV_FIELDS = ["frame", "time"]
for _side in HAND_NAMES:
    HAND_CSV_FIELDS += [f"{_side}_detected", f"{_side}_wrist_x", f"{_side}_wrist_y", f"{_side}_wrist_z"]
    HAND_CSV_FIELDS += [f"{_side}_palm_x", f"{_side}_palm_y", f"{_side}_palm_z"]
    for _tip in TIP_NAMES:
        HAND_CSV_FIELDS += [f"{_side}_{_tip}_tip_x", f"{_side}_{_tip}_tip_y", f"{_side}_{_tip}_tip_z"]


def build_csv_row(
    frame: int,
    time: float,
    left: HandResult,
    right: HandResult,
) -> dict[str, object]:
    row: dict[str, object] = {"frame": frame, "time": f"{time:.6f}"}
    for side, res in (("left", left), ("right", right)):
        row[f"{side}_detected"] = int(res.detected)
        for coord, val in zip("xyz", res.wrist_xyz):
            row[f"{side}_wrist_{coord}"] = f"{val:.6f}"
        for coord, val in zip("xyz", res.palm_xyz):
            row[f"{side}_palm_{coord}"] = f"{val:.6f}"
        for tip_name in TIP_NAMES:
            tip = res.tips_xyz.get(tip_name, np.zeros(3))
            for coord, val in zip("xyz", tip):
                row[f"{side}_{tip_name}_tip_{coord}"] = f"{val:.6f}"
    return row


def write_keypoints_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HAND_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_mano_pkl(path: Path, data: dict) -> None:
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HaMeR hand reconstruction stitched to GVHMR wrist anchors."
    )
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument(
        "--body-model-root",
        type=Path,
        default=Path("third-party/GVHMR/inputs/checkpoints/body_models"),
        help="Root directory containing SMPL-X body model files (for GVHMR wrist extraction).",
    )
    parser.add_argument(
        "--hamer-checkpoint",
        type=Path,
        default=None,
        help="Path to HaMeR checkpoint (.ckpt). Defaults to HaMeR's DEFAULT_CHECKPOINT.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=0.28,
        help="Hand crop size as fraction of min(img_w, img_h). Default 0.28.",
    )
    parser.add_argument(
        "--rescale-factor",
        type=float,
        default=2.0,
        help="HaMeR internal bbox rescale factor (larger = more context). Default 2.0.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma for temporal smoothing of fingertip positions (frames). 0 = off.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not HAMER_AVAILABLE:
        raise RuntimeError(
            "HaMeR is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/geopavlakos/hamer.git\n"
            "Then download checkpoints per the HaMeR README."
        )
    if not SMPLX_AVAILABLE:
        raise RuntimeError("smplx is not installed: pip install smplx")

    device = torch.device(args.device)
    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = results_dir / "hands"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GVHMR result and extract wrist positions
    gvhmr_path = results_dir / "gvhmr" / "result.pkl"
    print(f"Loading GVHMR result: {gvhmr_path}")
    gvhmr = read_gvhmr(gvhmr_path)
    K = gvhmr["K_fullimg"]  # [N, 3, 3] or [1, 3, 3]

    print("Building GVHMR wrist positions via SMPL-X forward pass...")
    gvhmr_wrists = build_gvhmr_wrists(gvhmr, args.body_model_root)
    n_gvhmr = gvhmr_wrists.shape[0]
    print(f"  {n_gvhmr} frames from GVHMR")

    # Resolve camera intrinsics per frame
    if K.ndim == 2:
        K = K[np.newaxis].repeat(n_gvhmr, axis=0)
    elif K.shape[0] == 1:
        K = np.repeat(K, n_gvhmr, axis=0)

    # Load HaMeR model
    # HaMeR uses paths relative to its own source directory (_DATA/...), so we
    # temporarily chdir there before calling load_hamer, then restore cwd.
    import os
    hamer_src = Path(__file__).resolve().parent.parent.parent / "third-party" / "hamer"
    if args.hamer_checkpoint:
        checkpoint = str(Path(args.hamer_checkpoint).resolve())
    else:
        checkpoint = str(hamer_src / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt")
    print(f"Loading HaMeR from: {checkpoint}")
    _orig_cwd = os.getcwd()
    os.chdir(hamer_src)
    try:
        model, model_cfg = load_hamer(checkpoint)
    finally:
        os.chdir(_orig_cwd)
    model = model.to(device)
    model.eval()

    # Enumerate frames
    frames_dir = sample_dir / "frames"
    frame_list = list_frames(frames_dir)
    if not frame_list:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")
    print(f"Processing {len(frame_list)} frames on {args.device}...")

    # Storage for all frames (indexed by frame number)
    all_left: list[HandResult] = []
    all_right: list[HandResult] = []
    all_frames: list[int] = []

    n_detected_left = 0
    n_detected_right = 0

    for frame_idx, (frame_num, frame_path) in enumerate(frame_list):
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            all_frames.append(frame_num)
            all_left.append(HandResult())
            all_right.append(HandResult())
            continue

        img_h, img_w = img_bgr.shape[:2]
        crop_size = args.crop_scale * min(img_w, img_h)

        # Map frame number to GVHMR index (clamp to available range)
        gvhmr_idx = min(frame_num - 1, n_gvhmr - 1)
        gvhmr_idx = max(0, gvhmr_idx)
        K_frame = K[gvhmr_idx]
        wrists = gvhmr_wrists[gvhmr_idx]  # [2, 3]

        hand_results: list[HandResult] = []
        for hand_i, (is_right, gvhmr_wrist) in enumerate(
            [(False, wrists[0]), (True, wrists[1])]
        ):
            u, v = project_to_2d(gvhmr_wrist, K_frame)
            bbox = make_bbox(u, v, crop_size, img_w, img_h)
            if bbox is None:
                hand_results.append(HandResult(wrist_xyz=gvhmr_wrist.copy()))
                continue

            raw = run_hamer_single(img_bgr, bbox, is_right, model, model_cfg, device, args.rescale_factor)
            if raw is None:
                hand_results.append(HandResult(wrist_xyz=gvhmr_wrist.copy()))
                continue

            hand_results.append(stitch_to_gvhmr(raw, gvhmr_wrist))

        all_frames.append(frame_num)
        all_left.append(hand_results[0])
        all_right.append(hand_results[1])
        if hand_results[0].detected:
            n_detected_left += 1
        if hand_results[1].detected:
            n_detected_right += 1

        if (frame_idx + 1) % 25 == 0 or (frame_idx + 1) == len(frame_list):
            print(f"  frame {frame_idx + 1}/{len(frame_list)}  "
                  f"L_det={n_detected_left}  R_det={n_detected_right}")

    n_frames = len(all_frames)

    # Temporal smoothing on fingertip and palm positions
    if args.smooth_sigma > 0:
        print(f"Applying temporal smoothing (sigma={args.smooth_sigma} frames)...")
        for side_results in (all_left, all_right):
            valid = np.array([r.detected for r in side_results], dtype=bool)
            if valid.sum() < 3:
                continue

            # Smooth palm
            palms = np.array([r.palm_xyz for r in side_results])
            palms_s = smooth_positions(palms, valid, args.smooth_sigma)
            for i, r in enumerate(side_results):
                if valid[i]:
                    r.palm_xyz = palms_s[i]

            # Smooth each fingertip
            for tip_name in TIP_NAMES:
                tips = np.array([r.tips_xyz.get(tip_name, np.zeros(3)) for r in side_results])
                tips_s = smooth_positions(tips, valid, args.smooth_sigma)
                for i, r in enumerate(side_results):
                    if valid[i] and tip_name in r.tips_xyz:
                        r.tips_xyz[tip_name] = tips_s[i]

    # Write keypoints CSV
    csv_path = out_dir / "hand_keypoints_3d.csv"
    csv_rows = []
    for i, frame_num in enumerate(all_frames):
        t = frame_num / args.fps
        csv_rows.append(build_csv_row(frame_num, t, all_left[i], all_right[i]))
    write_keypoints_csv(csv_path, csv_rows)

    # Write MANO params PKL (for stitch_hands_to_body.py)
    def _collect_mano(results: list[HandResult]) -> dict[str, np.ndarray]:
        return {
            "detected": np.array([r.detected for r in results], dtype=bool),
            "global_orient": np.stack([r.mano_global_orient for r in results]),
            "hand_pose": np.stack([r.mano_hand_pose for r in results]),
            "betas": np.stack([r.mano_betas for r in results]),
            "wrist_xyz": np.stack([r.wrist_xyz for r in results]),
        }

    pkl_path = out_dir / "hand_mano_params.pkl"
    write_mano_pkl(pkl_path, {
        "frames": np.array(all_frames, dtype=np.int32),
        "fps": args.fps,
        "left": _collect_mano(all_left),
        "right": _collect_mano(all_right),
        "K_fullimg": K,
    })

    # Summary
    det_rate_l = n_detected_left / max(n_frames, 1)
    det_rate_r = n_detected_right / max(n_frames, 1)
    summary_lines = [
        f"n_frames: {n_frames}",
        f"n_detected_left: {n_detected_left}  ({det_rate_l:.1%})",
        f"n_detected_right: {n_detected_right}  ({det_rate_r:.1%})",
        f"crop_scale: {args.crop_scale}",
        f"rescale_factor: {args.rescale_factor}",
        f"smooth_sigma: {args.smooth_sigma}",
        f"device: {args.device}",
        f"hamer_checkpoint: {checkpoint}",
    ]
    summary_path = out_dir / "hand_detection_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"hand_keypoints_csv: {csv_path}")
    print(f"hand_mano_pkl: {pkl_path}")
    print(f"hand_summary: {summary_path}")
    print(f"detection_rate_left: {det_rate_l:.1%}")
    print(f"detection_rate_right: {det_rate_r:.1%}")


if __name__ == "__main__":
    main()
