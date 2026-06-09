#!/usr/bin/env python3
"""Run HaMeR per frame, anchored to GVHMR's wrist projections.

Crops are placed around the GVHMR wrist (no separate hand detector). HaMeR
predicts MANO finger pose and a wrist orientation; we keep both and re-root
the keypoints at the GVHMR wrist so positions stay consistent with the body.

Outputs (in <sample-dir>/results/hands/):
  hand_keypoints_3d.csv
  hand_mano_params.pkl
  hand_detection_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

import smplx
from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset


# MANO 21-joint convention (thumb-first, HaMeR standard):
#   0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
MANO_WRIST = 0
MANO_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
MANO_PALM_JOINTS = [0, 5, 9, 13, 17]

GVHMR_LEFT_WRIST = 20
GVHMR_RIGHT_WRIST = 21
HAND_NAMES = ("left", "right")


def read_gvhmr(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    p = data["smpl_params_incam"]
    return {
        "body_pose": np.asarray(p["body_pose"], dtype=np.float32),
        "betas": np.asarray(p["betas"], dtype=np.float32),
        "global_orient": np.asarray(p["global_orient"], dtype=np.float32),
        "transl": np.asarray(p["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(data["K_fullimg"], dtype=np.float32),
    }


def build_gvhmr_wrists(gvhmr, body_models_root):
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
        out = model(
            body_pose=torch.from_numpy(gvhmr["body_pose"]),
            betas=torch.from_numpy(gvhmr["betas"]),
            global_orient=torch.from_numpy(gvhmr["global_orient"]),
            transl=torch.from_numpy(gvhmr["transl"]),
            return_verts=False,
        )
    joints = out.joints.detach().cpu().numpy().astype(np.float64)
    return np.stack(
        [joints[:, GVHMR_LEFT_WRIST, :], joints[:, GVHMR_RIGHT_WRIST, :]], axis=1
    )


def list_frames(frames_dir):
    out = []
    for p in sorted(frames_dir.glob("*.png")):
        try:
            out.append((int(p.stem), p))
        except ValueError:
            pass
    return out


def project_to_2d(xyz, K):
    z = max(float(xyz[2]), 1e-6)
    u = float(K[0, 0]) * float(xyz[0]) / z + float(K[0, 2])
    v = float(K[1, 1]) * float(xyz[1]) / z + float(K[1, 2])
    return u, v


def make_bbox(cx, cy, size, img_w, img_h):
    half = size / 2.0
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(float(img_w), cx + half)
    y2 = min(float(img_h), cy + half)
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return None
    return np.array([x1, y1, x2, y2, 1.0], dtype=np.float32)


@dataclass
class HandResult:
    detected: bool = False
    wrist_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    palm_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tips_xyz: dict = field(default_factory=dict)
    mano_global_orient: np.ndarray = field(default_factory=lambda: np.eye(3))
    mano_hand_pose: np.ndarray = field(default_factory=lambda: np.zeros((15, 3, 3)))
    mano_betas: np.ndarray = field(default_factory=lambda: np.zeros(10))


def run_hamer_single(img_bgr, bbox, is_right, model, model_cfg, device, rescale_factor):
    try:
        dataset = ViTDetDataset(
            model_cfg, img_bgr, bbox[np.newaxis, :],
            right=np.array([float(is_right)]),
            rescale_factor=rescale_factor,
        )
        if len(dataset) == 0:
            return None
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        item = next(iter(loader))
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


def stitch_to_gvhmr(raw, gvhmr_wrist):
    joints = raw["keypoints"]
    wrist_local = joints[MANO_WRIST].copy()
    joints_cam = joints - wrist_local + gvhmr_wrist

    tips = {name: joints_cam[idx].copy() for name, idx in MANO_TIPS.items()}
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


def smooth_positions(positions, valid, sigma):
    if sigma <= 0 or valid.sum() < 3:
        return positions
    out = positions.copy()
    for axis in range(positions.shape[1]):
        series = positions[:, axis].copy()
        if valid.sum() >= 2:
            series[~valid] = np.interp(
                np.where(~valid)[0], np.where(valid)[0], series[valid]
            )
        smoothed = gaussian_filter1d(series, sigma=sigma)
        out[valid, axis] = smoothed[valid]
    return out


TIP_NAMES = list(MANO_TIPS.keys())
HAND_CSV_FIELDS = ["frame", "time"]
for _side in HAND_NAMES:
    HAND_CSV_FIELDS += [f"{_side}_detected", f"{_side}_wrist_x", f"{_side}_wrist_y", f"{_side}_wrist_z"]
    HAND_CSV_FIELDS += [f"{_side}_palm_x", f"{_side}_palm_y", f"{_side}_palm_z"]
    for _tip in TIP_NAMES:
        HAND_CSV_FIELDS += [f"{_side}_{_tip}_tip_x", f"{_side}_{_tip}_tip_y", f"{_side}_{_tip}_tip_z"]


def build_csv_row(frame, time, left, right):
    row = {"frame": frame, "time": f"{time:.6f}"}
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


def write_keypoints_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HAND_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_mano_pkl(path, data):
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument(
        "--body-model-root",
        type=Path,
        default=Path("third-party/GVHMR/inputs/checkpoints/body_models"),
    )
    parser.add_argument("--hamer-checkpoint", type=Path, default=None)
    parser.add_argument("--crop-scale", type=float, default=0.28)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--smooth-sigma", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    results_dir = args.sample_dir / "results"
    out_dir = results_dir / "hands"
    out_dir.mkdir(parents=True, exist_ok=True)

    gvhmr_path = results_dir / "gvhmr" / "result.pkl"
    print(f"Loading GVHMR result: {gvhmr_path}")
    gvhmr = read_gvhmr(gvhmr_path)
    K = gvhmr["K_fullimg"]

    print("Building GVHMR wrist positions via SMPL-X forward pass...")
    gvhmr_wrists = build_gvhmr_wrists(gvhmr, args.body_model_root)
    n_gvhmr = gvhmr_wrists.shape[0]
    print(f"  {n_gvhmr} frames from GVHMR")

    if K.ndim == 2:
        K = K[np.newaxis].repeat(n_gvhmr, axis=0)
    elif K.shape[0] == 1:
        K = np.repeat(K, n_gvhmr, axis=0)

    # HaMeR loads checkpoints via relative _DATA/... paths, so chdir into its
    # source dir before load_hamer and restore cwd afterwards.
    hamer_src = Path(__file__).resolve().parent.parent.parent / "third-party" / "hamer"
    checkpoint = str(
        Path(args.hamer_checkpoint).resolve()
        if args.hamer_checkpoint
        else hamer_src / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    )
    print(f"Loading HaMeR from: {checkpoint}")
    cwd = os.getcwd()
    os.chdir(hamer_src)
    try:
        model, model_cfg = load_hamer(checkpoint)
    finally:
        os.chdir(cwd)
    model = model.to(device).eval()

    frames_dir = args.sample_dir / "frames"
    frame_list = list_frames(frames_dir)
    if not frame_list:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")
    print(f"Processing {len(frame_list)} frames on {args.device}...")

    all_left = []
    all_right = []
    all_frames = []
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

        gvhmr_idx = min(max(frame_num - 1, 0), n_gvhmr - 1)
        K_frame = K[gvhmr_idx]
        wrists = gvhmr_wrists[gvhmr_idx]

        hand_results = []
        for is_right, gvhmr_wrist in [(False, wrists[0]), (True, wrists[1])]:
            u, v = project_to_2d(gvhmr_wrist, K_frame)
            bbox = make_bbox(u, v, crop_size, img_w, img_h)
            if bbox is None:
                hand_results.append(HandResult(wrist_xyz=gvhmr_wrist.copy()))
                continue
            raw = run_hamer_single(
                img_bgr, bbox, is_right, model, model_cfg, device, args.rescale_factor
            )
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

    if args.smooth_sigma > 0:
        print(f"Applying temporal smoothing (sigma={args.smooth_sigma} frames)...")
        for side_results in (all_left, all_right):
            valid = np.array([r.detected for r in side_results], dtype=bool)
            if valid.sum() < 3:
                continue

            palms = np.array([r.palm_xyz for r in side_results])
            palms_s = smooth_positions(palms, valid, args.smooth_sigma)
            for i, r in enumerate(side_results):
                if valid[i]:
                    r.palm_xyz = palms_s[i]

            for tip_name in TIP_NAMES:
                tips = np.array([r.tips_xyz.get(tip_name, np.zeros(3)) for r in side_results])
                tips_s = smooth_positions(tips, valid, args.smooth_sigma)
                for i, r in enumerate(side_results):
                    if valid[i] and tip_name in r.tips_xyz:
                        r.tips_xyz[tip_name] = tips_s[i]

    csv_path = out_dir / "hand_keypoints_3d.csv"
    csv_rows = [
        build_csv_row(fn, fn / args.fps, all_left[i], all_right[i])
        for i, fn in enumerate(all_frames)
    ]
    write_keypoints_csv(csv_path, csv_rows)

    def _collect_mano(results):
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

    det_rate_l = n_detected_left / max(n_frames, 1)
    det_rate_r = n_detected_right / max(n_frames, 1)
    summary_path = out_dir / "hand_detection_summary.txt"
    summary_path.write_text("\n".join([
        f"n_frames: {n_frames}",
        f"n_detected_left: {n_detected_left}  ({det_rate_l:.1%})",
        f"n_detected_right: {n_detected_right}  ({det_rate_r:.1%})",
        f"crop_scale: {args.crop_scale}",
        f"rescale_factor: {args.rescale_factor}",
        f"smooth_sigma: {args.smooth_sigma}",
        f"device: {args.device}",
        f"hamer_checkpoint: {checkpoint}",
    ]) + "\n")

    print(f"hand_keypoints_csv: {csv_path}")
    print(f"hand_mano_pkl: {pkl_path}")
    print(f"hand_summary: {summary_path}")
    print(f"detection_rate_left: {det_rate_l:.1%}")
    print(f"detection_rate_right: {det_rate_r:.1%}")


if __name__ == "__main__":
    main()
