#!/usr/bin/env python3
"""Per-frame diagnostic for the HaMeR + GVHMR hand stitch.

For each selected frame, re-runs HaMeR and writes:
  - the 256x256 ViT input with HaMeR's predicted keypoints
  - the bbox crop with HaMeR's reprojection (orange) and the stitched
    fingertips (green)
  - the full frame with both overlays plus the GVHMR wrist
  - a text dump of all numbers, and a summary markdown across frames

Run in the hamer env:
  conda run -n hamer python -m scripts.shared.hands.diagnose_hand_overlay \
    --sample-dir samples/basketball_01 --frames 1,60,140
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import smplx
from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD


MANO_WRIST = 0
MANO_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
MANO_PALM_JOINTS = [0, 5, 9, 13, 17]
FINGER_CHAINS = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
]
GVHMR_LEFT_WRIST = 20
GVHMR_RIGHT_WRIST = 21


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
    n = gvhmr["transl"].shape[0]
    model = smplx.create(
        str(body_models_root), model_type="smplx", gender="neutral", ext="npz",
        use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=n,
    )
    with torch.inference_mode():
        out = model(
            body_pose=torch.from_numpy(gvhmr["body_pose"]),
            betas=torch.from_numpy(gvhmr["betas"]),
            global_orient=torch.from_numpy(gvhmr["global_orient"]),
            transl=torch.from_numpy(gvhmr["transl"]),
            return_verts=False,
        )
    j = out.joints.detach().cpu().numpy().astype(np.float64)
    return np.stack([j[:, GVHMR_LEFT_WRIST, :], j[:, GVHMR_RIGHT_WRIST, :]], axis=1)


def project(xyz, K):
    z = np.clip(xyz[:, 2:3], 1e-6, None)
    uv = (xyz[:, :2] / z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv


def make_bbox(cx, cy, size, w, h):
    half = size / 2.0
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(float(w), cx + half)
    y2 = min(float(h), cy + half)
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return None
    return np.array([x1, y1, x2, y2, 1.0], dtype=np.float32)


@dataclass
class HamerOutputs:
    keypoints_3d: np.ndarray
    vertices: np.ndarray
    pred_cam: np.ndarray
    pred_cam_t: np.ndarray
    box_center: np.ndarray
    box_size: float
    img_patch: np.ndarray


def run_hamer(img_bgr, bbox, is_right, model, model_cfg, device, rescale_factor):
    dataset = ViTDetDataset(
        model_cfg, img_bgr, bbox[np.newaxis, :],
        right=np.array([float(is_right)]),
        rescale_factor=rescale_factor,
    )
    if len(dataset) == 0:
        return None
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    item = next(iter(loader))
    item_dev = recursive_to(item, device)
    with torch.no_grad():
        out = model(item_dev)
    return HamerOutputs(
        keypoints_3d=out["pred_keypoints_3d"][0].cpu().numpy().astype(np.float64),
        vertices=out["pred_vertices"][0].cpu().numpy().astype(np.float64),
        pred_cam=out["pred_cam"][0].cpu().numpy().astype(np.float64),
        pred_cam_t=out["pred_cam_t"][0].cpu().numpy().astype(np.float64),
        box_center=item["box_center"][0].cpu().numpy().astype(np.float64),
        box_size=float(item["box_size"][0].item()),
        img_patch=item["img"][0].cpu().numpy().astype(np.float32),
    )


def project_to_patch(kpts_3d, pred_cam_t, patch_size, focal):
    K = np.array([
        [focal, 0, patch_size / 2.0],
        [0, focal, patch_size / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)
    return project(kpts_3d + pred_cam_t[None, :], K)


def project_to_full(kpts_3d, pred_cam_t, box_center, box_size, patch_size, focal):
    patch_uv = project_to_patch(kpts_3d, pred_cam_t, patch_size, focal)
    x1 = box_center[0] - box_size / 2.0
    y1 = box_center[1] - box_size / 2.0
    return np.stack([
        x1 + patch_uv[:, 0] * (box_size / patch_size),
        y1 + patch_uv[:, 1] * (box_size / patch_size),
    ], axis=1)


def draw_skeleton(img, kpts, color, joint_radius=3):
    h, w = img.shape[:2]
    for chain in FINGER_CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            pa, pb = kpts[a], kpts[b]
            if 0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h:
                cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         color, 1, cv2.LINE_AA)
    for p in kpts:
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img, (int(p[0]), int(p[1])), joint_radius, color, -1, cv2.LINE_AA)
    for tip in MANO_TIPS.values():
        p = kpts[tip]
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img, (int(p[0]), int(p[1])), joint_radius + 2, color, 1, cv2.LINE_AA)


def draw_x(img, p, color, size=10):
    h, w = img.shape[:2]
    if not (0 <= p[0] < w and 0 <= p[1] < h):
        return
    x, y = int(p[0]), int(p[1])
    cv2.line(img, (x - size, y - size), (x + size, y + size), color, 2, cv2.LINE_AA)
    cv2.line(img, (x - size, y + size), (x + size, y - size), color, 2, cv2.LINE_AA)


def denormalize_patch(patch):
    img = patch.copy()
    for c in range(3):
        img[c] = img[c] * DEFAULT_STD[c] + DEFAULT_MEAN[c]
    img = np.clip(img.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


@dataclass
class FrameDiag:
    frame: int
    side: str
    detected: bool
    bbox_xyxy: tuple
    crop_size_px: float
    bbox_final_size: float
    patch_size: int
    downsampling: float
    ball_overlap_frac: float | None
    gvhmr_wrist_2d: np.ndarray
    hamer_wrist_2d_full: np.ndarray
    stitched_wrist_2d: np.ndarray
    pred_cam_t: np.ndarray
    notes: list


def diagnose_frame(
    frame_num, frame_path, K_full, wrists_3d, ball_mask_path,
    crop_scale, rescale_factor, model, model_cfg, device, out_dir,
):
    img_bgr = cv2.imread(str(frame_path))
    if img_bgr is None:
        return []
    h, w = img_bgr.shape[:2]
    crop_size_px = crop_scale * min(w, h)

    ball_mask = None
    if ball_mask_path and ball_mask_path.exists():
        ball_mask = cv2.imread(str(ball_mask_path), cv2.IMREAD_GRAYSCALE)

    patch_size = int(model_cfg.MODEL.IMAGE_SIZE)
    focal = float(model_cfg.EXTRA.FOCAL_LENGTH)

    diags = []
    full_overlay = img_bgr.copy()

    for side_idx, (side, is_right) in enumerate([("left", False), ("right", True)]):
        wrist_3d = wrists_3d[side_idx]
        gvhmr_uv = project(wrist_3d[None, :], K_full)[0]
        bbox = make_bbox(gvhmr_uv[0], gvhmr_uv[1], crop_size_px, w, h)
        notes = []

        if bbox is None:
            notes.append("bbox degenerate (wrist near image edge)")
            diags.append(FrameDiag(
                frame=frame_num, side=side, detected=False,
                bbox_xyxy=(0, 0, 0, 0), crop_size_px=crop_size_px,
                bbox_final_size=0.0, patch_size=patch_size, downsampling=0.0,
                ball_overlap_frac=None, gvhmr_wrist_2d=gvhmr_uv,
                hamer_wrist_2d_full=np.zeros(2), stitched_wrist_2d=gvhmr_uv,
                pred_cam_t=np.zeros(3), notes=notes,
            ))
            continue

        bbox_final_size = rescale_factor * max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        downsampling = bbox_final_size / float(patch_size)

        ball_overlap = None
        if ball_mask is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            sub = ball_mask[y1:y2, x1:x2]
            if sub.size:
                ball_overlap = float((sub > 127).sum()) / float(sub.size)

        try:
            outs = run_hamer(img_bgr, bbox, is_right, model, model_cfg, device, rescale_factor)
        except Exception as e:
            notes.append(f"HaMeR exception: {type(e).__name__}: {e}")
            outs = None

        if outs is None:
            notes.append("HaMeR returned no output")
            diags.append(FrameDiag(
                frame=frame_num, side=side, detected=False,
                bbox_xyxy=tuple(bbox[:4].tolist()), crop_size_px=crop_size_px,
                bbox_final_size=bbox_final_size, patch_size=patch_size,
                downsampling=downsampling, ball_overlap_frac=ball_overlap,
                gvhmr_wrist_2d=gvhmr_uv, hamer_wrist_2d_full=np.zeros(2),
                stitched_wrist_2d=gvhmr_uv, pred_cam_t=np.zeros(3), notes=notes,
            ))
            continue

        patch_kpts = project_to_patch(outs.keypoints_3d, outs.pred_cam_t, patch_size, focal)
        full_kpts_hamer = project_to_full(
            outs.keypoints_3d, outs.pred_cam_t,
            outs.box_center, outs.box_size, patch_size, focal,
        )

        wrist_local = outs.keypoints_3d[MANO_WRIST]
        stitched_3d = outs.keypoints_3d - wrist_local + wrist_3d
        stitched_uv = project(stitched_3d, K_full)

        patch_img = denormalize_patch(outs.img_patch)
        if not is_right:
            patch_img = patch_img[:, ::-1].copy()
            patch_kpts_view = patch_kpts.copy()
            patch_kpts_view[:, 0] = patch_size - patch_kpts_view[:, 0]
        else:
            patch_kpts_view = patch_kpts
        draw_skeleton(patch_img, patch_kpts_view, color=(0, 255, 255))
        cv2.putText(patch_img, f"{side} patch  ds={downsampling:.2f}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"frame_{frame_num:04d}_{side}_patch.png"), patch_img)

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        crop_view = img_bgr[y1:y2, x1:x2].copy()
        draw_skeleton(crop_view, full_kpts_hamer - np.array([x1, y1]), color=(0, 165, 255))
        draw_skeleton(crop_view, stitched_uv - np.array([x1, y1]), color=(0, 255, 0))
        draw_x(crop_view, gvhmr_uv - np.array([x1, y1]), color=(255, 255, 0), size=6)
        cv2.putText(crop_view, f"{side}  orange=HaMeR  green=stitched  cyan=GVHMR wrist",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"frame_{frame_num:04d}_{side}_crop.png"), crop_view)

        cv2.rectangle(full_overlay, (x1, y1), (x2, y2), (200, 200, 200), 1)
        draw_skeleton(full_overlay, full_kpts_hamer, color=(0, 165, 255), joint_radius=2)
        draw_skeleton(full_overlay, stitched_uv, color=(0, 255, 0), joint_radius=2)
        draw_x(full_overlay, gvhmr_uv, color=(255, 255, 0), size=8)
        if ball_mask is not None:
            tint = np.zeros_like(full_overlay)
            tint[ball_mask > 127] = (0, 0, 200)
            full_overlay[:] = cv2.addWeighted(full_overlay, 1.0, tint, 0.25, 0)

        diags.append(FrameDiag(
            frame=frame_num, side=side, detected=True,
            bbox_xyxy=tuple(bbox[:4].tolist()),
            crop_size_px=crop_size_px, bbox_final_size=bbox_final_size,
            patch_size=patch_size, downsampling=downsampling,
            ball_overlap_frac=ball_overlap, gvhmr_wrist_2d=gvhmr_uv,
            hamer_wrist_2d_full=full_kpts_hamer[MANO_WRIST],
            stitched_wrist_2d=stitched_uv[MANO_WRIST],
            pred_cam_t=outs.pred_cam_t, notes=notes,
        ))

    cv2.putText(full_overlay, f"frame {frame_num}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"frame_{frame_num:04d}_full.png"), full_overlay)

    lines = [f"# Frame {frame_num} HaMeR diagnostic", ""]
    for d in diags:
        lines += [
            f"## {d.side} hand",
            f"detected: {d.detected}",
            f"gvhmr_wrist_2d: ({d.gvhmr_wrist_2d[0]:.1f}, {d.gvhmr_wrist_2d[1]:.1f})",
            f"bbox_xyxy: {tuple(round(v, 1) for v in d.bbox_xyxy)}",
            f"crop_size_px: {d.crop_size_px:.1f}",
            f"bbox_final_size: {d.bbox_final_size:.1f}",
            f"patch_size: {d.patch_size}",
            f"downsampling: {d.downsampling:.3f}",
            f"ball_overlap_frac: "
            f"{'n/a' if d.ball_overlap_frac is None else f'{d.ball_overlap_frac:.1%}'}",
            f"hamer_wrist_2d_full: ({d.hamer_wrist_2d_full[0]:.1f}, {d.hamer_wrist_2d_full[1]:.1f})",
            f"stitched_wrist_2d: ({d.stitched_wrist_2d[0]:.1f}, {d.stitched_wrist_2d[1]:.1f})",
            f"pred_cam_t: ({d.pred_cam_t[0]:.4f}, {d.pred_cam_t[1]:.4f}, {d.pred_cam_t[2]:.4f})",
        ]
        if d.notes:
            lines.append("notes:")
            for n in d.notes:
                lines.append(f"  - {n}")
        lines.append("")
    (out_dir / f"frame_{frame_num:04d}_diag.txt").write_text("\n".join(lines))

    return diags


def parse_frames(spec, n_total):
    if spec.strip().lower() == "auto":
        return [int(round(x)) for x in np.linspace(1, max(n_total, 1), 6)]
    return [int(tok) for tok in spec.split(",") if tok.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument(
        "--body-model-root", type=Path,
        default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"),
    )
    parser.add_argument("--hamer-checkpoint", type=Path, default=None)
    parser.add_argument("--crop-scale", type=float, default=0.28)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--frames", type=str, default="auto")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    results = args.sample_dir / "results"
    out_dir = results / "hands" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    gvhmr = read_gvhmr(results / "gvhmr" / "result.pkl")
    wrists = build_gvhmr_wrists(gvhmr, args.body_model_root)
    n_gvhmr = wrists.shape[0]
    K = gvhmr["K_fullimg"]
    if K.ndim == 2:
        K = K[np.newaxis].repeat(n_gvhmr, axis=0)
    elif K.shape[0] == 1:
        K = np.repeat(K, n_gvhmr, axis=0)

    # HaMeR loads checkpoints via relative _DATA/... paths, so chdir into its
    # source directory before load_hamer.
    hamer_src = Path(__file__).resolve().parent.parent.parent / "third-party" / "hamer"
    ckpt = str(
        Path(args.hamer_checkpoint).resolve()
        if args.hamer_checkpoint
        else hamer_src / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    )
    cwd = os.getcwd()
    os.chdir(hamer_src)
    try:
        model, model_cfg = load_hamer(ckpt)
    finally:
        os.chdir(cwd)
    model = model.to(device).eval()

    frame_paths = {int(p.stem): p for p in sorted((args.sample_dir / "frames").glob("*.png"))}
    if not frame_paths:
        raise RuntimeError(f"no PNG frames in {args.sample_dir / 'frames'}")

    masks_dir = results / "segmentation" / "masks"
    selected = parse_frames(args.frames, n_gvhmr)
    print(f"Diagnosing frames: {selected}")
    print(f"HaMeR cfg: IMAGE_SIZE={model_cfg.MODEL.IMAGE_SIZE} "
          f"FOCAL_LENGTH={model_cfg.EXTRA.FOCAL_LENGTH}")
    print(f"crop_scale={args.crop_scale}  rescale_factor={args.rescale_factor}")

    all_diags = []
    for fn in selected:
        if fn not in frame_paths:
            print(f"  frame {fn} not found, skipping")
            continue
        idx = min(max(fn - 1, 0), n_gvhmr - 1)
        mask_path = masks_dir / f"{fn:05d}.png"
        if not mask_path.exists():
            mask_path = masks_dir / f"{fn}.png"
        diags = diagnose_frame(
            fn, frame_paths[fn], K[idx], wrists[idx],
            mask_path if mask_path.exists() else None,
            args.crop_scale, args.rescale_factor,
            model, model_cfg, device, out_dir,
        )
        all_diags.extend(diags)
        for d in diags:
            print(f"  frame={d.frame} {d.side}: ds={d.downsampling:.2f}  "
                  f"gvhmr=({d.gvhmr_wrist_2d[0]:.0f},{d.gvhmr_wrist_2d[1]:.0f})  "
                  f"hamer_full=({d.hamer_wrist_2d_full[0]:.0f},{d.hamer_wrist_2d_full[1]:.0f})  "
                  f"stitched=({d.stitched_wrist_2d[0]:.0f},{d.stitched_wrist_2d[1]:.0f})  "
                  f"ball%={('n/a' if d.ball_overlap_frac is None else f'{d.ball_overlap_frac:.0%}')}")

    lines = [
        "# HaMeR hand-overlay diagnostic summary",
        "",
        f"- sample_dir: `{args.sample_dir}`",
        f"- HaMeR IMAGE_SIZE: {model_cfg.MODEL.IMAGE_SIZE}",
        f"- HaMeR FOCAL_LENGTH: {model_cfg.EXTRA.FOCAL_LENGTH}",
        f"- crop_scale: {args.crop_scale}",
        f"- rescale_factor: {args.rescale_factor}",
        "",
        "| frame | side | ds | crop_px | bbox_final | gvhmr_uv | hamer_full_uv | stitched_uv | ball% | notes |",
        "|---|---|---:|---:|---:|---|---|---|---:|---|",
    ]
    for d in all_diags:
        lines.append(
            f"| {d.frame} | {d.side} | {d.downsampling:.2f} | "
            f"{d.crop_size_px:.0f} | {d.bbox_final_size:.0f} | "
            f"({d.gvhmr_wrist_2d[0]:.0f},{d.gvhmr_wrist_2d[1]:.0f}) | "
            f"({d.hamer_wrist_2d_full[0]:.0f},{d.hamer_wrist_2d_full[1]:.0f}) | "
            f"({d.stitched_wrist_2d[0]:.0f},{d.stitched_wrist_2d[1]:.0f}) | "
            f"{'n/a' if d.ball_overlap_frac is None else f'{d.ball_overlap_frac:.0%}'} | "
            f"{'; '.join(d.notes) if d.notes else ''} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"\nWrote: {out_dir / 'summary.md'}")
    print(f"Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
