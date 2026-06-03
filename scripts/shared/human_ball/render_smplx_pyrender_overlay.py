#!/usr/bin/env python3
"""Render the SMPL-X body over the original video frames."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pyrender
import trimesh


# OpenCV camera (y-down, z-forward) -> OpenGL (y-up, z-back). pyrender writes
# its image bottom-up, so we only flip z here and undo the y with [::-1] later.
_CAM_FLIP = np.diag([1.0, 1.0, -1.0]).astype(np.float32)

BODY_COLORS = {
    "blue":  (0.65, 0.74, 0.86),
    "pink":  (0.90, 0.70, 0.72),
    "green": (0.65, 0.86, 0.72),
    "white": (0.92, 0.92, 0.92),
}


def load_pkl(path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_smplx_params(results_dir):
    raw = load_pkl(results_dir / "gvhmr" / "result.pkl")
    p = raw["smpl_params_incam"]
    gvhmr = {
        "body_pose": np.asarray(p["body_pose"], dtype=np.float32),
        "betas": np.asarray(p["betas"], dtype=np.float32),
        "global_orient": np.asarray(p["global_orient"], dtype=np.float32),
        "transl": np.asarray(p["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(raw["K_fullimg"], dtype=np.float32),
    }

    stitched_pkl = results_dir / "hands" / "stitched_smplx_params.pkl"
    stitched = None
    if stitched_pkl.exists():
        stitched = load_pkl(stitched_pkl)
        print(f"Using stitched hand params ({stitched_pkl.name})")
    else:
        print("No stitched hand params; rendering with flat hands.")
    return gvhmr, stitched


def build_vertices(body_model_root, gvhmr, stitched):
    import smplx

    n_frames = gvhmr["transl"].shape[0]
    betas = gvhmr["betas"]
    if betas.shape[0] == 1:
        betas = np.repeat(betas, n_frames, axis=0)

    # flat_hand_mean=True: HaMeR predicts the absolute finger pose, not a
    # residual on MANO's curled mean. With flat_hand_mean=False the curl mean
    # would be added on top and all fingers render as fists.
    model = smplx.create(
        str(body_model_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=n_frames,
    )

    body_pose = (
        stitched["body_pose"]
        if (stitched is not None and "body_pose" in stitched)
        else gvhmr["body_pose"]
    )
    kwargs = dict(
        body_pose=torch.from_numpy(body_pose),
        betas=torch.from_numpy(betas),
        global_orient=torch.from_numpy(gvhmr["global_orient"]),
        transl=torch.from_numpy(gvhmr["transl"]),
        return_verts=True,
    )
    if stitched is not None:
        kwargs["left_hand_pose"] = torch.from_numpy(stitched["left_hand_pose"])
        kwargs["right_hand_pose"] = torch.from_numpy(stitched["right_hand_pose"])

    with torch.inference_mode():
        out = model(**kwargs)

    verts = out.vertices.detach().cpu().numpy().astype(np.float32)
    faces = model.faces.astype(np.int32)
    return verts, faces


def make_camera(K):
    return pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        znear=0.01,
        zfar=100.0,
    )


def render_frame(renderer, verts, faces, camera, body_color):
    verts_gl = verts @ _CAM_FLIP.T
    mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices=verts_gl, faces=faces, process=False),
        material=pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.6,
            baseColorFactor=(*body_color, 1.0),
        ),
        smooth=True,
    )

    scene = pyrender.Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
    scene.add(camera, pose=np.eye(4, dtype=np.float64))
    scene.add(mesh)

    key_pose = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 2],
        [0, 1,  0, 2],
        [0, 0,  0, 1],
    ], dtype=np.float64)
    fill_pose = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 2],
        [ 0, 1, 0, 2],
        [ 0, 0, 0, 1],
    ], dtype=np.float64)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=key_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=fill_pose)

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return color


def composite(frame_bgr, render_rgba, alpha):
    mask = (render_rgba[:, :, 3] > 0).astype(np.float32)
    render_bgr = render_rgba[:, :, :3][:, :, ::-1]
    mask3 = mask[:, :, None] * alpha
    out = frame_bgr.astype(np.float32) * (1 - mask3) + render_bgr.astype(np.float32) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument(
        "--body-model-root",
        type=Path,
        default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"),
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--body-color", choices=tuple(BODY_COLORS), default="blue")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--no-hand-stitch", action="store_true")
    parser.add_argument("--out", type=str, default="smplx_overlay.mp4")
    args = parser.parse_args()

    body_color = BODY_COLORS[args.body_color]
    results_dir = args.sample_dir / "results"
    renders_dir = results_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    gvhmr, stitched = load_smplx_params(results_dir)
    if args.no_hand_stitch:
        stitched = None

    n_frames = gvhmr["transl"].shape[0]
    K = gvhmr["K_fullimg"]
    if K.ndim == 2:
        K = K[np.newaxis].repeat(n_frames, axis=0)
    elif K.shape[0] == 1 and n_frames > 1:
        K = np.repeat(K, n_frames, axis=0)

    print("Running SMPL-X forward pass...")
    verts_all, faces = build_vertices(args.body_model_root, gvhmr, stitched)
    print(f"  {n_frames} frames, {verts_all.shape[1]} vertices, {faces.shape[0]} faces")

    frame_paths = sorted((args.sample_dir / "frames").glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No PNG frames in {args.sample_dir / 'frames'}")
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]

    out_path = renders_dir / args.out
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H)
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    preview_written = False

    print(f"Rendering {len(frame_paths)} frames ({W}x{H})...")
    for i, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        if img is None:
            writer.write(np.zeros((H, W, 3), dtype=np.uint8))
            continue

        try:
            frame_num = int(frame_path.stem)
        except ValueError:
            frame_num = i + 1
        idx = min(max(frame_num - 1, 0), n_frames - 1)

        render_rgba = render_frame(
            renderer, verts_all[idx], faces, make_camera(K[idx]), body_color
        )[::-1]
        out_frame = composite(img, render_rgba, args.alpha)
        writer.write(out_frame)

        if not preview_written:
            preview = renders_dir / args.out.replace(".mp4", "_preview.png")
            cv2.imwrite(str(preview), out_frame)
            preview_written = True

        if (i + 1) % 24 == 0 or (i + 1) == len(frame_paths):
            print(f"  {i + 1}/{len(frame_paths)}")

    writer.release()
    renderer.delete()
    print(f"overlay_mp4: {out_path}")


if __name__ == "__main__":
    main()
