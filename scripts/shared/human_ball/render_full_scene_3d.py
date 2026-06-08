#!/usr/bin/env python3
"""Put the body, hands and object together in one 3D scene and render it.

Everything the pipeline produces ends up in the same GVHMR camera frame, so it
all just drops into one pyrender scene: the GVHMR SMPL-X body, the HaMeR fingers
stitched onto it, and the object at its lifted trajectory (DA3 contact-phase by
default). Two outputs:

  overlay.mp4   mesh drawn back over the input video, semi-transparent so you
                can eyeball the fit against the real person
  world.mp4     orbiting camera on a clean floor, HOI-paper style

The object is a plain sphere unless you pass --object-mesh (e.g. a SAM 3D
Objects export). The trajectory has no rotation, which is fine for a ball; an
asymmetric mesh would sit there unrotated until we estimate orientation.

Run it as a file, not -m: it imports the overlay renderer sitting next to it.
  conda run -n gvhmr python scripts/shared/human_ball/render_full_scene_3d.py --sample-dir samples/basketball_01
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# NumPy 2.0 dropped np.infty and pyrender's shadow code still reaches for it.
# Put it back so SHADOWS_DIRECTIONAL doesn't blow up.
if not hasattr(np, "infty"):
    np.infty = np.inf

# Grab the SMPL-X build + camera helpers from the overlay renderer next door
# instead of duplicating them (works because we run this as a file).
from render_smplx_pyrender_overlay import (
    BODY_COLORS,
    _CAM_FLIP,
    build_vertices,
    composite,
    load_smplx_params,
    make_camera,
)

import pyrender
import trimesh

PAPER_BODY = (0.62, 0.63, 0.68)   # neutral grey, if you want it
TAN_BODY = (0.87, 0.72, 0.54)     # the warm tone the HOI-PAGE figures use
BODY_PALETTE = {**BODY_COLORS, "paper": PAPER_BODY, "tan": TAN_BODY}
BALL_COLOR = (0.86, 0.42, 0.12)
GROUND_COLOR = (0.86, 0.86, 0.89)
BG_COLOR = np.array([0.97, 0.975, 0.98])

# OpenCV camera (y down, z forward) to a normal y-up world for the free camera.
# It's a reflection, which matters later when we build meshes (see render_world).
_CV_TO_WORLD = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def read_object_trajectory(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                rows[int(row["frame"])] = {
                    "t": np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64),
                    "r": float(row.get("radius_m", 0.12) or 0.12),
                }
            except (KeyError, ValueError):
                continue
    if not rows:
        raise RuntimeError(f"No object trajectory rows in {path}")
    return rows


def make_object_mesh(radius: float, object_mesh: Path | None, object_scale: float) -> trimesh.Trimesh:
    # default is a coloured sphere; if a real mesh is given, centre it and scale it
    # to the trajectory radius unless an explicit scale was passed
    if object_mesh is not None:
        mesh = trimesh.load(str(object_mesh), force="mesh")
        mesh.apply_translation(-mesh.bounding_box.centroid)
        if object_scale > 0:
            mesh.apply_scale(object_scale)
        else:
            half = float(np.max(mesh.bounding_box.extents)) / 2.0
            if half > 1e-6:
                mesh.apply_scale(radius / half)
        return mesh
    sphere = trimesh.creation.uv_sphere(radius=radius, count=[24, 24])
    sphere.visual.vertex_colors = np.tile(
        (np.array([*BALL_COLOR, 1.0]) * 255).astype(np.uint8), (sphere.vertices.shape[0], 1)
    )
    return sphere


def to_pyrender_mesh(tri, color=None, smooth=True, roughness=0.55):
    if color is not None:
        mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=roughness, baseColorFactor=(*color, 1.0)
        )
        return pyrender.Mesh.from_trimesh(tri, material=mat, smooth=smooth)
    return pyrender.Mesh.from_trimesh(tri, smooth=smooth)


def add_lights(scene):
    # cheap camera-relative key/fill for the overlay, no shadows
    key = np.array([[1, 0, 0, 0], [0, 0, -1, 3], [0, 1, 0, 3], [0, 0, 0, 1]], dtype=np.float64)
    fill = np.array([[-1, 0, 0, 0], [0, 0, 1, 3], [0, 1, 0, 3], [0, 0, 0, 1]], dtype=np.float64)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=key)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=fill)


def add_world_lights(scene, center, radius):
    # one directional key (this is what casts the shadow, since only directional
    # shadows are on), plus a couple of point lights for fill that don't. ambient
    # carries the rest. keeps a single clean shadow under the figure.
    d = max(radius, 1.0)
    up = np.array([0.0, 1.0, 0.0])
    key = pyrender.DirectionalLight(color=np.array([1.0, 0.99, 0.97]), intensity=2.0)
    scene.add(key, pose=look_at(center + np.array([1.2, 2.8, 1.6]) * d, center, up))
    for off, inten in [(np.array([-2.4, 1.4, 1.2]), 6.0), (np.array([0.0, 1.6, -2.6]), 4.0)]:
        pose = np.eye(4)
        pose[:3, 3] = center + off * d
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=inten * d * d), pose=pose)


def look_at(eye, target, up):
    # standard camera-to-world; pyrender cameras look down their -Z
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-9)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def ground_quad(y, cx, cz, size):
    verts = np.array([
        [cx - size, y, cz - size], [cx + size, y, cz - size],
        [cx + size, y, cz + size], [cx - size, y, cz + size],
    ], dtype=np.float64)
    # both windings so it shows from either side
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def render_overlay(renderer, verts, faces, ball_tri, ball_center, K, body_color, ball_color):
    scene = pyrender.Scene(ambient_light=np.array([0.3, 0.3, 0.3, 1.0]))
    scene.add(make_camera(K), pose=np.eye(4))
    body = trimesh.Trimesh(vertices=verts @ _CAM_FLIP.T, faces=faces, process=False)
    scene.add(to_pyrender_mesh(body, color=body_color))
    bt = ball_tri.copy()
    bt.apply_translation(ball_center)
    bt.vertices = bt.vertices @ _CAM_FLIP.T
    scene.add(to_pyrender_mesh(bt, color=ball_color, roughness=0.45))
    add_lights(scene)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES)
    return color[::-1]  # pyrender hands the image back bottom-up


def render_world(renderer, verts, faces, ball_tri, ball_center, cam_pose, yfov,
                 body_color, ball_color, ground, center, radius):
    scene = pyrender.Scene(ambient_light=np.array([0.5, 0.5, 0.53, 1.0]), bg_color=BG_COLOR)
    cam = pyrender.PerspectiveCamera(yfov=yfov, znear=0.05, zfar=100.0)
    scene.add(cam, pose=cam_pose)
    # _CV_TO_WORLD flips handedness (det -1), which flips the triangle winding too.
    # Reverse the faces so normals still point outward - otherwise the outward
    # faces get culled and you end up looking straight through the head.
    body = trimesh.Trimesh(vertices=verts @ _CV_TO_WORLD.T, faces=faces[:, ::-1], process=False)
    scene.add(to_pyrender_mesh(body, color=body_color, roughness=0.6))
    bt = ball_tri.copy()
    bt.apply_translation(ball_center)
    bt.vertices = bt.vertices @ _CV_TO_WORLD.T
    bt.faces = bt.faces[:, ::-1]
    scene.add(to_pyrender_mesh(bt, color=ball_color, roughness=0.45))
    if ground is not None:
        scene.add(to_pyrender_mesh(ground, color=GROUND_COLOR, smooth=False, roughness=0.95))
    add_world_lights(scene, center, radius)
    # SKIP_CULL_FACES on top as a belt-and-braces against any stray winding
    flags = (pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
             | pyrender.RenderFlags.SKIP_CULL_FACES)
    color, _ = renderer.render(scene, flags=flags)
    return color  # y-up world + look_at is already the right way up


def pick_trajectory(results_dir: Path, override: Path | None) -> Path:
    if override is not None:
        if not override.exists():
            raise RuntimeError(f"trajectory not found: {override}")
        return override
    # best available: contact-refined DA3, then plain DA3, then the old sphere one
    for cand in [
        results_dir / "pose6d_sharedcam_contactphase_depthv3" / "ball_pose6d_sharedcam_contactphase_trajectory.csv",
        results_dir / "pose6d_sharedcam_depthv3" / "ball_pose6d_sharedcam_trajectory.csv",
        results_dir / "pose6d_sharedcam" / "ball_pose6d_sharedcam_trajectory.csv",
    ]:
        if cand.exists():
            return cand
    raise RuntimeError("No object trajectory found; run the lifting stage first.")


def main():
    ap = argparse.ArgumentParser(description="Body + hands + object in one 3D scene.")
    ap.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--object-trajectory", type=Path, default=None,
                    help="override the auto-picked trajectory csv")
    ap.add_argument("--object-mesh", type=Path, default=None,
                    help="real object mesh (.glb/.obj/.ply), e.g. from SAM 3D Objects; "
                         "otherwise a sphere")
    ap.add_argument("--object-scale", type=float, default=0.0,
                    help="metric scale for --object-mesh (0 = fit to the trajectory radius)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--body-color", choices=tuple(BODY_PALETTE), default="tan")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="overlay mesh opacity over the video; low so you can see the real "
                         "person underneath (doesn't touch the world view)")
    ap.add_argument("--orbit-turns", type=float, default=1.0, help="full camera turns over the clip")
    ap.add_argument("--elevation-deg", type=float, default=16.0)
    ap.add_argument("--mode", choices=["both", "overlay", "world"], default="both")
    args = ap.parse_args()

    body_color = BODY_PALETTE[args.body_color]
    results_dir = args.sample_dir / "results"
    out_dir = results_dir / "renders" / "full_scene_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = pick_trajectory(results_dir, args.object_trajectory)
    print(f"object trajectory: {traj_path}")
    obj = read_object_trajectory(traj_path)

    gvhmr, stitched = load_smplx_params(results_dir)
    n_frames = gvhmr["transl"].shape[0]
    K = gvhmr["K_fullimg"]
    if K.ndim == 2:
        K = K[np.newaxis].repeat(n_frames, axis=0)
    elif K.shape[0] == 1 and n_frames > 1:
        K = np.repeat(K, n_frames, axis=0)

    print("Running SMPL-X forward pass...")
    verts_all, faces = build_vertices(args.body_model_root, gvhmr, stitched)

    median_r = float(np.median([v["r"] for v in obj.values()]))
    ball_tri = make_object_mesh(median_r, args.object_mesh, args.object_scale)
    # sphere gets an orange material; a real mesh keeps whatever texture it shipped with
    ball_color = None if args.object_mesh else BALL_COLOR
    print(f"object: {'mesh ' + args.object_mesh.name if args.object_mesh else 'sphere proxy'} "
          f"(radius ~{median_r:.3f} m)")

    frame_paths = sorted((args.sample_dir / "frames").glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames in {args.sample_dir / 'frames'}")
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    # line everything up by frame number
    frame_nums = [int(p.stem) if p.stem.isdigit() else i + 1 for i, p in enumerate(frame_paths)]
    body_idx = [min(max(fn - 1, 0), n_frames - 1) for fn in frame_nums]

    # ball position per displayed frame; hold the last one if a frame is missing
    ball_centers = []
    last = obj[min(obj)]["t"]
    for fn in frame_nums:
        if fn in obj:
            last = obj[fn]["t"]
        ball_centers.append(last.copy())

    # Frame the orbit around everything the body+ball cover over the whole clip,
    # so the camera doesn't have to chase anything.
    floor_y_cv = float(np.percentile(verts_all[..., 1], 99.0))  # ~the feet (y points down here)
    pts_world = []
    for bi, bc in zip(body_idx, ball_centers):
        pts_world.append(verts_all[bi] @ _CV_TO_WORLD.T)
        pts_world.append((bc[None] @ _CV_TO_WORLD.T))
    pts_world = np.concatenate(pts_world, axis=0)
    cmin, cmax = pts_world.min(0), pts_world.max(0)
    center_w = 0.5 * (cmin + cmax)
    scene_radius = float(np.linalg.norm(cmax - cmin)) * 0.5
    yfov = np.deg2rad(50.0)
    orbit_dist = scene_radius / np.tan(yfov / 2.0) * 1.1
    ground = ground_quad(-floor_y_cv, center_w[0], center_w[2], max(scene_radius * 3.0, 3.0))
    elev = np.deg2rad(args.elevation_deg)
    up = np.array([0.0, 1.0, 0.0])

    def writer(name):
        return cv2.VideoWriter(str(out_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    do_overlay = args.mode in ("both", "overlay")
    do_world = args.mode in ("both", "world")
    w_over = writer("overlay.mp4") if do_overlay else None
    w_world = writer("world.mp4") if do_world else None

    print(f"Rendering {len(frame_paths)} frames ({W}x{H}); mode={args.mode}")
    n = len(frame_paths)
    for i, (fp, bi, bc) in enumerate(zip(frame_paths, body_idx, ball_centers)):
        if do_overlay:
            img = cv2.imread(str(fp))
            rgba = render_overlay(renderer, verts_all[bi], faces, ball_tri, bc, K[bi], body_color, ball_color)
            frame = composite(img, rgba, args.alpha)
            w_over.write(frame)
            if i == 0:
                cv2.imwrite(str(out_dir / "overlay_preview.png"), frame)
        if do_world:
            theta = 2.0 * np.pi * args.orbit_turns * (i / max(n - 1, 1))
            eye = center_w + orbit_dist * np.array([
                np.cos(elev) * np.sin(theta), np.sin(elev), np.cos(elev) * np.cos(theta)
            ])
            cam_pose = look_at(eye, center_w, up)
            rgba = render_world(renderer, verts_all[bi], faces, ball_tri, bc, cam_pose, yfov,
                                body_color, ball_color, ground, center_w, scene_radius)
            frame = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            w_world.write(frame)
            if i == 0:
                cv2.imwrite(str(out_dir / "world_preview.png"), frame)
        if (i + 1) % 24 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}")

    if w_over:
        w_over.release()
        print(f"overlay_mp4: {out_dir / 'overlay.mp4'}")
    if w_world:
        w_world.release()
        print(f"world_mp4: {out_dir / 'world.mp4'}")
    renderer.delete()


if __name__ == "__main__":
    main()
