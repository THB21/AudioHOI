#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
POSE_SOLVERS = REPO / "scripts/shared/generic_contact_pipeline/components/pose/solvers"
for path in (REPO, HERE, POSE_SOLVERS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import fit_mug_articraft_keyframe_pose as base  # noqa: E402
import fit_mug_body_only_cylinder_pose as bodyfit  # noqa: E402
try:
    import render_mug_segmented_body_with_visual_handle_contact_region as m9  # noqa: E402
except ModuleNotFoundError:
    m9 = None  # deleted during cleanup; only needed by legacy rendering paths

BODY_H = bodyfit.BODY_H
Z_MID = BODY_H * 0.5

PART_STYLE = {
    'body_shell': ((225, 225, 225), 1),
    'rim_ring': ((255, 0, 255), 2),
    'bottom_disk': ((0, 255, 255), 2),
    'handle_loop': ((0, 210, 255), 2),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def ff(row: dict[str, str], key: str, default: float = math.nan) -> float:
    v = row.get(key, '')
    try:
        return float(v)
    except Exception:
        return default


def read_pose_sequence(path: Path) -> dict[int, np.ndarray]:
    out = {}
    for row in read_csv(path):
        fr = int(float(row['frame']))
        if any(row.get(k, "") != "" for k in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")):
            quat = [ff(row, "qx", 0.0), ff(row, "qy", 0.0), ff(row, "qz", 0.0), ff(row, "qw", 1.0)]
            yaw, pitch, roll = Rotation.from_quat(quat).as_euler("zyx")
            out[fr] = np.array(
                [
                    ff(row, "tx", ff(row, "x", 0.0)),
                    ff(row, "ty", ff(row, "y", 0.0)),
                    ff(row, "tz", ff(row, "z", 3.0)),
                    yaw,
                    pitch,
                    roll,
                    ff(row, "scale", 1.0),
                ],
                dtype=float,
            )
        else:
            out[fr] = np.array([ff(row, k) for k in ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'scale']], dtype=float)
    return out


def load_obj(path: Path, origin: tuple[float, float, float]) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    ox, oy, oz = origin
    with path.open() as f:
        for line in f:
            if line.startswith('v '):
                x, y, z = [float(v) for v in line.split()[1:4]]
                # Articraft: z is vertical-up, x/y are cup-plane axes.
                # Pipeline body pose: y is image/down vertical, z is camera-depth local axis.
                # Mirror x so the materialized right-side handle matches the left-side canonical
                # geometry already used by the M5/M9 mug diagnostics.
                verts.append([-(x + ox), Z_MID - (z + oz), y + oy])
            elif line.startswith('f '):
                idx = []
                for tok in line.split()[1:]:
                    idx.append(int(tok.split('/')[0]) - 1)
                if len(idx) >= 3:
                    for i in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[i], idx[i + 1]))
    return np.asarray(verts, dtype=float), faces


def mesh_edges(faces: list[tuple[int, int, int]], stride: int = 1) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for j, (a, b, c) in enumerate(faces):
        if stride > 1 and j % stride != 0:
            continue
        for u, v in [(a, b), (b, c), (c, a)]:
            if u > v:
                u, v = v, u
            edges.add((u, v))
    return sorted(edges)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def load_articraft_meshes(root: Path) -> dict[str, tuple[np.ndarray, list[tuple[int, int]]]]:
    mesh_dir = root / 'assets' / 'meshes'
    specs = {
        'body_shell': ('body_shell.obj', (0.0, 0.0, 0.0), 4),
        'rim_ring': ('rim_ring.obj', (0.0, 0.0, 0.099), 2),
        'bottom_disk': ('bottom_disk.obj', (0.0, 0.0, 0.0035), 2),
        'handle_loop': ('handle_loop.obj', (0.0, 0.0, 0.0), 2),
    }
    out = {}
    for name, (file_name, origin, stride) in specs.items():
        verts, faces = load_obj(mesh_dir / file_name, origin)
        out[name] = (verts, mesh_edges(faces, stride=stride))
    return out


def load_articraft_meshes_solid(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray, list]]:
    """Like load_articraft_meshes but returns (verts, face_array_Nx3, edges) for solid rendering."""
    mesh_dir = root / 'assets' / 'meshes'
    specs = {
        'body_shell': ('body_shell.obj', (0.0, 0.0, 0.0), 4),
        'rim_ring': ('rim_ring.obj', (0.0, 0.0, 0.099), 2),
        'bottom_disk': ('bottom_disk.obj', (0.0, 0.0, 0.0035), 2),
        'handle_loop': ('handle_loop.obj', (0.0, 0.0, 0.0), 2),
    }
    out = {}
    for name, (file_name, origin, stride) in specs.items():
        verts, faces = load_obj(mesh_dir / file_name, origin)
        out[name] = (verts, np.array(faces, dtype=np.int32), mesh_edges(faces, stride=stride))
    return out


# Solid-render base colors (BGR). Bright warm values that darken to sandy-beige after shading.
MUG_SOLID_COLORS_BGR: dict[str, np.ndarray] = {
    'body_shell':  np.array([175.0, 205.0, 222.0]),   # warm cream
    'rim_ring':    np.array([155.0, 183.0, 200.0]),   # slightly darker at rim
    'bottom_disk': np.array([145.0, 173.0, 190.0]),
    'handle_loop': np.array([175.0, 205.0, 222.0]),   # same cream as body
}

# Light dir: unit vector from surface toward the light, in camera space. Camera looks down +z;
# light sits upper-left-front (between camera and scene) so z is negative. x<0 lights the left, y<0 from above.
_SOLID_LIGHT = np.array([-0.30, -0.55, -0.78])
_SOLID_LIGHT = _SOLID_LIGHT / np.linalg.norm(_SOLID_LIGHT)
_SOLID_LIGHT = _SOLID_LIGHT / np.linalg.norm(_SOLID_LIGHT)


def _project_to_uv(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.maximum(pts_cam[:, 2], 1e-6)
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def draw_solid_mesh(
    img: np.ndarray,
    params: np.ndarray,
    K: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    base_color_bgr: np.ndarray,
    phase: float,
    ambient: float = 0.42,
    alpha: float = 0.92,
) -> None:
    """Rasterize a solid face mesh onto img (painter's algorithm + diffuse shading).

    After load_obj's x-axis negation the face winding reverses, so face normals
    computed via cross product point INWARD.  Front-facing inward normals have
    their z-component > 0 (they point toward the camera along +z depth axis).
    """
    pts_local = verts @ rot_y(phase).T
    pts_cam   = base.transform(params, pts_local)
    uv        = _project_to_uv(pts_cam, K)

    h, w = img.shape[:2]
    face_items: list[tuple[float, np.ndarray, float]] = []

    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0c, v1c, v2c = pts_cam[i0], pts_cam[i1], pts_cam[i2]
        mean_z = (v0c[2] + v1c[2] + v2c[2]) / 3.0

        e1 = v1c - v0c
        e2 = v2c - v0c
        n = np.cross(e1, e2)
        nlen = np.linalg.norm(n)
        if nlen < 1e-12:
            continue
        n /= nlen

        # inward normals with z > 0 are front-facing (surface toward the camera)
        if n[2] < 0:
            continue

        p0, p1, p2 = uv[i0], uv[i1], uv[i2]
        xs = (p0[0], p1[0], p2[0])
        ys = (p0[1], p1[1], p2[1])
        if max(xs) < -8 or min(xs) > w + 8 or max(ys) < -8 or min(ys) > h + 8:
            continue

        # Diffuse shading: use outward normal (-n) dotted with light direction.
        diffuse = float(np.clip(np.dot(-n, _SOLID_LIGHT), 0.0, 1.0))
        shade   = ambient + (1.0 - ambient) * diffuse
        pts_int = np.array([[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]]], dtype=np.int32)
        face_items.append((mean_z, pts_int, shade))

    if not face_items:
        return

    face_items.sort(key=lambda x: -x[0])   # back-to-front
    overlay = img.copy()
    for _, pts_int, shade in face_items:
        color = tuple(int(np.clip(c * shade, 0, 255)) for c in base_color_bgr)
        cv2.fillPoly(overlay, [pts_int], color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def draw_solid_mesh_from_cam(
    img: np.ndarray,
    pts_cam: np.ndarray,
    K: np.ndarray,
    faces: np.ndarray,
    base_color_bgr: np.ndarray,
    ambient: float = 0.42,
) -> list[tuple[float, np.ndarray, float]]:
    """Return painter-sorted face list from already-transformed camera-space pts.

    Does NOT draw anything — caller collects face lists from all parts, merges,
    sorts, and draws once for correct cross-part occlusion.
    """
    uv = _project_to_uv(pts_cam, K)
    h, w = img.shape[:2]
    out = []
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0c, v1c, v2c = pts_cam[i0], pts_cam[i1], pts_cam[i2]
        mean_z = (v0c[2] + v1c[2] + v2c[2]) / 3.0
        e1, e2 = v1c - v0c, v2c - v0c
        n = np.cross(e1, e2)
        nlen = np.linalg.norm(n)
        if nlen < 1e-12:
            continue
        n /= nlen
        if n[2] < 0:
            continue
        p0, p1, p2 = uv[i0], uv[i1], uv[i2]
        xs = (p0[0], p1[0], p2[0])
        ys = (p0[1], p1[1], p2[1])
        if max(xs) < -8 or min(xs) > w + 8 or max(ys) < -8 or min(ys) > h + 8:
            continue
        diffuse = float(np.clip(np.dot(-n, _SOLID_LIGHT), 0.0, 1.0))
        shade   = ambient + (1.0 - ambient) * diffuse
        pts_int = np.array([[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]]], dtype=np.int32)
        # Pack color into face item so cross-part sort works
        color_shaded = tuple(int(np.clip(c * shade, 0, 255)) for c in base_color_bgr)
        out.append((mean_z, pts_int, color_shaded))
    return out


def draw_solid_overlay_all_parts(
    img: np.ndarray,
    mesh_cam_frame: dict[str, np.ndarray],
    meshes_solid: dict[str, tuple[np.ndarray, np.ndarray, list]],
    K: np.ndarray,
    ambient: float = 0.55,
    alpha: float = 1.0,
) -> None:
    """Render all mug parts as solid faces with cross-part depth sorting."""
    all_faces: list[tuple[float, np.ndarray, tuple]] = []
    draw_order = ['body_shell', 'bottom_disk', 'rim_ring', 'handle_loop']
    for name in draw_order:
        if name not in mesh_cam_frame or name not in meshes_solid:
            continue
        _verts, faces, _edges = meshes_solid[name]
        pts_cam = mesh_cam_frame[name]
        color_base = MUG_SOLID_COLORS_BGR[name]
        all_faces.extend(draw_solid_mesh_from_cam(img, pts_cam, K, faces, color_base, ambient))
    if not all_faces:
        return
    all_faces.sort(key=lambda x: -x[0])
    overlay = img.copy()
    for _, pts_int, color in all_faces:
        cv2.fillPoly(overlay, [pts_int], color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def draw_mesh_edges(img: np.ndarray, params: np.ndarray, K: np.ndarray, verts: np.ndarray, edges: list[tuple[int, int]], color: tuple[int, int, int], thick: int, phase: float) -> None:
    pts = verts @ rot_y(phase).T
    uv = bodyfit.project_pts(params, pts, K)
    h, w = img.shape[:2]
    for a, b in edges:
        p = uv[a]
        q = uv[b]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
            continue
        # Skip wild off-screen edges to avoid giant lines when projection is bad.
        if (p[0] < -w or p[0] > 2*w or q[0] < -w or q[0] > 2*w or p[1] < -h or p[1] > 2*h or q[1] < -h or q[1] > 2*h):
            continue
        cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), color, thick, cv2.LINE_AA)


def finalize(tmp: Path, out: Path) -> None:
    ffmpeg = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
    try:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run([ffmpeg, '-y', '-i', str(tmp), '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p', str(out)], check=True, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)


def render(sample: Path, mesh_root: Path, mode: str, min_conf: float, fps: float, visibility_source: str = 'auto', vlm_path: Path | None = None) -> dict[str, str]:
    poses = read_pose_sequence(sample / 'proxy' / 'mug_body_only_cylinder_pose_segmented_sequence.csv')
    K_all = base.load_K(sample)
    vis, vis_source = m9.load_visibility_policy(sample, visibility_source, vlm_path)
    phase_by_frame, anchors = m9.build_phase_track(sample, poses, K_all, min_conf=min_conf, visibility_policy=vis)
    obs_rows = {int(r['frame']): r for r in read_csv(sample / 'results' / 'object_observations' / 'object_observations.csv')}
    meshes = load_articraft_meshes(mesh_root)
    out_dir = sample / 'results' / 'renders' / 'M12_articraft_rigid_mesh_vlm'
    out_dir.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(sample / 'frames' / '00001.png'))
    h, w = first.shape[:2]
    tmp = out_dir / f'overlay_{mode}.tmp.mp4'
    out = out_dir / f'overlay_{mode}.mp4'
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    rows = []
    for fr in sorted(poses):
        img = cv2.imread(str(sample / 'frames' / f'{fr:05d}.png'))
        if img is None:
            continue
        overlay = img.copy()
        policy = vis.get(fr, 'unknown')
        hidden = m9.visibility_kind(policy) == 'hidden'
        show_handle = (not hidden) and policy.startswith('visible')
        phase = phase_by_frame[fr]
        draw_order = ['body_shell', 'rim_ring', 'bottom_disk', 'handle_loop']
        if hidden and show_handle:
            draw_order = ['handle_loop', 'body_shell', 'rim_ring', 'bottom_disk']
        for name in draw_order:
            if name == 'handle_loop' and not show_handle:
                continue
            color, thick = PART_STYLE[name]
            if hidden and name == 'handle_loop':
                color, thick = (145, 145, 145), 1
            verts, edges = meshes[name]
            draw_mesh_edges(overlay, poses[fr], K_all[fr - 1], verts, edges, color, thick, phase)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        contact_drawn, contact_src = m9.draw_contact_region(img, obs_rows.get(fr, {}))
        label = f'M12_articraft_rigid_mesh_{mode} frame {fr:03d} phase={math.degrees(phase):.1f} visual_handle={policy} contact={contact_src}'
        cv2.putText(img, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .65, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(img, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, .65, (245,245,245), 1, cv2.LINE_AA)
        writer.write(img)
        rows.append({'frame': fr, 'time': (fr-1)/fps, 'mug_axial_phase_rad': phase, 'mug_axial_phase_deg': math.degrees(phase), 'visual_handle_policy': policy, 'visual_handle_source': vis_source, 'visual_handle_drawn': int(show_handle), 'contact_region_drawn': contact_drawn, 'contact_region_source': contact_src})
    writer.release()
    finalize(tmp, out)
    phase_csv = out_dir / f'handle_phase_{mode}.csv'
    with phase_csv.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    (out_dir / 'mug_axial_phase_anchors.json').write_text(json.dumps({'min_conf': min_conf, 'visibility_source': vis_source, 'vlm_path': str(vlm_path) if vlm_path else '', 'anchors': anchors}, indent=2))
    summary = {'render': str(out), 'phase_csv': str(phase_csv), 'mesh_root': str(mesh_root), 'num_anchors': len(anchors), 'visibility_source': vis_source, 'note': 'Articraft body/rim/bottom/handle share one mug axial phase; handle has no independent transform.'}
    (out_dir / f'outputs_{mode}.json').write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-dir', type=Path, required=True)
    ap.add_argument('--mesh-root', type=Path, default=Path('samples_known_object/02_mug/articraft/materialized_mug_mesh'))
    ap.add_argument('--mode', choices=['all', 'visible'], default='all')
    ap.add_argument('--min-conf', type=float, default=0.15)
    ap.add_argument('--fps', type=float, default=24.0)
    ap.add_argument('--visibility-source', choices=['auto', 'manual', 'vlm'], default='auto')
    ap.add_argument('--vlm-path', type=Path, default=None)
    args = ap.parse_args()
    print(json.dumps(render(args.sample_dir, args.mesh_root, args.mode, args.min_conf, args.fps, args.visibility_source, args.vlm_path), indent=2))


if __name__ == '__main__':
    main()
