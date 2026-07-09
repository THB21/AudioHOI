#!/usr/bin/env python3
"""Inherit object rotation from the grasping hand (or foot).

The lifted trajectories are translation-only (qw..qz identity). During a grasp
the object is rigidly attached to the hand, so its rotation is simply the hand
rotation times a constant grasp offset:

    R_obj(t) = R_hand(t) @ R_off        (inside each grasp segment)

Per-frame hand frames come from SMPL-X joints (wrist + two knuckles give an
orthonormal basis in the GVHMR incam frame — the same frame as tx,ty,tz).
Grasp segments are contiguous runs of human_contact_state==1. The constant
offset R_off is solved once per segment at its best-visibility frame t_ref:

  axis mode (elongated objects, e.g. hammer): align the mesh long axis (+Z of
    the GLB by default, or the longest bbox axis of --object-mesh) with the
    SAM2 mask's 2D principal axis at t_ref, lifted to 3D in the image plane at
    the object depth. Axis sign is picked to point AWAY from the wrist (handle
    end is in the hand); --flip-axis overrides.
  upright mode (e.g. mug): R_obj(t_ref) = identity, i.e. the object keeps its
    canonical (as-loaded, upright) orientation at t_ref and then follows the
    hand's rotation *changes*.

Between/outside segments the rotation is SLERP'd across the gap (constant
before the first and after the last segment), then lightly smoothed with a
±window quaternion mean.

Output: a NEW rotation-augmented CSV next to the input trajectory
(*_trajectory_rot.csv) with qw,qx,qy,qz filled; the original is not touched.
render_full_scene_3d.py auto-prefers the _rot.csv.

  conda run -n gvhmr python scripts/shared/human_ball/contact/inherit_grasp_rotation.py \
      --sample-dir samples/03_hammer --mode axis --object-mesh assets/object_meshes/hammer.glb
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# SMPL-X joint triplets used to build an orthonormal part frame:
# (anchor, radial landmark, ulnar landmark) ->
#   x = normalize(radial - anchor); z = normalize(x × (ulnar - anchor)); y = z × x
# hands: wrist (20/21), index knuckle (25/40), ring-side knuckle (34/49)
# feet:  ankle (7/8), foot (10/11), heel landmark (62/65)
_FRAME_JOINTS = {
    "left_hand": (20, 25, 34),
    "right_hand": (21, 40, 49),
    "left_foot": (7, 10, 62),
    "right_foot": (8, 11, 65),
}


def load_params(results_dir: Path) -> dict:
    """GVHMR incam SMPL-X params, preferring contact-refined then stitched-hand
    body pose so the hand frames match what the renderer draws."""
    raw = pickle.load((results_dir / "gvhmr" / "result.pkl").open("rb"))
    p = raw["smpl_params_incam"]
    params = {
        "body_pose": np.asarray(p["body_pose"], dtype=np.float32),
        "betas": np.asarray(p["betas"], dtype=np.float32),
        "global_orient": np.asarray(p["global_orient"], dtype=np.float32),
        "transl": np.asarray(p["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(raw["K_fullimg"], dtype=np.float32),
    }
    for pkl in [results_dir / "contact_refine" / "contact_refined_smplx_params.pkl",
                results_dir / "hands" / "stitched_smplx_params.pkl"]:
        if pkl.exists():
            extra = pickle.load(pkl.open("rb"))
            params.update({k: np.asarray(v) for k, v in extra.items() if isinstance(v, np.ndarray)})
            print(f"body params: + {pkl.name}")
            break
    return params


def smplx_joints(body_model_root: Path, params: dict) -> np.ndarray:
    import torch
    import smplx

    n = params["body_pose"].shape[0]
    betas = params["betas"]
    if betas.shape[0] == 1:
        betas = np.repeat(betas, n, axis=0)
    model = smplx.create(str(body_model_root), model_type="smplx", gender="neutral", ext="npz",
                         use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=n)
    kwargs = dict(
        body_pose=torch.from_numpy(params["body_pose"].astype(np.float32)),
        betas=torch.from_numpy(betas.astype(np.float32)),
        global_orient=torch.from_numpy(params["global_orient"].astype(np.float32)),
        transl=torch.from_numpy(params["transl"].astype(np.float32)),
        return_verts=False,
    )
    if "left_hand_pose" in params:
        kwargs["left_hand_pose"] = torch.from_numpy(params["left_hand_pose"].astype(np.float32))
        kwargs["right_hand_pose"] = torch.from_numpy(params["right_hand_pose"].astype(np.float32))
    with torch.inference_mode():
        out = model(**kwargs)
    return out.joints.detach().cpu().numpy().astype(np.float64)  # [N, 127, 3] incam


def part_frames(joints: np.ndarray, part: str) -> np.ndarray:
    """Per-frame orthonormal basis R_hand(t) ∈ SO(3) for a hand/foot, [N,3,3].
    Columns are the frame axes expressed in the camera frame."""
    a, i, u = _FRAME_JOINTS[part]
    anchor, radial, ulnar = joints[:, a], joints[:, i], joints[:, u]
    x = radial - anchor
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    z = np.cross(x, ulnar - anchor)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=2)  # columns
    # guard against degenerate frames (collinear landmarks): reuse the neighbor
    dets = np.linalg.det(R)
    for t in range(len(R)):
        if not (0.9 < dets[t] < 1.1):
            R[t] = R[t - 1] if t > 0 else np.eye(3)
    return R


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector a onto unit vector b (Rodrigues)."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180°: rotate about any axis perpendicular to a
        p = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, p)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def mesh_long_axis(object_mesh: Path | None) -> np.ndarray:
    """Object-local long axis: longest bbox extent of the GLB, else assume +Z."""
    if object_mesh is None:
        return np.array([0.0, 0.0, 1.0])
    import trimesh
    mesh = trimesh.load(str(object_mesh), force="mesh")
    ext = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
    axis = np.zeros(3)
    axis[int(np.argmax(ext))] = 1.0
    print(f"mesh long axis: {axis.astype(int).tolist()} (extents {np.round(ext, 3).tolist()})")
    return axis


def mask_principal_axis(mask_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """2D principal axis (unit du,dv) + centroid (u,v) of a binary mask via PCA."""
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.nonzero(mask > 127)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    return evecs[:, -1] / np.linalg.norm(evecs[:, -1]), mean


def read_observations(path: Path) -> dict[int, dict]:
    obs = {}
    if not path.exists():
        return obs
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                obs[int(r["frame"])] = r
            except (KeyError, ValueError):
                continue
    return obs


def find_segments(state: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Contiguous runs of state==1 (inclusive index ranges) of length >= min_len."""
    segs, start = [], None
    for i, v in enumerate(state):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                segs.append((start, i - 1))
            start = None
    if start is not None and len(state) - start >= min_len:
        segs.append((start, len(state) - 1))
    return segs


def smooth_rotations(rots: Rotation, halfwin: int) -> Rotation:
    if halfwin <= 0:
        return rots
    n = len(rots)
    out = []
    for t in range(n):
        lo, hi = max(0, t - halfwin), min(n, t + halfwin + 1)
        out.append(rots[lo:hi].mean())
    return Rotation.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill trajectory quaternions by inheriting hand rotation during grasps.")
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--trajectory-csv", type=Path, default=None,
                    help="input trajectory (default: pose6d_sharedcam_contactphase_depthv3)")
    ap.add_argument("--observations-csv", type=Path, default=None,
                    help="object_observations.csv (default: results/object_observations/)")
    ap.add_argument("--masks-dir", type=Path, default=None,
                    help="SAM2 mask dir with %%05d_mask.png (default: results/masks/)")
    ap.add_argument("--object-mesh", type=Path, default=None,
                    help="GLB used for rendering; determines the object-local long axis "
                         "in axis mode (default: assume +Z)")
    ap.add_argument("--mode", choices=["auto", "axis", "upright"], default="auto",
                    help="axis: align mesh long axis with the mask 2D principal axis at t_ref "
                         "(elongated objects); upright: object canonical/upright at t_ref (mug). "
                         "auto: axis if median mask aspect_ratio < 0.6")
    ap.add_argument("--flip-axis", action="store_true",
                    help="flip the lifted mask axis 180° (if the mesh +long-axis end is the handle)")
    ap.add_argument("--min-segment-frames", type=int, default=5)
    ap.add_argument("--smooth-halfwin", type=int, default=2,
                    help="±frames for the final quaternion-mean smoothing (0 = off)")
    args = ap.parse_args()

    results_dir = args.sample_dir / "results"
    traj_csv = args.trajectory_csv or (
        results_dir / "pose6d_sharedcam_contactphase_depthv3" / "ball_pose6d_sharedcam_contactphase_trajectory.csv")
    obs_csv = args.observations_csv or (results_dir / "object_observations" / "object_observations.csv")
    masks_dir = args.masks_dir or (results_dir / "masks")

    with traj_csv.open() as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No trajectory rows in {traj_csv}")
    n = len(rows)
    frames = np.array([int(r["frame"]) for r in rows])
    state = np.array([int(float(r.get("human_contact_state", 0) or 0)) for r in rows])
    parts = [str(r.get("active_part", "") or "") for r in rows]
    tz = np.array([float(r["tz"]) for r in rows])
    obs = read_observations(obs_csv)

    # mode
    mode = args.mode
    if mode == "auto":
        ars = [float(r["aspect_ratio"]) for r in obs.values()
               if r.get("aspect_ratio") not in (None, "",) ]
        med_ar = float(np.median(ars)) if ars else 1.0
        mode = "axis" if med_ar < 0.6 else "upright"
        print(f"auto mode: median mask aspect_ratio {med_ar:.3f} -> {mode}")
    print(f"mode: {mode}")

    # hand/foot frames from SMPL-X (incam)
    params = load_params(results_dir)
    joints = smplx_joints(args.body_model_root, params)
    n_body = joints.shape[0]
    K = params["K_fullimg"]
    K0 = K[0] if K.ndim == 3 else K
    fx, fy, cx, cy = float(K0[0, 0]), float(K0[1, 1]), float(K0[0, 2]), float(K0[1, 2])

    frames_by_part = {}

    def hand_R(part: str, ti: int) -> np.ndarray:
        if part not in frames_by_part:
            frames_by_part[part] = part_frames(joints, part)
        bi = min(max(int(frames[ti]) - 1, 0), n_body - 1)
        return frames_by_part[part][bi]

    # grasp segments
    segs = find_segments(state, args.min_segment_frames)
    if not segs:
        raise RuntimeError("No grasp segments (human_contact_state==1 runs >= "
                           f"{args.min_segment_frames} frames) — nothing to inherit.")
    long_axis = mesh_long_axis(args.object_mesh) if mode == "axis" else None

    def segment_part(s: int, e: int) -> str:
        counts = {}
        for t in range(s, e + 1):
            if parts[t] in _FRAME_JOINTS:
                counts[parts[t]] = counts.get(parts[t], 0) + 1
        if not counts:
            return ""
        return max(counts, key=counts.get)

    def best_visibility_frame(s: int, e: int) -> int:
        best_t, best_area = (s + e) // 2, -1.0
        for t in range(s, e + 1):
            o = obs.get(int(frames[t]))
            if not o:
                continue
            try:
                if int(float(o.get("mask_available", 0) or 0)) != 1:
                    continue
                area = float(o.get("mask_area_px", 0) or 0)
            except ValueError:
                continue
            if area > best_area:
                best_area, best_t = area, t
        return best_t

    # per-frame rotation, filled segment by segment
    R_obj: list[np.ndarray | None] = [None] * n
    for (s, e) in segs:
        part = segment_part(s, e)
        if not part:
            print(f"segment frames {frames[s]}-{frames[e]}: no usable active_part, skipped")
            continue
        t_ref = best_visibility_frame(s, e)
        R_ref = hand_R(part, t_ref)

        if mode == "axis":
            # lift the mask's 2D principal axis to a 3D in-plane direction at
            # the object depth; sign points away from the grasping anchor joint
            pa = mask_principal_axis(masks_dir / f"{int(frames[t_ref]):05d}_mask.png")
            if pa is None:
                print(f"segment frames {frames[s]}-{frames[e]}: no mask at t_ref, using upright offset")
                R_axis = np.eye(3)
            else:
                axis2d, centroid = pa
                anchor = joints[min(max(int(frames[t_ref]) - 1, 0), n_body - 1), _FRAME_JOINTS[part][0]]
                anchor_px = np.array([fx * anchor[0] / anchor[2] + cx, fy * anchor[1] / anchor[2] + cy])
                if float(np.dot(axis2d, centroid - anchor_px)) < 0:
                    axis2d = -axis2d
                if args.flip_axis:
                    axis2d = -axis2d
                d_cam = np.array([axis2d[0] / fx, axis2d[1] / fy, 0.0])
                d_cam /= np.linalg.norm(d_cam)
                R_axis = rot_between(long_axis, d_cam)
                ang = np.degrees(np.arccos(np.clip((np.trace(R_axis) - 1) / 2, -1, 1)))
                print(f"segment frames {frames[s]}-{frames[e]} [{part}] t_ref={frames[t_ref]} "
                      f"axis2d=({axis2d[0]:+.2f},{axis2d[1]:+.2f}) tilt {ang:.1f} deg (tz {tz[t_ref]:.2f} m)")
            R_off = R_ref.T @ R_axis
        else:
            # upright: object keeps its canonical (identity) orientation at t_ref
            R_off = R_ref.T
            print(f"segment frames {frames[s]}-{frames[e]} [{part}] t_ref={frames[t_ref]} upright anchor")

        for t in range(s, e + 1):
            R_obj[t] = hand_R(part, t) @ R_off

    solved = [t for t in range(n) if R_obj[t] is not None]
    if not solved:
        raise RuntimeError("All segments skipped — no rotations solved.")

    # fill gaps: constant before/after, SLERP between segments
    first, last = solved[0], solved[-1]
    for t in range(0, first):
        R_obj[t] = R_obj[first]
    for t in range(last + 1, n):
        R_obj[t] = R_obj[last]
    t = first
    while t <= last:
        if R_obj[t] is None:
            g0 = t - 1                     # last solved before the gap
            g1 = t
            while R_obj[g1] is None:
                g1 += 1                    # first solved after the gap
            sl = Slerp([g0, g1], Rotation.from_matrix(np.stack([R_obj[g0], R_obj[g1]])))
            for u in range(g0 + 1, g1):
                R_obj[u] = sl(u).as_matrix()
            t = g1
        t += 1

    rots = Rotation.from_matrix(np.stack(R_obj))
    rots = smooth_rotations(rots, args.smooth_halfwin)
    quats = rots.as_quat()  # (x, y, z, w)

    out_csv = traj_csv.with_name(traj_csv.stem + "_rot.csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows):
            qx, qy, qz, qw = quats[i]
            if qw < 0:  # canonical sign
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
            r["qw"], r["qx"], r["qy"], r["qz"] = (f"{v:.6f}" for v in (qw, qx, qy, qz))
            w.writerow(r)
    print(f"segments: {len(segs)}, frames with inherited rotation: {len(solved)}/{n}")
    print(f"rotation-augmented trajectory: {out_csv}")


if __name__ == "__main__":
    main()
