#!/usr/bin/env python3
"""Incremental object lift - the step-by-step ladder.

The idea is to start from the irreducible minimum and add one loss term at a time,
re-rendering between each, so we can see exactly what each term buys.

    L0  back-projection only      x,y,z = backproj(u, v, Z_DA3)
    L1  + confidence-weighted     pull z toward Z_DA3 by depth_conf, + smoothness   <-- HERE
        depth & smoothness
    L2  + ground / support        ray<->ground-plane intersection on contact
    L3  + hand/foot contact       pin to the contacting part at audio/visual frames

L0 is purely the 2D track (object centre from `object_observations.csv`) pushed out to
the model depth (`depth/object_depth.csv`, DA3 aligned to the metric human) and
back-projected through the camera intrinsics. No optimization, no priors - it is the
baseline whose failure modes (vertical float, depth jitter) motivate the later terms.

L1 keeps the 2D track fixed and solves only the per-frame depth z. Raw DA3 depth on a
small/fast object is full of outliers (football: jumps up to 20 m), but `depth_conf` is
low exactly on those frames - so the data pull is `w_depth * conf_t * (z_t - Z_DA3,t)`:
high-confidence frames anchor the depth, low-confidence outliers get almost no pull and
are carried by the acceleration-smoothness term `w_smooth * (z_{t+1}-2z_t+z_{t-1})`. A
robust `soft_l1` loss suppresses what's left. Set `--smooth-weight 0` to recover L0.

Output: ``results/<out-subdir>/ball_pose6d_sharedcam_trajectory.csv`` (frame,time,tx,ty,
tz,radius_m,... ) - the schema the scene renderer reads.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _ff(row: dict, key: str, default: float = float("nan")) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def read_observations(path: Path) -> dict[int, dict[str, float]]:
    """Per-frame 2D object centre + apparent radius (px)."""
    out: dict[int, dict[str, float]] = {}
    for r in _read_csv(path):
        fr = int(float(r["frame"]))
        out[fr] = {
            "time": _ff(r, "time", (fr - 1) / 24.0),
            "u": _ff(r, "center_x"),
            "v": _ff(r, "center_y"),
            "radius_px": _ff(r, "enclosing_radius_px", float("nan")),
            # lowest silhouette pixel from the mask/bbox - the object's "bottom",
            # purely observational (no shape assumption); used by the ground term
            "v_bottom": _ff(r, "bbox_y2", _ff(r, "bottom_y", float("nan"))),
            "conf": _ff(r, "observation_conf", 1.0),
        }
    return out


def read_depth(path: Path) -> dict[int, dict[str, float]]:
    """Per-frame metric object depth (DA3, affine-aligned to the human)."""
    out: dict[int, dict[str, float]] = {}
    for r in _read_csv(path):
        fr = int(float(r["frame"]))
        out[fr] = {"z": _ff(r, "object_z_aligned_m"), "depth_conf": _ff(r, "depth_conf", 0.0)}
    return out


def read_K(result_pkl: Path) -> np.ndarray:
    with result_pkl.open("rb") as f:
        data = pickle.load(f)
    return np.asarray(data["K_fullimg"], dtype=np.float64)


def backproject(u: float, v: float, z: float, K: np.ndarray) -> tuple[float, float, float]:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return (u - cx) * z / fx, (v - cy) * z / fy, z


def solve_depth(z_da3: np.ndarray, conf: np.ndarray, *, w_depth: float, w_smooth: float,
                conf_power: float = 1.0,
                z_ground: np.ndarray | None = None, ground_mask: np.ndarray | None = None,
                w_ground: float = 0.0,
                z_contact: np.ndarray | None = None, contact_mask: np.ndarray | None = None,
                w_contact: float = 0.0) -> np.ndarray:
    """Solve per-frame depth z from a stack of object-agnostic terms (L1 + L2 + L3).

      L1 data  : w_depth * conf**p * (z - Z_DA3)          confidence-weighted DA3 pull
      L2 ground: w_ground  * 1[on ground] * (z - Z_grnd)  silhouette-bottom <-> floor
      L3 contact: w_contact * 1[contact]  * (z - Z_part)  object depth = contacting part
      smooth   : w_smooth * (z_{t+1}-2z_t+z_{t-1})        acceleration regularizer

    Every cue is generic (depth, depth_conf, the human's floor/body, audio timing) - no
    object size, category, gravity or ballistic prior. The DA3 pull is weak where conf is
    low; ground/contact terms (only on their gated frames) are the reliable geometry;
    smoothness carries the free frames. Robust soft_l1.
    """
    z_da3 = np.asarray(z_da3, dtype=np.float64)
    n = len(z_da3)
    finite = np.isfinite(z_da3) & (z_da3 > 0)
    w = np.where(finite, np.clip(np.asarray(conf, dtype=np.float64), 0.0, 1.0) ** conf_power, 0.0)

    gm = (np.asarray(ground_mask, bool) & np.isfinite(z_ground)) if (z_ground is not None and ground_mask is not None) else np.zeros(n, bool)
    cm = (np.asarray(contact_mask, bool) & np.isfinite(z_contact)) if (z_contact is not None and contact_mask is not None) else np.zeros(n, bool)
    zg = np.where(gm, np.nan_to_num(z_ground, nan=0.0) if z_ground is not None else 0.0, 0.0)
    zc = np.where(cm, np.nan_to_num(z_contact, nan=0.0) if z_contact is not None else 0.0, 0.0)

    # initialise from the most reliable available cue per frame: ground/contact > DA3 > interp
    z0 = z_da3.copy()
    idx = np.arange(n)
    seed = finite.copy()
    z0[gm] = zg[gm]
    z0[cm] = zc[cm]
    seed |= gm | cm
    if seed.any() and not seed.all():
        z0[~seed] = np.interp(idx[~seed], idx[seed], z0[seed])
    elif not seed.any():
        return np.full(n, 1.0)

    def residuals(z: np.ndarray) -> np.ndarray:
        out = [w_depth * w * (z - z0)]
        if w_ground > 0.0 and gm.any():
            out.append(w_ground * gm * (z - zg))
        if w_contact > 0.0 and cm.any():
            out.append(w_contact * cm * (z - zc))
        if n >= 3:
            out.append(w_smooth * (z[2:] - 2.0 * z[1:-1] + z[:-2]))
        return np.concatenate([np.ravel(r) for r in out]).astype(np.float64)

    res = least_squares(residuals, z0.copy(), method="trf", loss="soft_l1",
                        f_scale=0.5, max_nfev=800)
    return np.maximum(res.x, 0.20)


def compute_ground_y(body_models_root: Path, result_pkl: Path, smooth_sigma: float = 3.0) -> np.ndarray:
    """Per-frame ground height Y_g in camera coords (Y points down), taken as the 99th
    percentile of the SMPL-X body vertices (~the feet/soles), lightly time-smoothed.

    Assumes the floor is roughly horizontal in the camera frame (constant Y) - consistent
    with the scene renderer's ground plane. Needs smplx (run in the gvhmr env)."""
    import pickle
    import smplx
    import torch
    from scipy.ndimage import gaussian_filter1d

    with result_pkl.open("rb") as f:
        data = pickle.load(f)
    p = data["smpl_params_incam"]
    m = smplx.create(str(body_models_root), model_type="smplx", gender="neutral", ext="npz",
                     use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=p["transl"].shape[0])
    with torch.inference_mode():
        out = m(body_pose=torch.from_numpy(np.asarray(p["body_pose"], np.float32)),
                betas=torch.from_numpy(np.asarray(p["betas"], np.float32)),
                global_orient=torch.from_numpy(np.asarray(p["global_orient"], np.float32)),
                transl=torch.from_numpy(np.asarray(p["transl"], np.float32)), return_verts=True)
    verts = out.vertices.detach().cpu().numpy().astype(np.float64)  # [N, V, 3]
    yg = np.percentile(verts[..., 1], 99.0, axis=1)
    return gaussian_filter1d(yg, sigma=smooth_sigma, mode="nearest")


def compute_part_centers(body_models_root: Path, result_pkl: Path) -> dict[str, np.ndarray]:
    """Per-frame 3D centres of the generic contact parts (hands, feet) in camera coords.

    Object-agnostic: it only knows about the human body, not the object. Which part is
    'the contact' is chosen later as the part nearest the object - no per-object rule."""
    import pickle
    import sys as _sys
    import smplx
    import torch
    _cp = Path(__file__).resolve().parents[1] / "human_ball" / "contact"
    if str(_cp) not in _sys.path:
        _sys.path.insert(0, str(_cp))
    from contact_part_utils import build_contact_part_centers

    with result_pkl.open("rb") as f:
        data = pickle.load(f)
    p = data["smpl_params_incam"]
    m = smplx.create(str(body_models_root), model_type="smplx", gender="neutral", ext="npz",
                     use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=p["transl"].shape[0])
    with torch.inference_mode():
        out = m(body_pose=torch.from_numpy(np.asarray(p["body_pose"], np.float32)),
                betas=torch.from_numpy(np.asarray(p["betas"], np.float32)),
                global_orient=torch.from_numpy(np.asarray(p["global_orient"], np.float32)),
                transl=torch.from_numpy(np.asarray(p["transl"], np.float32)), return_verts=False)
    joints = out.joints.detach().cpu().numpy().astype(np.float64)
    return build_contact_part_centers(joints)


def read_audio_frames(path: Path) -> set[int]:
    """Audio onset frames (generic onset detector - works for any impacting object)."""
    out: set[int] = set()
    if not path.exists():
        return out
    for r in _read_csv(path):
        if r.get("audio_frame"):
            out.add(int(float(r["audio_frame"])))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--out-subdir", type=str, default="pose6d_L1_confsmooth")
    ap.add_argument("--observation-csv", type=Path, default=None)
    ap.add_argument("--depth-csv", type=Path, default=None)
    ap.add_argument("--default-radius-m", type=float, default=0.11,
                    help="fallback metric radius when no apparent radius is available")
    ap.add_argument("--depth-weight", type=float, default=1.0, help="L1 confidence-weighted depth pull")
    ap.add_argument("--smooth-weight", type=float, default=5.0,
                    help="L1 depth-acceleration smoothness (0 = closed-form L0 back-projection)")
    ap.add_argument("--conf-power", type=float, default=1.0,
                    help="sharpen depth_conf weighting (conf**power); >1 trusts only the most confident frames")
    # L2: ground / support (silhouette bottom <-> floor plane from the human)
    ap.add_argument("--ground", action="store_true", help="enable L2: silhouette-bottom <-> ground-plane depth on contact frames")
    ap.add_argument("--ground-weight", type=float, default=8.0, help="L2 ground-depth pull strength")
    ap.add_argument("--ground-mode", choices=["floor_state", "all"], default="floor_state",
                    help="which frames get the ground term: detected floor contacts, or every frame")
    # L3: contact to the nearest human body part at audio/visual contact frames
    ap.add_argument("--contact", action="store_true", help="enable L3: object depth = nearest human part at contact frames")
    ap.add_argument("--contact-weight", type=float, default=8.0, help="L3 contact-depth pull strength")
    ap.add_argument("--contact-mode", choices=["audio", "human_state", "audio_or_state"], default="audio_or_state",
                    help="which frames are contacts: audio onsets, detected human contact, or either")
    ap.add_argument("--audio-events-csv", type=Path, default=None)
    ap.add_argument("--contact-state-csv", type=Path, default=None)
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    args = ap.parse_args(argv)

    results = args.sample_dir / "results"
    obs_csv = args.observation_csv or (results / "object_observations" / "object_observations.csv")
    depth_csv = args.depth_csv or (results / "depth" / "object_depth.csv")

    obs = read_observations(obs_csv)
    depth = read_depth(depth_csv)
    K_all = read_K(results / "gvhmr" / "result.pkl")

    frames = sorted(set(obs) & set(depth))
    if not frames:
        raise RuntimeError(f"no frames shared between {obs_csv} and {depth_csv}")
    # drop frames with no usable 2D centre (depth outliers are kept - L1 handles them)
    frames = [fr for fr in frames if np.isfinite(obs[fr]["u"]) and np.isfinite(obs[fr]["v"])]

    z_da3 = np.array([depth[fr]["z"] for fr in frames], dtype=np.float64)
    conf = np.array([depth[fr]["depth_conf"] for fr in frames], dtype=np.float64)
    Ks = [K_all[fr - 1] if K_all.ndim == 3 and fr - 1 < len(K_all) else (K_all if K_all.ndim == 2 else K_all[-1]) for fr in frames]

    # --- L2: ground ray<->plane depth on contact frames ---------------------------
    z_ground = ground_mask = None
    if args.ground:
        yg_all = compute_ground_y(args.body_model_root, results / "gvhmr" / "result.pkl")
        # floor-contact gate
        if args.ground_mode == "all":
            on_ground = {fr: True for fr in frames}
        else:
            state_csv = args.contact_state_csv or (results / "contact_candidates" / "contact_state_frames.csv")
            state = {int(float(r["frame"])): r for r in _read_csv(state_csv)} if state_csv.exists() else {}
            on_ground = {fr: (state.get(fr, {}).get("floor_contact_state", "0") == "1") for fr in frames}
        z_ground = np.full(len(frames), np.nan)
        ground_mask = np.zeros(len(frames), bool)
        for i, fr in enumerate(frames):
            if not on_ground.get(fr, False):
                continue
            K = Ks[i]
            fy, cyv = K[1, 1], K[1, 2]
            yg = yg_all[fr - 1] if fr - 1 < len(yg_all) else yg_all[-1]
            # the object's lowest *observed* pixel (mask/bbox bottom) sits on the floor:
            # (v_bottom - cy)/fy * Z = Y_ground  -> Z. Purely observational, no shape prior.
            v_bot = obs[fr]["v_bottom"] if np.isfinite(obs[fr]["v_bottom"]) else obs[fr]["v"]
            denom = (v_bot - cyv) / fy
            if denom <= 1e-4:
                continue
            zg = yg / denom
            if 0.3 < zg < 30.0:
                z_ground[i] = zg
                ground_mask[i] = True

    # --- L3: contact depth = nearest human body part at contact frames -------------
    z_contact = contact_mask = None
    if args.contact:
        parts3d = compute_part_centers(args.body_model_root, results / "gvhmr" / "result.pkl")
        # contact frames from generic cues: audio onsets and/or detected human contact
        audio_fr = read_audio_frames(args.audio_events_csv or (results / "events" / "audio_events.csv"))
        state_csv = args.contact_state_csv or (results / "contact_candidates" / "contact_state_frames.csv")
        state = {int(float(r["frame"])): r for r in _read_csv(state_csv)} if state_csv.exists() else {}
        def _is_contact(fr):
            a = fr in audio_fr
            h = any(state.get(fr, {}).get(k, "0") == "1" for k in ("human_contact_state", "anchor_contact_state"))
            return {"audio": a, "human_state": h, "audio_or_state": a or h}[args.contact_mode]
        z_contact = np.full(len(frames), np.nan)
        contact_mask = np.zeros(len(frames), bool)
        for i, fr in enumerate(frames):
            if not _is_contact(fr):
                continue
            idx = fr - 1
            o3 = backproject(obs[fr]["u"], obs[fr]["v"], z_da3[i] if np.isfinite(z_da3[i]) and z_da3[i] > 0 else 5.0, Ks[i])
            o3 = np.asarray(o3)
            # pick the body part nearest the object (geometric, no per-object rule)
            best_z, best_d = None, np.inf
            for arr in parts3d.values():
                if idx >= len(arr):
                    continue
                p = np.asarray(arr[idx], dtype=np.float64)
                if not np.all(np.isfinite(p)):
                    continue
                d = float(np.linalg.norm(p - o3))
                if d < best_d:
                    best_d, best_z = d, float(p[2])
            if best_z is not None and 0.3 < best_z < 30.0:
                z_contact[i] = best_z
                contact_mask[i] = True

    if args.smooth_weight > 0.0 or args.ground or args.contact:
        z_solved = solve_depth(
            z_da3, conf, w_depth=args.depth_weight, w_smooth=args.smooth_weight, conf_power=args.conf_power,
            z_ground=z_ground, ground_mask=ground_mask, w_ground=(args.ground_weight if args.ground else 0.0),
            z_contact=z_contact, contact_mask=contact_mask, w_contact=(args.contact_weight if args.contact else 0.0),
        )
        parts = ["L1"] + (["L2ground"] if args.ground else []) + (["L3contact"] if args.contact else [])
        source = "_".join(parts)
    else:
        z_solved = z_da3
        source = "L0_backproject"

    rows = []
    for i, fr in enumerate(frames):
        o, d = obs[fr], depth[fr]
        K = K_all[fr - 1] if K_all.ndim == 3 and fr - 1 < len(K_all) else (K_all if K_all.ndim == 2 else K_all[-1])
        z = float(z_solved[i])
        if not np.isfinite(z) or z <= 0:
            continue
        tx, ty, tz = backproject(o["u"], o["v"], z, K)
        # metric radius from the apparent radius at this depth (object-agnostic);
        # fall back to a fixed radius when no silhouette radius exists
        if np.isfinite(o["radius_px"]) and o["radius_px"] > 0:
            radius_m = float(o["radius_px"] * z / K[0, 0])
        else:
            radius_m = args.default_radius_m
        rows.append({
            "frame": fr, "time": f"{o['time']:.6f}",
            "tx": f"{tx:.6f}", "ty": f"{ty:.6f}", "tz": f"{tz:.6f}",
            "qw": "1.000000", "qx": "0.000000", "qy": "0.000000", "qz": "0.000000",
            "radius_m": f"{radius_m:.6f}", "coord_frame": "shared_fullimg_cam",
            "u_obs": f"{o['u']:.3f}", "v_obs": f"{o['v']:.3f}",
            "radius_obs_px": f"{o['radius_px']:.3f}" if np.isfinite(o["radius_px"]) else "",
            "z_da3_raw": f"{d['z']:.6f}" if np.isfinite(d["z"]) else "",
            "depth_conf": f"{d['depth_conf']:.6f}", "source": source,
        })

    out_dir = results / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ball_pose6d_sharedcam_trajectory.csv"
    fields = ["frame", "time", "tx", "ty", "tz", "qw", "qx", "qy", "qz", "radius_m",
              "coord_frame", "u_obs", "v_obs", "radius_obs_px", "z_da3_raw", "depth_conf", "source"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ng = int(np.count_nonzero(ground_mask)) if ground_mask is not None else 0
    nc = int(np.count_nonzero(contact_mask)) if contact_mask is not None else 0
    print(f"[{source}] {len(rows)}/{len(frames)} frames"
          + (f", {ng} ground-anchored" if args.ground else "")
          + (f", {nc} contact-anchored" if args.contact else "") + f" -> {out_csv}")


if __name__ == "__main__":
    main()
