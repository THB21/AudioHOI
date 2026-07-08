#!/usr/bin/env python3
"""Per-frame metric object depth from Depth Anything 3, scaled to GVHMR.

The old baseline read object depth from the apparent ball radius (Z = f*R/r),
which only works for a sphere of known size. Here we just ask DA3 for a depth
map and read off the object. The catch with mono metric depth is that the scale
wanders, so we tie it to something we already trust at metric scale in the same
camera: the GVHMR body. Sample DA3 at the projected SMPL-X joints, fit Z = a*D+b,
done. Since that fit eats any global scale, we don't even bother with DA3's own
focal*net/300 conversion and feed the raw depth in.

Writes results/depth/object_depth.csv (+ a small alignment json and a preview).
gvhmr env.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Just the body tree (pelvis -> spine -> limbs -> head). The face/hand keypoints
# that SMPL-X also returns are noisy, so leave them out of the depth anchoring.
NUM_BODY_JOINTS = 22


def load_gvhmr(results_dir: Path):
    with (results_dir / "gvhmr" / "result.pkl").open("rb") as f:
        raw = pickle.load(f)
    p = raw["smpl_params_incam"]
    return {
        "body_pose": np.asarray(p["body_pose"], dtype=np.float32),
        "betas": np.asarray(p["betas"], dtype=np.float32),
        "global_orient": np.asarray(p["global_orient"], dtype=np.float32),
        "transl": np.asarray(p["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(raw["K_fullimg"], dtype=np.float32),
    }


def build_smplx_joints(body_model_root: Path, gvhmr: dict) -> np.ndarray:
    """Run SMPL-X and hand back the body joints in the camera frame, [N, 22, 3]."""
    import smplx

    n_frames = gvhmr["transl"].shape[0]
    betas = gvhmr["betas"]
    if betas.shape[0] == 1:
        betas = np.repeat(betas, n_frames, axis=0)

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
    with torch.inference_mode():
        out = model(
            body_pose=torch.from_numpy(gvhmr["body_pose"]),
            betas=torch.from_numpy(betas),
            global_orient=torch.from_numpy(gvhmr["global_orient"]),
            transl=torch.from_numpy(gvhmr["transl"]),
            return_verts=False,
        )
    joints = out.joints.detach().cpu().numpy().astype(np.float64)
    return joints[:, :NUM_BODY_JOINTS, :]


def project(points_cam, K):
    z = np.clip(points_cam[..., 2], 1e-6, None)
    u = K[0, 0] * points_cam[..., 0] / z + K[0, 2]
    v = K[1, 1] * points_cam[..., 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def robust_affine_fit(d, z, n_iter=5):
    # plain line fit, but throw out points more than ~3 sigma off and refit a few times
    keep = np.ones(d.shape[0], dtype=bool)
    a, b = 1.0, 0.0
    for _ in range(n_iter):
        if keep.sum() < 3:
            break
        a, b = np.polyfit(d[keep], z[keep], deg=1)
        resid = np.abs((a * d + b) - z)
        mad = np.median(resid[keep]) + 1e-9
        keep = resid < 3.0 * 1.4826 * mad
    final = np.abs((a * d + b) - z)[keep]
    rms = float(np.sqrt(np.mean(final ** 2))) if final.size else float("nan")
    return float(a), float(b), rms, int(keep.sum())


def read_object_observations(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            cx, cy = row.get("center_x", ""), row.get("center_y", "")
            if not cx or not cy:
                continue
            out[int(row["frame"])] = {
                "cx": float(cx),
                "cy": float(cy),
                "r": float(row.get("enclosing_radius_px", 0.0) or 0.0),
                "time": float(row.get("time", 0.0) or 0.0),
            }
    return out


def sample_object_raw_depth(raw, frame, masks_dir, obs):
    """Median raw depth over the object. Use the mask if we have one, else a little
    disk around the tracked centre. Returns (median, n_px, spread, where_from).

    We pull the raw median and scale it later: a*median+b == median(a*depth+b),
    so there's no reason to scale the whole map first.
    """
    H, W = raw.shape
    mask_path = masks_dir / f"{frame:05d}_mask.png"
    sel = None
    source = ""
    if mask_path.exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is not None and (m > 0).any():
            sel = m > 0
            source = "mask"
    if sel is None and frame in obs:
        o = obs[frame]
        rr = max(3.0, min(o["r"] if o["r"] > 0 else 8.0, 40.0))
        yy, xx = np.ogrid[:H, :W]
        sel = (xx - o["cx"]) ** 2 + (yy - o["cy"]) ** 2 <= rr * rr
        source = "obs_disk"
    if sel is None or not sel.any():
        return float("nan"), 0, float("nan"), "none"

    vals = raw[sel]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), 0, float("nan"), "none"
    return float(np.median(vals)), int(vals.size), float(np.std(vals)), source


def sample_patch_median(dmap, u, v, rad=2):
    """Median DA3 over a small patch around (u,v). A patch is robust to a single joint pixel
    landing just off the body onto far background."""
    H, W = dmap.shape
    y0, y1 = max(0, v - rad), min(H, v + rad + 1)
    x0, x1 = max(0, u - rad), min(W, u + rad + 1)
    patch = dmap[y0:y1, x0:x1]
    patch = patch[np.isfinite(patch)]
    return float(np.median(patch)) if patch.size else float("nan")


def reject_background_joints(d, z, k=4.0):
    """Drop joints whose DA3 value is a MAD-outlier vs the body median (e.g. a joint projected
    onto the far grass/horizon). Returns the kept (d, z)."""
    ok = np.isfinite(d) & np.isfinite(z) & (d > 0)
    d, z = d[ok], z[ok]
    if d.size < 6:
        return d, z
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-9
    keep = np.abs(d - med) < k * 1.4826 * mad
    return d[keep], z[keep]


def fit_frame_affine(d, z, min_pts=6, min_span=0.05):
    # bail out if there aren't enough joints in frame, or they're all at one depth
    # (a line through a single x is meaningless). Returns (a, b, n, corr, dlo, dhi) or None.
    ok = np.isfinite(d) & np.isfinite(z) & (d > 0)
    d, z = d[ok], z[ok]
    if d.shape[0] < min_pts or float(np.ptp(d)) < min_span:
        return None
    corr = float(np.corrcoef(d, z)[0, 1]) if d.size >= 3 else float("nan")
    a, b, _, n = robust_affine_fit(d, z, n_iter=3)
    return a, b, n, corr, float(np.min(d)), float(np.max(d))


def interpolate_fill(values):
    # fill the frames where the fit failed by linear interp over time
    x = np.arange(values.shape[0])
    good = np.isfinite(values)
    if good.sum() == 0:
        return values
    return np.interp(x, x[good], values[good])


def main() -> None:
    ap = argparse.ArgumentParser(description="DA3 metric object depth, scaled to GVHMR.")
    ap.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    ap.add_argument("--model-id", type=str, default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument(
        "--body-model-root",
        type=Path,
        default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"),
    )
    ap.add_argument("--dump-maps", action="store_true", help="also save the scaled depth maps (float16 .npy)")
    ap.add_argument("--max-frames", type=int, default=0, help="only do the first N frames (debugging)")
    args = ap.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = results_dir / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = results_dir / "segmentation" / "masks"

    frame_paths = sorted((sample_dir / "frames").glob("*.png"))
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError(f"No frames in {sample_dir/'frames'}")
    frame_ids = [int(p.stem) for p in frame_paths]

    gvhmr = load_gvhmr(results_dir)
    K = gvhmr["K_fullimg"]
    n_smplx = gvhmr["transl"].shape[0]
    joints = build_smplx_joints(args.body_model_root, gvhmr)
    obs = read_object_observations(results_dir / "object_observations" / "object_observations.csv")

    # fps isn't fixed across samples, so take the timestamps from the obs table when we can
    times = np.array(
        [obs[fid]["time"] if fid in obs else (fid - frame_ids[0]) / 30.0 for fid in frame_ids],
        dtype=np.float64,
    )

    print(f"Loading {args.model_id} ...")
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(args.model_id).to("cuda").eval()

    if args.dump_maps:
        (out_dir / "maps").mkdir(exist_ok=True)

    n_f = len(frame_paths)
    a_t = np.full(n_f, np.nan)
    b_t = np.full(n_f, np.nan)
    corr_t = np.full(n_f, np.nan)   # per-frame corr(DA3, trueZ) on the body joints
    dlo_t = np.full(n_f, np.nan)    # joint DA3 range (to forbid object extrapolation)
    dhi_t = np.full(n_f, np.nan)
    obj_raw = np.full(n_f, np.nan)
    obj_n = np.zeros(n_f, dtype=int)
    obj_spread = np.full(n_f, np.nan)
    obj_src = ["none"] * n_f
    raw_maps: list[np.ndarray | None] = [None] * n_f
    glob_d: list[float] = []
    glob_z: list[float] = []
    frame_size = (0, 0)

    # One pass over the frames: run DA3, fit this frame's scale off the body joints,
    # grab the object's depth. The per-frame fit matters - DA3 runs each frame on its
    # own and the absolute scale drifts, so a single global fit flattens the motion.
    for i, (fp, fid) in enumerate(zip(frame_paths, frame_ids)):
        img = cv2.imread(str(fp))
        H, W = img.shape[:2]
        frame_size = (H, W)
        with torch.inference_mode():
            pred = model.inference([str(fp)])
        dmap = np.asarray(pred.depth)[0].astype(np.float32)  # comes back at model res
        dmap = cv2.resize(dmap, (W, H), interpolation=cv2.INTER_LINEAR)
        if args.dump_maps:
            raw_maps[i] = dmap

        # smpl arrays are 0-based, the frame files are 1-based
        j_idx = fid - frame_ids[0]
        if 0 <= j_idx < n_smplx:
            jc = joints[j_idx]
            uv = project(jc, K[j_idx])
            uu = np.round(uv[:, 0]).astype(int)
            vv = np.round(uv[:, 1]).astype(int)
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (jc[:, 2] > 1e-3)
            if inb.any():
                # patch-median per joint (robust to a pixel landing just off the body), then
                # drop joints whose DA3 is a far-background outlier vs the body median.
                dj = np.array([sample_patch_median(dmap, int(u), int(v))
                               for u, v in zip(uu[inb], vv[inb])], dtype=np.float64)
                zj = jc[inb, 2].astype(np.float64)
                dj, zj = reject_background_joints(dj, zj)
                glob_d.extend(dj.tolist())
                glob_z.extend(zj.tolist())
                fit = fit_frame_affine(dj, zj)
                if fit is not None:
                    a_t[i], b_t[i], _, corr_t[i], dlo_t[i], dhi_t[i] = fit

        obj_raw[i], obj_n[i], obj_spread[i], obj_src[i] = sample_object_raw_depth(
            dmap, fid, masks_dir, obs
        )
        if (i + 1) % 40 == 0 or i == n_f - 1:
            print(f"  DA3 {i+1}/{n_f} frames, {int(np.isfinite(a_t).sum())} per-frame fits")

    n_fit = int(np.isfinite(a_t).sum())
    if n_fit < 1:
        raise RuntimeError("No per-frame affine fits succeeded (human joints never in frame?)")

    # patch the frames that didn't fit, then take the jitter out of the scale curve
    a_fill = interpolate_fill(a_t)
    b_fill = interpolate_fill(b_t)
    try:
        from scipy.ndimage import gaussian_filter1d
        a_s = gaussian_filter1d(a_fill, sigma=2.0, mode="nearest")
        b_s = gaussian_filter1d(b_fill, sigma=2.0, mode="nearest")
    except Exception:
        a_s, b_s = a_fill, b_fill

    # forbid the object from extrapolating beyond the body-joint DA3 range: clamp its raw value
    # to [dlo, dhi] (interp/smoothed), and record how far out of range it was.
    dlo_s = interpolate_fill(dlo_t)
    dhi_s = interpolate_fill(dhi_t)
    corr_fill = interpolate_fill(corr_t)
    span = np.maximum(dhi_s - dlo_s, 1e-6)
    extrap_t = np.maximum(0.0, np.maximum(dlo_s - obj_raw, obj_raw - dhi_s)) / span
    obj_clamped = np.clip(obj_raw, dlo_s, dhi_s)
    obj_aligned = a_s * obj_clamped + b_s

    # also fit one global line, only so we can print it / keep it around for sanity
    gd = np.asarray(glob_d); gz = np.asarray(glob_z)
    gvalid = np.isfinite(gd) & np.isfinite(gz) & (gd > 0)
    g_a, g_b, g_rms, g_in = robust_affine_fit(gd[gvalid], gz[gvalid]) if gvalid.sum() >= 3 else (float("nan"),) * 4
    med_slope = float(np.nanmedian(a_t))
    print(f"Per-frame affine: {n_fit}/{n_f} frames fit, median slope {med_slope:.4f}")
    print(f"Global affine (diagnostic): Z = {g_a:.4f} * D + {g_b:.4f}  (rms {g_rms:.4f} m)")

    rows: list[dict[str, object]] = []
    for i, (fid, t) in enumerate(zip(frame_ids, times)):
        # confidence now reflects FIT QUALITY, not just mask spread:
        #   spread_term  - tight object depth sample
        #   corr_q       - DA3 monotonic with true depth on the body (0 if anti-correlated)
        #   extrap_pen   - object DA3 inside the joint range (penalise extrapolation)
        spread_term = float(1.0 / (1.0 + obj_spread[i])) if np.isfinite(obj_spread[i]) else 0.0
        corr_q = float(max(0.0, corr_fill[i])) if np.isfinite(corr_fill[i]) else 0.0
        extrap_pen = float(1.0 / (1.0 + extrap_t[i])) if np.isfinite(extrap_t[i]) else 0.0
        conf = spread_term * corr_q * extrap_pen
        if args.dump_maps and raw_maps[i] is not None:
            np.save(out_dir / "maps" / f"{fid:05d}.npy", (a_s[i] * raw_maps[i] + b_s[i]).astype(np.float16))
        rows.append({
            "frame": fid,
            "time": f"{t:.6f}",
            "object_z_raw": f"{obj_raw[i]:.6f}",
            "object_z_aligned_m": f"{obj_aligned[i]:.6f}",
            "affine_a": f"{a_s[i]:.6f}",
            "affine_b": f"{b_s[i]:.6f}",
            "sample_count": int(obj_n[i]),
            "depth_conf": f"{conf:.6f}",
            "fit_corr": f"{corr_fill[i]:.4f}" if np.isfinite(corr_fill[i]) else "",
            "extrapolation": f"{extrap_t[i]:.4f}" if np.isfinite(extrap_t[i]) else "",
            "source": obj_src[i],
        })

    csv_path = out_dir / "object_depth.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time", "object_z_raw", "object_z_aligned_m",
                                          "affine_a", "affine_b", "sample_count", "depth_conf",
                                          "fit_corr", "extrapolation", "source"])
        w.writeheader()
        w.writerows(rows)

    with (out_dir / "depth_alignment.json").open("w") as f:
        json.dump({
            "mode": "per_frame_affine_smoothed_gated",
            "median_slope": med_slope,
            "median_corr": float(np.nanmedian(corr_t)) if np.isfinite(corr_t).any() else None,
            "median_extrapolation": float(np.nanmedian(extrap_t)) if np.isfinite(extrap_t).any() else None,
            "n_frames_fit": n_fit,
            "n_frames": n_f,
            "smoothing_sigma_frames": 2.0,
            "global_affine_diagnostic": {"a": g_a, "b": g_b, "fit_residual_m": g_rms},
            "model_id": args.model_id,
            "num_body_joints": NUM_BODY_JOINTS,
            "frame_size": [int(frame_size[0]), int(frame_size[1])],
        }, f, indent=2)

    z_plot = obj_aligned
    fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
    ax.plot(frame_ids, z_plot, color="#4c72b0", label="object Z (per-frame aligned, m)")
    ax.set_xlabel("frame")
    ax.set_ylabel("Z_cam (m)")
    ax.set_title("DA3 object depth aligned to GVHMR (per-frame affine)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "object_depth_preview.png")
    plt.close(fig)

    n_obj = sum(1 for s in obj_src if s != "none")
    print(f"object_depth_csv: {csv_path}")
    print(f"alignment_json: {out_dir/'depth_alignment.json'}")
    print(f"preview_png: {out_dir/'object_depth_preview.png'}")
    print(f"frames_with_object_depth: {n_obj}/{len(rows)}")
    finite_z = z_plot[np.isfinite(z_plot)]
    if finite_z.size:
        print(f"object_z_aligned_m  min/med/max: {finite_z.min():.3f} / "
              f"{np.median(finite_z):.3f} / {finite_z.max():.3f}")


if __name__ == "__main__":
    main()
