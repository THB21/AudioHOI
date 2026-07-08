"""One graph over ALL sources: per-frame reliability/instability of each input source.

Each source contributes one curve (its per-frame "cost", normalised to [0,1] by its own 95th
percentile, so higher = that source is less trustworthy on that frame). Lets you compare all
sources against each other on a single timeline. Audio onsets are drawn as vertical markers.

Per-source cost:
  SAM2 mask     center jitter (px/frame)
  CoTracker     disagreement with mask centre (px)
  DA3 depth     1 - depth_conf
  HaMeR hands   palm 3D jitter (m/frame)
  GVHMR body    translation jitter (accel)
  Object 3D     center reprojection residual (px) = E_2d

Output: results/diagnostics/source_health_graph.png
Run:    python scripts/shared/diagnostics/source_health_graph.py --sample-dir samples/basketball_01
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _f(v):
    try:
        return float(v) if v not in (None, "", "nan") else np.nan
    except (ValueError, TypeError):
        return np.nan


def read(p: Path):
    return list(csv.DictReader(open(p))) if p.exists() else []


def norm(y):
    y = np.asarray(y, float)
    p = np.nanpercentile(np.abs(y), 95) if np.isfinite(y).any() else 1.0
    return np.abs(y) / (p + 1e-9)


def masks_dir(s):
    for c in ("masks", "segmentation/masks"):
        d = s / "results" / c
        if d.exists() and any(d.glob("*_mask.png")):
            return d
    return None


def mask_centers(s, n):
    md = masks_dir(s)
    cx = np.full(n, np.nan); cy = np.full(n, np.nan)
    if not md:
        return cx, cy
    for i in range(n):
        mp = md / f"{i+1:05d}_mask.png"
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
        if m is not None and (m > 127).any():
            ys, xs = np.where(m > 127)
            cx[i], cy[i] = xs.mean(), ys.mean()
    return cx, cy


def vel(x):
    v = np.zeros_like(x); v[1:] = np.abs(np.diff(x))
    return np.nan_to_num(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    args = ap.parse_args()
    s = args.sample_dir
    R = s / "results"
    n = len(sorted((s / "frames").glob("*.png")))
    frames = np.arange(1, n + 1)
    curves = {}

    # SAM2 mask: center jitter
    cx, cy = mask_centers(s, n)
    if np.isfinite(cx).any():
        curves["SAM2 mask (center jitter)"] = norm(np.hypot(vel(cx), vel(cy)))

    # CoTracker: disagreement vs mask center
    ct = read(R / "tracking" / "cotracker_center_trajectory.csv")
    if ct:
        m = {int(_f(r["frame"])): (_f(r.get("ball_center_x")), _f(r.get("ball_center_y"))) for r in ct}
        dis = np.array([np.hypot(m.get(f, (np.nan, np.nan))[0] - cx[f-1],
                                 m.get(f, (np.nan, np.nan))[1] - cy[f-1]) if f in m else np.nan
                        for f in frames])
        curves["CoTracker (vs mask)"] = norm(dis)

    # DA3 depth: 1 - conf
    dp = read(R / "depth" / "object_depth.csv")
    if dp:
        conf = {int(_f(r["frame"])): _f(r.get("depth_conf")) for r in dp}
        c = np.array([conf.get(f, 0.0) for f in frames])
        curves["DA3 depth (1-conf)"] = np.clip(1 - np.nan_to_num(c), 0, 1)

    # HaMeR: palm 3D jitter
    hk = read(R / "hands" / "hand_keypoints_3d.csv")
    if hk:
        by = {int(_f(r["frame"])): r for r in hk}
        pj = np.full(n, np.nan)
        prev = None
        for i, f in enumerate(frames):
            r = by.get(f)
            if r:
                p = np.array([_f(r.get("right_palm_x")), _f(r.get("right_palm_y")), _f(r.get("right_palm_z"))])
                if not np.any(np.isnan(p)) and prev is not None:
                    pj[i] = np.linalg.norm(p - prev)
                if not np.any(np.isnan(p)):
                    prev = p
        if np.isfinite(pj).any():
            curves["HaMeR (palm jitter)"] = norm(pj)

    # GVHMR: translation accel
    gp = R / "gvhmr" / "result.pkl"
    if gp.exists():
        d = pickle.load(open(gp, "rb")).get("smpl_params_incam", {})
        if "transl" in d:
            tr = np.asarray(d["transl"])
            acc = np.zeros(len(tr)); acc[1:-1] = np.linalg.norm(tr[2:] - 2*tr[1:-1] + tr[:-2], axis=1)
            acc = np.pad(acc, (0, max(0, n - len(acc))))[:n]
            curves["GVHMR (transl jitter)"] = norm(acc)

    # Object: reprojection residual (E_2d)
    for rel in ("pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv",):
        p = R / rel
        if p.exists():
            rows = read(p)
            if "residual_px" in rows[0]:
                rmap = {int(_f(r["frame"])): _f(r.get("residual_px")) for r in rows}
                curves["Object (E_2d residual)"] = norm([rmap.get(f, 0.0) for f in frames])
            break

    # audio markers
    ae = read(R / "events" / "audio_events.csv")
    af = [int(_f(r.get("audio_frame", r.get("frame")))) for r in ae
          if not np.isnan(_f(r.get("audio_frame", r.get("frame"))))]

    fig, ax = plt.subplots(figsize=(14, 6))
    for name, y in curves.items():
        ax.plot(frames, y, lw=1.6, label=name, alpha=0.85)
    for f in af:
        ax.axvline(f, color="magenta", alpha=0.15, lw=1)
    if af:
        ax.plot([], [], color="magenta", alpha=0.5, label="audio onset")
    ax.set_xlabel("frame")
    ax.set_ylabel("per-source cost  (norm. to own 95th pct; higher = less reliable)")
    ax.set_title(f"All sources on one timeline — {s.name}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.6); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9, ncol=2)
    out = R / "diagnostics" / "source_health_graph.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out, "| sources:", list(curves))


if __name__ == "__main__":
    main()
