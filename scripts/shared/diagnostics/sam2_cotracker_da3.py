"""Compare the object-localisation sources: SAM2 mask centre vs CoTracker centre (2D agreement)
and DA3 depth (the metric Z cue). One figure: do the 2D trackers agree, and how does depth behave.

Output: <out-png>
Run: python scripts/shared/diagnostics/sam2_cotracker_da3.py --sample-dir samples/basketball_01 --out foo.png
"""
from __future__ import annotations

import argparse
import csv
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


def read(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def masks_dir(s):
    for c in ("masks", "segmentation/masks"):
        d = s / "results" / c
        if d.exists() and any(d.glob("*_mask.png")):
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    s = args.sample_dir
    n = len(sorted((s / "frames").glob("*.png")))
    fr = np.arange(1, n + 1)

    # SAM2 mask center
    md = masks_dir(s)
    mcx = np.full(n, np.nan); mcy = np.full(n, np.nan)
    if md:
        for i in range(n):
            mp = md / f"{i+1:05d}_mask.png"
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
            if m is not None and (m > 127).any():
                ys, xs = np.where(m > 127); mcx[i], mcy[i] = xs.mean(), ys.mean()

    # CoTracker center
    ct = read(s / "results" / "tracking" / "cotracker_center_trajectory.csv")
    ccx = np.full(n, np.nan); ccy = np.full(n, np.nan)
    cm = {int(_f(r["frame"])): r for r in ct}
    for i, f in enumerate(fr):
        if f in cm:
            ccx[i] = _f(cm[f].get("ball_center_x")); ccy[i] = _f(cm[f].get("ball_center_y"))

    # DA3 depth
    dp = read(s / "results" / "depth" / "object_depth.csv")
    dm = {int(_f(r["frame"])): r for r in dp}
    z = np.array([_f(dm.get(f, {}).get("object_z_aligned_m", np.nan)) for f in fr])
    conf = np.array([_f(dm.get(f, {}).get("depth_conf", np.nan)) for f in fr])

    disagree = np.hypot(mcx - ccx, mcy - ccy)

    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(fr, mcx, color="tab:blue", lw=1.6, label="SAM2 mask center_x")
    ax[0].plot(fr, ccx, color="tab:orange", lw=1.4, ls="--", label="CoTracker center_x")
    ax[0].set_ylabel("x (px)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(fr, mcy, color="tab:blue", lw=1.6, label="SAM2 mask center_y")
    ax[1].plot(fr, ccy, color="tab:orange", lw=1.4, ls="--", label="CoTracker center_y")
    ax[1].set_ylabel("y (px)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[2].plot(fr, disagree, color="tab:red", lw=1.4)
    ax[2].set_ylabel("SAM2↔CoTracker\ndisagree (px)")
    ax[2].set_title(f"mean 2D disagreement = {np.nanmean(disagree):.1f} px", fontsize=9, loc="left")
    ax[2].grid(alpha=0.3)
    ax[3].plot(fr, z, color="tab:green", lw=1.6, label="DA3 depth Z (m)")
    ax3b = ax[3].twinx()
    ax3b.plot(fr, conf, color="gray", lw=1.0, alpha=0.6, label="depth_conf")
    ax3b.set_ylabel("conf", color="gray"); ax3b.set_ylim(0, 1)
    ax[3].set_ylabel("Z (m)", color="tab:green"); ax[3].set_xlabel("frame")
    ax[3].legend(loc="upper left", fontsize=8); ax[3].grid(alpha=0.3)
    fig.suptitle(f"SAM2 mask vs CoTracker (2D) + DA3 depth — {s.name}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print("wrote", args.out, f"| 2D disagree {np.nanmean(disagree):.1f}px | depth conf {np.nanmean(conf):.2f}")


if __name__ == "__main__":
    main()
