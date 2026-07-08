"""Measure the 3D gap between the object and the body part it is supposed to contact.

The user observes the ball sits far from the corresponding foot in the 3D scene, even at
audio/visual contact frames. The contact-phase anchor only pins object DEPTH (z) to the part;
X/Y come from back-projecting the 2D centre, so the full 3D distance can still be large, and
between the sparse contacts the object drifts. This quantifies it.

For each frame: distance from object (tx,ty,tz) to each candidate part (GVHMR ankles/feet and
HaMeR palms), the nearest part, and whether the frame is a detected contact (visual/audio).

Output: results/diagnostics/contact_gap/foot_ball_gap.png + .csv
Run: python scripts/shared/diagnostics/foot_ball_contact_gap.py --sample-dir samples/football_10
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("da3", HERE.parent / "depth" / "run_depth_anything_v3.py")
da3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(da3)

# SMPL-X body joint indices
PARTS = {"L_ankle": 7, "R_ankle": 8, "L_foot": 10, "R_foot": 11,
         "L_knee": 4, "R_knee": 5, "L_wrist": 20, "R_wrist": 21}


def _f(v):
    try:
        return float(v) if v not in (None, "", "nan") else np.nan
    except (ValueError, TypeError):
        return np.nan


def read(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def pick_traj(s):
    for rel in ("pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv",
                "pose6d_sharedcam_depthv3/ball_pose6d_sharedcam_trajectory.csv"):
        p = s / "results" / rel
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    args = ap.parse_args()
    s = args.sample_dir
    tp = pick_traj(s)
    rows = read(tp)
    fr = np.array([int(_f(r["frame"])) for r in rows])
    obj = np.array([[_f(r["tx"]), _f(r["ty"]), _f(r["tz"])] for r in rows])

    gvhmr = da3.load_gvhmr(s / "results")
    joints = da3.build_smplx_joints(args.body_model_root, gvhmr)  # [N,22,3]
    f0 = fr[0]

    # contact flags
    cf = np.array([_f(r.get("contact_frame")) for r in rows]) >= 1
    af = np.array([_f(r.get("audio_contact_frame")) for r in rows]) >= 1

    # per-frame distance to each part + nearest
    dists = {p: np.full(len(fr), np.nan) for p in PARTS}
    nearest_d = np.full(len(fr), np.nan)
    nearest_name = [""] * len(fr)
    for i, f in enumerate(fr):
        j = f - f0
        if j < 0 or j >= len(joints) or np.any(np.isnan(obj[i])):
            continue
        best = None
        for name, idx in PARTS.items():
            d = float(np.linalg.norm(obj[i] - joints[j, idx]))
            dists[name][i] = d
            if best is None or d < best[0]:
                best = (d, name)
        nearest_d[i] = best[0]
        nearest_name[i] = best[1]

    # which part is closest at contact frames (the part it SHOULD be touching)
    contact = cf | af
    foot_d = np.nanmin(np.vstack([dists["L_foot"], dists["R_foot"],
                                  dists["L_ankle"], dists["R_ankle"]]), axis=0)

    out = s / "results" / "diagnostics" / "contact_gap"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(fr, nearest_d, color="tab:blue", lw=1.6, label="dist to NEAREST body part")
    ax.plot(fr, foot_d, color="tab:green", lw=1.4, alpha=0.8, label="dist to nearest FOOT/ankle")
    for f in fr[contact]:
        ax.axvline(f, color="red", alpha=0.25)
    ax.plot([], [], color="red", alpha=0.5, label="detected contact (visual/audio)")
    ax.axhline(0.12, color="gray", ls="--", lw=1, label="~ball radius (0.12 m)")
    ax.set_xlabel("frame"); ax.set_ylabel("3D distance object -> body part (m)")
    ax.set_title(f"Object vs contacting body part — {s.name}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "foot_ball_gap.png", dpi=120); plt.close(fig)

    with open(out / "foot_ball_gap.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["frame", "contact", "nearest_part", "nearest_dist_m", "foot_dist_m"])
        for i, f in enumerate(fr):
            w.writerow([f, int(contact[i]), nearest_name[i],
                        f"{nearest_d[i]:.4f}", f"{foot_d[i]:.4f}"])

    print(f"== {s.name} ==")
    print(f"  detected contact frames: {int(contact.sum())}")
    if contact.any():
        print(f"  at contact: nearest-part dist mean={np.nanmean(nearest_d[contact]):.3f} max={np.nanmax(nearest_d[contact]):.3f} m")
        print(f"  at contact: FOOT/ankle dist  mean={np.nanmean(foot_d[contact]):.3f} max={np.nanmax(foot_d[contact]):.3f} m")
        from collections import Counter
        print("  nearest part at contact frames:", dict(Counter(np.array(nearest_name)[contact])))
    print(f"  overall: nearest-part dist mean={np.nanmean(nearest_d):.3f} max={np.nanmax(nearest_d):.3f} m")
    print("  wrote", out / "foot_ball_gap.png")


if __name__ == "__main__":
    main()
