"""Contact-point-constrained grasp: keep the grasping hand ON the object surface.

Principled alternative to object-removal (which only hallucinates a hand). At grasp frames the
closest fingertip/palm should lie on the known object surface: ||hand_pt - center|| = radius.
Where the recovered hand floats off (or penetrates), we apply a small RADIAL translation to the
grasping hand so the contact point sits on the surface, then temporally smooth it. This is
`R_contact` for the hand — real geometry, no generated pixels.

A frame is treated as a grasp only if the hand is already within `--grasp-max` of the object
centre (so a genuine release is left alone). Body pose / finger configuration are preserved;
only the hand's global position is nudged.

Inputs (camera frame, metres): hand_keypoints_3d.csv + an object pose CSV (tx,ty,tz,radius_m).
Outputs under results/hands/grasp_contact/: corrected keypoints, wrist deltas, before/after plot,
report.

Run:
  python scripts/shared/hands/constrain_grasp_to_contact.py --sample-dir samples_known_object/02_mug_v2 \
    --object-csv samples_known_object/02_mug_v2/results/mug_oriented_pose.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIPS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip", "palm"]
SIDES = ["right", "left"]


def _f(v):
    try:
        return float(v) if v not in (None, "", "nan") else np.nan
    except ValueError:
        return np.nan


def read(path: Path):
    with open(path) as f:
        return list(csv.DictReader(f))


def gaussian_smooth_vec(vecs: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth an (N,3) correction over time, only mixing within masked (grasp) frames."""
    if sigma <= 0:
        return vecs
    n = len(vecs)
    out = np.zeros_like(vecs)
    half = int(3 * sigma)
    for i in range(n):
        if not mask[i]:
            continue
        w_sum = 0.0
        acc = np.zeros(3)
        for j in range(max(0, i - half), min(n, i + half + 1)):
            if not mask[j]:
                continue
            w = np.exp(-0.5 * ((i - j) / sigma) ** 2)
            acc += w * vecs[j]
            w_sum += w
        out[i] = acc / w_sum if w_sum > 0 else vecs[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--object-csv", type=Path, required=True)
    ap.add_argument("--hand-csv", type=Path, default=None)
    ap.add_argument("--grasp-max", type=float, default=0.12, help="m: beyond this from centre = released, skip")
    ap.add_argument("--tol", type=float, default=0.01, help="m: leave gaps smaller than this alone")
    ap.add_argument("--smooth-sigma", type=float, default=2.0)
    args = ap.parse_args()

    sample = args.sample_dir
    hand_csv = args.hand_csv or (sample / "results" / "hands" / "hand_keypoints_3d.csv")
    obj = {int(float(r["frame"])): r for r in read(args.object_csv)}
    hand_rows = read(hand_csv)

    frames, gap_before, gap_after, corr = [], [], [], []
    grasp_mask = []
    out_rows = []
    for h in hand_rows:
        f = int(float(h["frame"]))
        new = dict(h)
        if f not in obj:
            out_rows.append(new); continue
        tr = obj[f]
        C = np.array([_f(tr.get("tx")), _f(tr.get("ty")), _f(tr.get("tz"))])
        R = _f(tr.get("radius_m"))
        if np.any(np.isnan(C)) or np.isnan(R):
            out_rows.append(new); continue
        # closest hand point to the object centre
        best = None
        for side in SIDES:
            for t in TIPS:
                p = np.array([_f(h.get(f"{side}_{t}_x")), _f(h.get(f"{side}_{t}_y")), _f(h.get(f"{side}_{t}_z"))])
                if np.any(np.isnan(p)):
                    continue
                d = np.linalg.norm(p - C)
                if best is None or d < best[0]:
                    best = (d, side, p)
        if best is None:
            out_rows.append(new); continue
        d, side, p = best
        is_grasp = d < args.grasp_max
        frames.append(f)
        gap_before.append(d - R)
        grasp_mask.append(is_grasp)
        # radial correction so the closest point lands on the surface
        if is_grasp and abs(d - R) > args.tol and d > 1e-6:
            delta = (R - d) * (p - C) / d   # move along centre->point ray
        else:
            delta = np.zeros(3)
        corr.append(delta)
        out_rows.append(new)

    frames = np.array(frames)
    corr = np.array(corr)
    grasp_mask = np.array(grasp_mask)
    corr_s = gaussian_smooth_vec(corr, grasp_mask, args.smooth_sigma)

    # apply smoothed correction to that side's keypoints, recompute gap
    fi = {f: i for i, f in enumerate(frames)}
    for new in out_rows:
        f = int(float(new["frame"]))
        if f not in fi:
            continue
        i = fi[f]
        if f not in obj:
            continue
        tr = obj[f]
        C = np.array([_f(tr.get("tx")), _f(tr.get("ty")), _f(tr.get("tz"))]); R = _f(tr.get("radius_m"))
        d = np.linalg.norm(corr_s[i])
        # apply to BOTH sides' points equally is wrong; apply to the grasping side only.
        # recompute which side is closest, shift it.
        best = None
        for side in SIDES:
            p = np.array([_f(new.get(f"{side}_palm_x")), _f(new.get(f"{side}_palm_y")), _f(new.get(f"{side}_palm_z"))])
            if np.any(np.isnan(p)):
                continue
            dd = np.linalg.norm(p - C)
            if best is None or dd < best[0]:
                best = (dd, side)
        if best is None:
            continue
        side = best[1]
        for t in TIPS + ["wrist"]:
            for a, comp in enumerate(("x", "y", "z")):
                k = f"{side}_{t}_{comp}"
                if k in new and new[k] not in (None, "", "nan"):
                    new[k] = f"{_f(new[k]) + corr_s[i][a]:.6f}"
        # recompute closest-point gap after correction
        bestd = None
        for s2 in SIDES:
            for t in TIPS:
                p = np.array([_f(new.get(f"{s2}_{t}_x")), _f(new.get(f"{s2}_{t}_y")), _f(new.get(f"{s2}_{t}_z"))])
                if np.any(np.isnan(p)):
                    continue
                dd = np.linalg.norm(p - C)
                if bestd is None or dd < bestd:
                    bestd = dd
        gap_after.append((f, (bestd - R) if bestd is not None else np.nan))

    gb = np.array(gap_before)
    ga = np.array([g for _, g in gap_after])
    out_dir = sample / "results" / "hands" / "grasp_contact"
    out_dir.mkdir(parents=True, exist_ok=True)

    # write corrected keypoints
    with open(out_dir / "hand_keypoints_3d_grasp.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # before/after plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(frames, gb, color="tab:red", alpha=0.7, label="before (surface gap)")
    ax.plot(frames, ga, color="tab:green", lw=1.5, label="after")
    ax.fill_between(frames, -1, 1, where=grasp_mask, color="tab:blue", alpha=0.05, label="grasp frames")
    ax.set_ylim(min(gb.min(), -0.05) - 0.02, gb.max() + 0.02)
    ax.set_xlabel("frame"); ax.set_ylabel("closest-hand-pt gap to surface (m)")
    ax.set_title(f"{sample.name}: grasp-contact gap (0 = fingertip on object surface)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "grasp_gap_before_after.png", dpi=110); plt.close(fig)

    grasp = grasp_mask
    rep = [
        f"# Grasp-contact constraint — {sample.name}", "",
        f"- object: `{args.object_csv.name}`  | frames: {len(frames)}  | grasp frames (within {args.grasp_max} m): {int(grasp.sum())}",
        "",
        "| metric | before | after |",
        "|---|---|---|",
        f"| mean \\|gap\\| on grasp frames (m) | {np.abs(gb[grasp]).mean():.4f} | {np.abs(ga[grasp]).mean():.4f} |",
        f"| max \\|gap\\| on grasp frames (m) | {np.abs(gb[grasp]).max():.4f} | {np.abs(ga[grasp]).max():.4f} |",
        f"| grasp frames floating >5cm | {int((np.abs(gb[grasp])>0.05).sum())} | {int((np.abs(ga[grasp])>0.05).sum())} |",
        "",
        "0 gap = closest fingertip/palm exactly on the object surface. Released frames (hand far "
        f"from the object, >{args.grasp_max} m) are left untouched. Correction is a smoothed radial "
        "translation of the grasping hand; finger configuration is preserved.",
        "",
        "Outputs: `hand_keypoints_3d_grasp.csv`, `grasp_gap_before_after.png`.",
    ]
    (out_dir / "grasp_report.md").write_text("\n".join(rep))
    print(f"{sample.name}: grasp frames={int(grasp.sum())}/{len(frames)}")
    print(f"  mean |gap| {np.abs(gb[grasp]).mean():.4f} -> {np.abs(ga[grasp]).mean():.4f} m")
    print(f"  max  |gap| {np.abs(gb[grasp]).max():.4f} -> {np.abs(ga[grasp]).max():.4f} m")
    print(f"  floating>5cm {int((np.abs(gb[grasp])>0.05).sum())} -> {int((np.abs(ga[grasp])>0.05).sum())}")
    print("wrote", out_dir / "grasp_report.md")


if __name__ == "__main__":
    main()
