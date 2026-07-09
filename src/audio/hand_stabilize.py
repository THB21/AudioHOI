"""Stabilize hand articulation during a grasp (occlusion-robust).

HaMeR re-estimates the finger pose every frame. While an object is held, the hand is often
occluded behind it, so HaMeR's per-frame estimate jitters (~7-9 deg) even though a stable
grasp should keep the finger configuration constant. This is the hand-articulation analog of
``grasp_attach`` (which stabilizes object *position*): over a confirmed grasp interval we
replace the jittery per-frame finger pose of the grasping hand with a single stable grasp
pose (robust median over the most confident frames of the interval), then blend back out at
the interval edges so non-grasp motion is preserved.

Grasp intervals + the grasping side come from the contact records (persistence state machine).
Writes a stabilized ``stitched_smplx_params.pkl`` the renderer picks up automatically.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def _intervals(records_csv: Path):
    with Path(records_csv).open() as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("contact_state") in ("direct_contact", "keep_grasp")]
    rows.sort(key=lambda r: int(r["frame"]))
    out, cur = [], []
    for r in rows:
        if cur and int(r["frame"]) - int(cur[-1]["frame"]) > 30:
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    intervals = []
    for iv in out:
        sides = [r.get("target_entity") or r.get("vlm_hand_side") for r in iv]
        side = next((s for s in sides if s in ("left_hand", "right_hand")), "left_hand")
        intervals.append((int(iv[0]["frame"]), int(iv[-1]["frame"]), side))
    return intervals


def stabilize(stitched_pkl: Path, records_csv: Path, out_pkl: Path | None = None,
              edge_blend: int = 4):
    d = pickle.load(open(stitched_pkl, "rb"))
    intervals = _intervals(records_csv)
    if not intervals:
        print("no grasp intervals"); return

    report = []
    for start, end, side in intervals:
        key = "left_hand_pose" if side == "left_hand" else "right_hand_pose"
        if key not in d:
            continue
        pose = np.asarray(d[key]).copy()           # [N,45]
        lo, hi = start - 1, min(end - 1, pose.shape[0] - 1)   # 0-indexed
        if hi <= lo:
            continue
        seg = pose[lo:hi + 1]
        before = float(np.mean(np.std(seg, axis=0)))
        # robust stable grasp pose = per-DoF median over the interval
        stable = np.median(seg, axis=0)
        new = pose.copy()
        for j in range(lo, hi + 1):
            # blend toward the stable pose, easing in/out at the interval edges
            de = min(j - lo, hi - j)
            w = min(1.0, (de + 1) / max(1, edge_blend))
            new[j] = (1 - w) * pose[j] + w * stable
        after = float(np.mean(np.std(new[lo:hi + 1], axis=0)))
        d[key] = new
        report.append((start, end, side, before, after))

    out_pkl = out_pkl or stitched_pkl.with_name("stitched_smplx_params_stable.pkl")
    pickle.dump(d, open(out_pkl, "wb"))
    for s, e, side, b, a in report:
        print(f"  grasp {s}-{e} [{side}]: finger jitter std {np.degrees(b):.1f}° → {np.degrees(a):.1f}°")
    print(f"stabilized hand params: {out_pkl}")
    return out_pkl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--records-csv", type=Path, default=None)
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite stitched_smplx_params.pkl (renderer auto-uses it)")
    args = ap.parse_args()
    res = args.sample_dir / "results"
    stitched = res / "hands" / "stitched_smplx_params.pkl"
    records = args.records_csv or (res / "audio_semantics" / "contact_records.csv")
    out = stitched if args.in_place else None
    stabilize(stitched, records, out)


if __name__ == "__main__":
    main()
