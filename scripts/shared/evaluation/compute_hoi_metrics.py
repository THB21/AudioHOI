#!/usr/bin/env python3
"""Reference-free HOI reconstruction metrics (in-the-wild protocol, see docs/loop_plan.md §6).

Computes, for one object trajectory CSV:
  jerk            mean ||third finite difference of T_t||^2 (m^2) — smoothness (co-report with 2D fit)
  contact_rate    fraction of expected-contact frames (contact events/states) with
                  |tz - active_part_z| < thresh  (only for CSVs that carry contact columns)
  contact_gap_cm  mean |tz - active_part_z| at contact-event frames
  pen_frames      frames where the object center is closer than radius to any body part
                  center / body joint (needs GVHMR + smplx)
  audio_err_ms    median |trajectory contact onset - audio onset|; onsets recovered from
                  depth-velocity kinks (arg-local-max of |dv/dt|) matched to audio_events.csv
                  NOTE: for solver outputs this is partly self-referential (audio anchors are
                  constraints) — meaningful mainly for the no-audio ablation and baselines.
  tz_range        min/max depth (sanity)

Usage:
  conda run -n gvhmr python scripts/shared/evaluation/compute_hoi_metrics.py \
    --sample-dir samples/football_10 --trajectory-csv <csv> [--label name]
Append rows to a shared table with --out-csv (default results/evaluation/metrics.csv).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "human_ball" / "contact"))


def read_traj(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--trajectory-csv", type=Path, required=True)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--contact-thresh-m", type=float, default=0.05)
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--skip-penetration", action="store_true")
    args = ap.parse_args()

    rows = read_traj(args.trajectory_csv)
    T = np.array([[float(r["tx"]), float(r["ty"]), float(r["tz"])] for r in rows])
    times = np.array([float(r["time"]) for r in rows])
    tz = T[:, 2]

    out: dict[str, object] = {
        "sample": args.sample_dir.name,
        "label": args.label or args.trajectory_csv.parent.name,
        "n_frames": len(rows),
        "tz_min": round(float(tz.min()), 3),
        "tz_max": round(float(tz.max()), 3),
    }

    # jerk (third difference), per-frame mean
    if len(T) >= 4:
        jerk = T[3:] - 3 * T[2:-1] + 3 * T[1:-2] - T[:-3]
        out["jerk"] = round(float((jerk ** 2).sum(axis=1).mean()), 6)

    # contact metrics (only if the CSV carries them)
    if "active_part_z" in rows[0] and "contact_frame" in rows[0]:
        ev = [i for i, r in enumerate(rows)
              if int(r.get("contact_frame", 0) or 0) == 1 or int(r.get("audio_anchor", 0) or 0) == 1]
        if ev:
            gaps = np.array([abs(float(rows[i]["tz"]) - float(rows[i]["active_part_z"])) for i in ev])
            out["contact_rate"] = round(float((gaps < args.contact_thresh_m).mean()), 3)
            out["contact_gap_cm"] = round(float(gaps.mean() * 100), 2)

    # audio-contact timing error: depth-velocity kink times vs audio onsets
    audio_csv = args.sample_dir / "results" / "events" / "audio_events.csv"
    if audio_csv.exists() and len(T) >= 3:
        with audio_csv.open() as f:
            arows = list(csv.DictReader(f))
        a_times = [float(r["audio_time"]) for r in arows if r.get("audio_time")]
        dv = np.abs(np.diff(tz, 2))  # |acceleration| proxy
        # local maxima above the 80th percentile = trajectory "impact kinks"
        thr = np.percentile(dv, 80)
        kinks = [i + 1 for i in range(1, len(dv) - 1) if dv[i] >= thr and dv[i] >= dv[i - 1] and dv[i] >= dv[i + 1]]
        kink_times = times[kinks] if kinks else np.array([])
        errs = [min(abs(t - kt) for kt in kink_times) * 1000 for t in a_times if len(kink_times)]
        if errs:
            out["audio_err_ms"] = round(float(np.median(errs)), 1)

    # penetration (object center vs body part centers + joints)
    if not args.skip_penetration:
        try:
            from run_human_ball_contact_phase_calibration_anchorinterp_generic import (
                read_human_result, build_body_joints)
            from contact_part_utils import build_contact_part_centers
            human = read_human_result(args.sample_dir / "results" / "gvhmr" / "result.pkl")
            joints = build_body_joints(args.body_model_root, human)[: len(rows)]
            centers = build_contact_part_centers(joints)
            pen_centers = np.stack([*centers.values(), *[joints[:, j, :] for j in range(22)]], axis=0)
            radius = float(rows[0].get("radius_m", 0.12) or 0.12)
            dist = np.linalg.norm(T[None, : len(joints)] - pen_centers[:, : len(T)], axis=2)
            out["pen_frames"] = int(((dist < radius).any(axis=0)).sum())
        except Exception as e:  # noqa: BLE001 — metrics stay usable without GVHMR
            out["pen_frames"] = f"err:{type(e).__name__}"

    print("  ".join(f"{k}={v}" for k, v in out.items()))
    out_csv = args.out_csv or (args.sample_dir / "results" / "evaluation" / "metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = out_csv.exists()
    all_fields = ["sample", "label", "n_frames", "tz_min", "tz_max", "jerk",
                  "contact_rate", "contact_gap_cm", "audio_err_ms", "pen_frames"]
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(out)


if __name__ == "__main__":
    main()
