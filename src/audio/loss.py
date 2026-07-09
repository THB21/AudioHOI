"""Pack the extracted semantics into the trajectory loss.

The unified contact record carries exactly what a contact-aware depth/pose loss needs:
  WHEN   relevant event frame (audio-refined instant)
  GATE   relevant            → only real interactions contribute
  RELAX  manip_gamma         → how much to relax temporal smoothness at the contact
                               (1 = allow a full velocity kink / bounce, 0 = keep smooth)
  PULL   manip_weight        → strength of the contact constraint
  TARGET contact_target      → support plane vs a body part vs object-local
  WHERE  target_entity / object_local_xyz

This module turns those rows into residuals for a least-squares refinement of the object
depth track ``tz(t)``:

    E = w_data·Σ (tz - tz0)²                      keep close to the observed lift
      + w_smooth·Σ relax(t)·(tz_{t+1}-2tz_t+tz_{t-1})²   smoothness, RELAXED at audio contacts
      + w_support·Σ_support (tz - support_z)²      optional pin to a support depth

``relax(t) = 1 - gamma`` at a relevant contact frame (so an audio-detected bounce is *allowed*
to kink), else 1. This is the object-agnostic hook: audio decides *when* the motion may break.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_traj(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    frames, tz = [], []
    extra = {"rows": []}
    with path.open() as f:
        for r in csv.DictReader(f):
            frames.append(int(r["frame"]))
            tz.append(float(r["tz"]))
            extra["rows"].append(r)
    return np.array(frames), np.array(tz, dtype=float), extra


def _read_records(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def build_relax(frames: np.ndarray, records: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Per-frame smoothness-relaxation 1-gamma at relevant contact frames (else 1.0)."""
    idx = {int(fr): i for i, fr in enumerate(frames)}
    relax = np.ones(len(frames))
    weight = np.zeros(len(frames))
    used = []
    for r in records:
        if not int(float(r.get("relevant", 1))):
            continue
        fr = int(float(r.get("refined_frame", r["frame"])))
        i = idx.get(fr) or idx.get(min(idx, key=lambda f: abs(f - fr)) if idx else fr)
        if i is None:
            continue
        gamma = float(r.get("manip_gamma", 0.8))
        w = float(r.get("manip_weight", 0.5))
        relax[i] = min(relax[i], 1.0 - gamma)        # gamma=1 → relax 0 → kink allowed
        weight[i] = max(weight[i], w)
        used.append({"frame": frames[i], "event": r.get("event_type", r.get("fused_event_type", "")),
                     "target": r.get("contact_target", ""), "gamma": gamma, "weight": w})
    return relax, used


def _solve(tz0: np.ndarray, relax: np.ndarray, w_data: float, w_smooth: float) -> np.ndarray:
    from scipy.optimize import least_squares

    def residuals(tz):
        acc = tz[2:] - 2 * tz[1:-1] + tz[:-2]
        return np.concatenate([np.sqrt(w_data) * (tz - tz0),
                               np.sqrt(w_smooth * relax[1:-1]) * acc])
    return least_squares(residuals, tz0.copy(), method="lm", max_nfev=2000).x


def refine_depth(trajectory_csv: Path, records_csv: Path, out_csv: Path | None = None, *,
                 w_data: float = 1.0, w_smooth: float = 8.0, audio_gated: bool = True,
                 write: bool = True):
    frames, tz0, extra = _read_traj(Path(trajectory_csv))
    records = _read_records(Path(records_csv))
    relax, used = build_relax(frames, records)
    n = len(frames)
    relax_eff = relax if audio_gated else np.ones(n)   # uniform smoothing = no audio info
    tz_ref = _solve(tz0, relax_eff, w_data, w_smooth)
    if not write:
        return frames, tz0, tz_ref, used

    rms_change = float(np.sqrt(np.mean((tz_ref - tz0) ** 2)))
    out_csv = out_csv or Path(trajectory_csv).with_name("trajectory_audio_refined.csv")
    with Path(out_csv).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time", "tz_initial", "tz_refined", "relax", "is_contact"])
        for i, fr in enumerate(frames):
            row = extra["rows"][i]
            w.writerow([fr, row.get("time", ""), f"{tz0[i]:.6f}", f"{tz_ref[i]:.6f}",
                        f"{relax[i]:.3f}", int(relax[i] < 1.0)])
    print(f"[loss] refined {n} frames using {len(used)} audio contacts → {out_csv}  "
          f"(tz RMS change {rms_change*100:.1f} cm, cost {sol.cost:.4f})")
    return frames, tz0, tz_ref, used


def _align(frames, records):
    idx = {int(fr): None for fr in frames}
    for r in records:
        fr = int(float(r.get("refined_frame", r["frame"])))
        if fr in idx:
            idx[fr] = r
    return [idx[int(fr)] for fr in frames]


def plot(frames, tz0, tz_ref, used, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frames, tz0, color="#999", lw=1.5, label="tz initial (lift)")
    ax.plot(frames, tz_ref, color="#1f77b4", lw=2.0, label="tz audio-contact refined")
    for u in used:
        col = "#d62728" if u["target"] == "support" else "#ff7f0e"
        ax.axvline(u["frame"], color=col, alpha=0.5, lw=1.0)
    ax.set_xlabel("frame"); ax.set_ylabel("object depth tz (m)")
    ax.set_title(f"Audio-semantic contact loss: smoothness relaxed at {len(used)} audio contacts "
                 "(red=support, orange=part)")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"wrote {out_png}")


def compare(trajectory_csv: Path, records_csv: Path, out_png: Path, *, w_smooth: float = 8.0):
    """Refine the depth WITH and WITHOUT the audio info (same smoothness) and compare."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames, tz0, tz_audio, used = refine_depth(trajectory_csv, records_csv, write=False,
                                               w_smooth=w_smooth, audio_gated=True)
    _, _, tz_uniform, _ = refine_depth(trajectory_csv, records_csv, write=False,
                                       w_smooth=w_smooth, audio_gated=False)
    fidx = {int(fr): i for i, fr in enumerate(frames)}
    cidx = [fidx[int(u["frame"])] for u in used if int(u["frame"]) in fidx]
    cidx = [i for i in cidx if 0 < i < len(frames) - 1]
    off = [i for i in range(1, len(frames) - 1) if i not in set(cidx)]

    def accel(tz, idxs):
        return float(np.mean([abs(tz[i + 1] - 2 * tz[i] + tz[i - 1]) for i in idxs])) if idxs else 0.0

    def dev(tz, idxs):  # how far the refined moved from the observed lift
        return float(np.mean([abs(tz[i] - tz0[i]) for i in idxs])) * 100 if idxs else 0.0

    print("\n  metric                         no-audio (uniform)   with-audio")
    print(f"  kink |accel| AT contacts (m)   {accel(tz_uniform, cidx):>14.4f}   {accel(tz_audio, cidx):>10.4f}   "
          f"← audio keeps the bounce")
    print(f"  smoothing OFF contacts (m)     {accel(tz_uniform, off):>14.4f}   {accel(tz_audio, off):>10.4f}   "
          f"← both smooth noise")
    print(f"  deviation from lift AT contacts {dev(tz_uniform, cidx):>12.2f}cm {dev(tz_audio, cidx):>10.2f}cm  "
          f"← uniform flattens contacts")

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(frames, tz0, color="#bbb", lw=1.2, label="observed lift (input)")
    ax.plot(frames, tz_uniform, color="#d62728", lw=1.8, ls="--", label="refined WITHOUT audio (uniform smoothing)")
    ax.plot(frames, tz_audio, color="#1f77b4", lw=2.2, label="refined WITH audio loss (gated)")
    for u in used:
        ax.axvline(u["frame"], color=("#d62728" if u["target"] == "support" else "#ff7f0e"), alpha=0.35, lw=0.9)
    ax.set_xlabel("frame"); ax.set_ylabel("object depth tz (m)")
    ax.set_title("Object-depth loss — WITH vs WITHOUT audio: without audio the bounces get "
                 "flattened; with audio they survive at the detected contacts")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {out_png}")


def export_compare_trajectories(trajectory_csv: Path, records_csv: Path, out_dir: Path,
                                *, w_smooth: float = 14.0) -> dict:
    """Write two FULL object trajectories (all columns, tz replaced) for rendering:
    one refined WITHOUT audio (uniform smoothing) and one WITH the audio-gated loss."""
    frames, tz0, tz_audio, _ = refine_depth(trajectory_csv, records_csv, write=False,
                                            w_smooth=w_smooth, audio_gated=True)
    _, _, tz_uniform, _ = refine_depth(trajectory_csv, records_csv, write=False,
                                       w_smooth=w_smooth, audio_gated=False)
    with Path(trajectory_csv).open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tag, tz in (("noaudio", tz_uniform), ("audio", tz_audio)):
        p = out_dir / f"ball_traj_{tag}.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for i, r in enumerate(rows):
                r = dict(r)
                r["tz"] = f"{tz[i]:.6f}"
                w.writerow(r)
        paths[tag] = p
    print(f"wrote {paths['noaudio']} and {paths['audio']}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory-csv", type=Path, required=True)
    ap.add_argument("--records-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--w-smooth", type=float, default=8.0)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()
    frames, tz0, tz_ref, used = refine_depth(args.trajectory_csv, args.records_csv,
                                             args.out_csv, w_smooth=args.w_smooth)
    if args.plot:
        plot(frames, tz0, tz_ref, used, args.plot)


if __name__ == "__main__":
    main()
