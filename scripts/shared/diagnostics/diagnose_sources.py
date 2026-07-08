"""Per-source / per-loss diagnostics for an AudioHOI sample.

For each available input source / loss term this writes:
  - an overlay video so you can SEE if that source is wrong (mask drift, bad reproj, ...)
  - a per-term loss curve (read straight from the solved trajectory CSV where possible)

Schema-aware: ball samples (basketball/football) expose a rich trajectory CSV
(residual_px, contact_depth_gap, u_obs/u_proj, floor_v ...); the mug exposes a 6DOF
pose CSV (tx..roll) with no precomputed residuals, so it gets pose-component curves
plus mask/depth overlays. Missing sources are skipped cleanly (printed as SKIP).

Outputs under: <sample>/results/diagnostics/{sources/*.mp4, curves/loss_curves.png, SUMMARY.md}

Run (gvhmr env):
  python scripts/shared/diagnostics/diagnose_sources.py --sample-dir samples/basketball_01
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 24.0


# io
def read_csv(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str):
    v = row.get(key)
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def frames_list(sample: Path) -> list[Path]:
    return sorted((sample / "frames").glob("*.png"))


def masks_dir(sample: Path) -> Path | None:
    for cand in ("segmentation/masks", "masks"):
        d = sample / "results" / cand
        if d.exists() and any(d.glob("*_mask.png")):
            return d
    return None


def pick_trajectory(sample: Path) -> tuple[Path | None, str]:
    """Return (csv_path, kind) where kind in {ball, mug6d, none}."""
    ball = [
        "pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv",
        "pose6d_sharedcam_depthv3/ball_pose6d_sharedcam_trajectory.csv",
        "pose6d_sharedcam/ball_pose6d_sharedcam_trajectory.csv",
    ]
    for rel in ball:
        p = sample / "results" / rel
        if p.exists():
            return p, "ball"
    for rel in ("mug_oriented_pose.csv", "mug_handle_grip_pose.csv", "mug_rough_trajectory.csv"):
        p = sample / "results" / rel
        if p.exists():
            return p, "mug6d"
    return None, "none"


def video_writer(path: Path, w: int, h: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))


# overlays
def overlay_mask(sample: Path, frames: list[Path], mdir: Path, out: Path) -> str:
    h, w = cv2.imread(str(frames[0])).shape[:2]
    vw = video_writer(out, w, h)
    n_ok = 0
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        mp = mdir / f"{i+1:05d}_mask.png"
        if mp.exists():
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape[:2] == (h, w):
                cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
                ys, xs = np.where(m > 127)
                if len(xs) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    r = int(np.sqrt(len(xs) / np.pi))
                    cv2.circle(img, (cx, cy), r, (0, 200, 255), 2)
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
                    n_ok += 1
        cv2.putText(img, f"MASK f{i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        vw.write(img)
    vw.release()
    return f"mask_overlay.mp4 ({n_ok}/{len(frames)} frames with mask)"


def overlay_reproj(sample: Path, frames: list[Path], traj: list[dict], out: Path) -> str:
    by_f = {int(float(r["frame"])): r for r in traj}
    h, w = cv2.imread(str(frames[0])).shape[:2]
    vw = video_writer(out, w, h)
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        r = by_f.get(i + 1) or by_f.get(i)
        if r:
            uo, vo, ro = fnum(r, "u_obs"), fnum(r, "v_obs"), fnum(r, "radius_obs_px")
            up, vp, rp = fnum(r, "u_proj"), fnum(r, "v_proj"), fnum(r, "radius_proj_px")
            res = fnum(r, "residual_px")
            if uo is not None and vo is not None:
                cv2.circle(img, (int(uo), int(vo)), int(ro or 6), (0, 255, 0), 2)  # obs green
            if up is not None and vp is not None:
                cv2.circle(img, (int(up), int(vp)), int(rp or 6), (0, 140, 255), 2)  # proj orange
            if res is not None:
                cv2.putText(img, f"residual_px={res:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        cv2.putText(img, "REPROJ  green=obs  orange=proj", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(img)
    vw.release()
    return "reproj_overlay.mp4 (green=2D obs, orange=projected 3D)"


def overlay_contact(sample: Path, frames: list[Path], traj: list[dict], out: Path) -> str:
    by_f = {int(float(r["frame"])): r for r in traj}
    h, w = cv2.imread(str(frames[0])).shape[:2]
    vw = video_writer(out, w, h)
    n_contact = 0
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        r = by_f.get(i + 1) or by_f.get(i)
        if r:
            cf = fnum(r, "contact_frame") or 0
            af = fnum(r, "audio_contact_frame") or 0
            gap = fnum(r, "contact_depth_gap")
            tag = []
            if cf >= 1:
                tag.append("HUMAN-CONTACT"); n_contact += 1
                cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            if af >= 1:
                tag.append("AUDIO")
                cv2.rectangle(img, (4, 4), (w - 5, h - 5), (255, 0, 255), 4)
            if gap is not None:
                cv2.putText(img, f"contact_gap={gap:+.3f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if tag:
                cv2.putText(img, " ".join(tag), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"CONTACT f{i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(img)
    vw.release()
    return f"contact_overlay.mp4 ({n_contact} human-contact frames)"


# curves
def accel(series: list[float]) -> list[float]:
    a = [0.0] * len(series)
    for k in range(1, len(series) - 1):
        a[k] = abs(series[k + 1] - 2 * series[k] + series[k - 1])
    return a


def loss_curves_ball(traj: list[dict], depth: list[dict], out: Path) -> str:
    fr = [int(float(r["frame"])) for r in traj]
    tz = [fnum(r, "tz") or np.nan for r in traj]
    res = [fnum(r, "residual_px") or 0.0 for r in traj]
    gap = [fnum(r, "contact_depth_gap") for r in traj]
    cfr = [int(fnum(r, "contact_frame") or 0) for r in traj]
    afr = [int(fnum(r, "audio_contact_frame") or 0) for r in traj]
    tz_acc = accel([0.0 if np.isnan(x) else x for x in tz])
    dmap = {int(float(d["frame"])): fnum(d, "object_z_aligned_m") for d in depth}
    da3 = [dmap.get(f) for f in fr]

    panels = [
        ("E_2d : center reprojection residual (px)", res, "tab:blue"),
        ("E_contact : human/floor contact depth gap (m)", [np.nan if g is None else g for g in gap], "tab:red"),
        ("depth : object tz (m)  +  DA3 aligned (dashed)", tz, "tab:green"),
        ("E_smooth : |tz acceleration| (m/frame^2)", tz_acc, "tab:purple"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 9), sharex=True)
    for ax, (title, y, c) in zip(axes, panels):
        ax.plot(fr, y, color=c, lw=1.5)
        if title.startswith("depth"):
            ax.plot(fr, da3, color="black", ls="--", lw=1.0, label="DA3 aligned")
            ax.legend(loc="upper right", fontsize=8)
        for f, cf, af in zip(fr, cfr, afr):
            if cf:
                ax.axvline(f, color="red", alpha=0.12)
            if af:
                ax.axvline(f, color="magenta", alpha=0.18, ls=":")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame  (red=human-contact, magenta:=audio)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return "loss_curves.png (E_2d, E_contact, depth/DA3, E_smooth)"


def loss_curves_mug(traj: list[dict], depth: list[dict], out: Path) -> str:
    fr = [int(float(r["frame"])) for r in traj]
    comps = [c for c in ("tx", "ty", "tz", "yaw", "pitch", "roll") if c in traj[0]]
    fig, axes = plt.subplots(len(comps) + (1 if depth else 0), 1, figsize=(12, 2 + 1.4 * len(comps)), sharex=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ax, c in zip(axes, comps):
        y = [fnum(r, c) for r in traj]
        ax.plot(fr, y, lw=1.5)
        acc = accel([v or 0.0 for v in y])
        j = int(np.argmax(acc)) if acc else 0
        if acc and acc[j] > 1e-6:
            ax.axvline(fr[j], color="red", alpha=0.4, ls=":")
            ax.set_title(f"{c}  (max jump @f{fr[j]}, |acc|={acc[j]:.3g})", fontsize=10, loc="left")
        else:
            ax.set_title(f"{c}  (constant / not estimated)", fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    if depth:
        dmap = {int(float(d["frame"])): fnum(d, "object_z_aligned_m") for d in depth}
        axes[len(comps)].plot(fr, [dmap.get(f) for f in fr], color="black")
        axes[len(comps)].set_title("DA3 aligned object depth (m)", fontsize=10, loc="left")
        axes[len(comps)].grid(alpha=0.3)
    axes[-1].set_xlabel("frame  (red dotted = largest per-component jump)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return "loss_curves.png (6DOF pose components + jump markers + DA3 depth)"


# main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--sources", default="all")
    args = ap.parse_args()
    sample = args.sample_dir
    out_root = sample / "results" / "diagnostics"
    src_dir = out_root / "sources"
    frames = frames_list(sample)
    if not frames:
        raise SystemExit(f"no frames in {sample}/frames")

    traj_path, kind = pick_trajectory(sample)
    traj = read_csv(traj_path)
    depth = read_csv(sample / "results" / "depth" / "object_depth.csv")
    mdir = masks_dir(sample)

    notes = []
    print(f"sample={sample.name}  kind={kind}  traj={traj_path.name if traj_path else None}  "
          f"masks={'yes' if mdir else 'NO'}  depth={'yes' if depth else 'NO'}  frames={len(frames)}")

    if mdir:
        notes.append("mask: " + overlay_mask(sample, frames, mdir, src_dir / "mask_overlay.mp4"))
    else:
        notes.append("mask: SKIP (no masks)")

    if kind == "ball" and traj:
        notes.append("reproj: " + overlay_reproj(sample, frames, traj, src_dir / "reproj_overlay.mp4"))
        notes.append("contact: " + overlay_contact(sample, frames, traj, src_dir / "contact_overlay.mp4"))
        notes.append("curves: " + loss_curves_ball(traj, depth, out_root / "curves" / "loss_curves.png"))
    elif kind == "mug6d" and traj:
        notes.append("curves: " + loss_curves_mug(traj, depth, out_root / "curves" / "loss_curves.png"))
        notes.append("reproj: SKIP (mug pose CSV has no precomputed reprojection residual)")
        notes.append("contact: SKIP (no contact_frame column in mug pose CSV)")
    else:
        notes.append("curves: SKIP (no usable trajectory)")

    summ = out_root / "SUMMARY.md"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(summ, "w") as f:
        f.write(f"# Source / loss diagnostics — {sample.name}\n\n")
        f.write(f"- trajectory: `{traj_path}` (kind={kind})\n")
        f.write(f"- masks: {'yes' if mdir else 'NO'} | depth: {'yes' if depth else 'NO'}\n\n")
        f.write("## Artifacts\n\n")
        for n in notes:
            f.write(f"- {n}\n")
        f.write("\n## Source correctness (fill in after viewing)\n\n")
        f.write("| source | looks correct? | how it influences positioning |\n|---|---|---|\n")
        for s in ("SAM2 mask", "CoTracker", "DA3 depth", "2D reproj (E_2d)", "contact (E_contact)",
                  "floor/support", "temporal smooth", "audio events"):
            f.write(f"| {s} |  |  |\n")
    print("wrote", summ)
    for n in notes:
        print("  -", n)


if __name__ == "__main__":
    main()
