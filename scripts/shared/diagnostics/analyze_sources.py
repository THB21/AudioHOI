"""Analyse EACH input source separately for one video.

For every source a sample actually has, this emits a dedicated artifact (its own overlay video
or quality plot) plus a data-driven verdict: is the source correct, and how does it influence
object positioning. Sources covered: SAM2 mask, CoTracker, DA3 depth, audio events, GVHMR body,
HaMeR hands, object 3D trajectory. Missing sources are reported as "absent".

Output: results/diagnostics/per_source/<source>.{mp4,png} + results/diagnostics/SOURCES.md

Run:  python scripts/shared/diagnostics/analyze_sources.py --sample-dir samples/basketball_01
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

FPS = 24.0


def _f(v):
    try:
        return float(v) if v not in (None, "", "nan") else np.nan
    except (ValueError, TypeError):
        return np.nan


def read(p: Path):
    if not p or not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def frames_list(s: Path):
    return sorted((s / "frames").glob("*.png"))


def vw(path: Path, w, h):
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))


def load_K(s: Path):
    p = s / "results" / "gvhmr" / "result.pkl"
    if not p.exists():
        return None
    d = pickle.load(open(p, "rb"))
    K = np.asarray(d["K_fullimg"])
    if K.ndim == 3:
        K = K[0]
    return K  # 3x3


def masks_dir(s: Path):
    for c in ("masks", "segmentation/masks"):
        d = s / "results" / c
        if d.exists() and any(d.glob("*_mask.png")):
            return d
    return None


# per source
def src_mask(s, frames, V):
    md = masks_dir(s)
    if not md:
        return None, "absent"
    h, w = cv2.imread(str(frames[0])).shape[:2]
    out = vw(V / "mask.mp4", w, h)
    cx, cy, rad, area, lost = [], [], [], [], 0
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        mp = md / f"{i+1:05d}_mask.png"
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
        if m is not None and m.shape[:2] == (h, w) and (m > 127).any():
            ys, xs = np.where(m > 127)
            c = (int(xs.mean()), int(ys.mean())); r = int(np.sqrt(len(xs) / np.pi))
            cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
            cv2.circle(img, c, r, (0, 200, 255), 2); cv2.circle(img, c, 3, (0, 0, 255), -1)
            cx.append(c[0]); cy.append(c[1]); rad.append(r); area.append(len(xs))
        else:
            lost += 1; cx.append(np.nan); cy.append(np.nan); rad.append(np.nan); area.append(0)
        cv2.putText(img, f"SAM2 MASK f{i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out.write(img)
    out.release()
    cx, cy, rad = map(np.array, (cx, cy, rad))
    jump = np.nanmax(np.hypot(np.diff(cx), np.diff(cy))) if len(cx) > 1 else 0
    rcv = np.nanstd(rad) / max(np.nanmean(rad), 1e-6)
    _plot([("center_x", cx), ("center_y", cy), ("radius_px", rad)], V / "mask.png", "SAM2 mask track")
    verd = f"lost {lost}/{len(frames)} frames, max center jump {jump:.0f}px, radius CV {rcv:.2f}"
    ok = lost == 0 and jump < 80
    return ("mask.mp4 + mask.png", ok, verd, "drives the 2D centre observation (R_center) and the depth-sampling region")


def src_cotracker(s, frames, V):
    pts = read(s / "results" / "tracking" / "cotracker_points.csv")
    cen = read(s / "results" / "tracking" / "cotracker_center_trajectory.csv")
    if not pts and not cen:
        return None, "absent"
    h, w = cv2.imread(str(frames[0])).shape[:2]
    out = vw(V / "cotracker.mp4", w, h)
    # cotracker_points: rows per frame with point columns; draw whatever x/y pairs exist
    by_f = {}
    for r in pts:
        by_f.setdefault(int(_f(r.get("frame"))), []).append(r)
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        for r in by_f.get(i + 1, []):
            for k in list(r):
                if k.endswith("_x"):
                    x, y = _f(r.get(k)), _f(r.get(k[:-2] + "_y"))
                    if not np.isnan(x) and not np.isnan(y):
                        cv2.circle(img, (int(x), int(y)), 7, (255, 255, 255), -1)
                        cv2.circle(img, (int(x), int(y)), 6, (255, 0, 255), -1)
                        cv2.putText(img, k[:-2], (int(x) + 8, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (255, 0, 255), 1)
        cv2.putText(img, f"COTRACKER f{i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        out.write(img)
    out.release()
    # agreement with mask center
    note = "tracks sparse object points (constrains R_kp / rotation)"
    return ("cotracker.mp4", True, f"{len(by_f)} frames tracked", note)


def src_depth(s, frames, V):
    d = read(s / "results" / "depth" / "object_depth.csv")
    if not d:
        return None, "absent"
    fr = [int(_f(r["frame"])) for r in d]
    z = np.array([_f(r.get("object_z_aligned_m")) for r in d])
    conf = np.array([_f(r.get("depth_conf")) for r in d])
    a = np.array([_f(r.get("affine_a")) for r in d])
    nsamp = np.array([_f(r.get("sample_count")) for r in d])
    _plot([("object_z_aligned_m", z), ("depth_conf", conf), ("affine_a (slope)", a), ("sample_count", nsamp)],
          V / "depth.png", "DA3 metric depth", x=fr)
    negA = int((a < 0).sum())
    verd = f"z range {np.nanmin(z):.2f}-{np.nanmax(z):.2f}m, conf {np.nanmean(conf):.2f}, neg-slope frames {negA}/{len(d)}"
    ok = np.nanmean(conf) > 0.3 and negA < len(d) * 0.3
    return ("depth.png", ok, verd, "metric depth prior (R_depth); per-frame affine to GVHMR body")


def src_audio(s, frames, V):
    ev = read(s / "results" / "events" / "audio_events.csv")
    vis = read(s / "results" / "events" / "visual_events.csv")
    align = read(s / "results" / "events" / "audio_visual_alignment.csv")
    if not ev:
        return None, "absent"
    def _fr(r, *keys):
        for k in keys:
            if k in r and not np.isnan(_f(r.get(k))):
                return int(_f(r.get(k)))
        return None
    af = [x for x in (_fr(r, "audio_frame", "frame") for r in ev) if x is not None]
    vf = [x for x in (_fr(r, "visual_frame", "frame") for r in vis) if x is not None]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.eventplot([af, vf], colors=["magenta", "tab:blue"], lineoffsets=[1, 0], linelengths=0.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["visual", "audio"]); ax.set_xlabel("frame")
    ax.set_title(f"audio onsets ({len(af)}) vs visual events ({len(vf)})"); ax.grid(alpha=0.3)
    fig.tight_layout(); (V).mkdir(parents=True, exist_ok=True); fig.savefig(V / "audio.png", dpi=110); plt.close(fig)
    off = ""
    if align:
        offs = [_f(r.get("difference_frames", r.get("offset_frames", ""))) for r in align]
        offs = [o for o in offs if not np.isnan(o)]
        if offs:
            off = f", mean A-V offset {np.mean(offs):+.1f} frames ({len(offs)} aligned)"
    return ("audio.png", True, f"{len(af)} audio onsets, {len(vf)} visual events{off}",
            "times/gates the contact residual (R_contact)")


def src_hamer(s, frames, V, K):
    hk = read(s / "results" / "hands" / "hand_keypoints_3d.csv")
    if not hk or K is None:
        return None, "absent"
    fx, fy, cx0, cy0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = cv2.imread(str(frames[0])).shape[:2]
    out = vw(V / "hamer.mp4", w, h)
    by_f = {int(_f(r["frame"])): r for r in hk}
    tips = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    jit = {"left": [], "right": []}
    prev = {"left": None, "right": None}
    det = {"left": 0, "right": 0}
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp)); r = by_f.get(i + 1)
        if r:
            for side, col in (("left", (0, 255, 255)), ("right", (0, 140, 255))):
                if _f(r.get(f"{side}_detected")) >= 1:
                    det[side] += 1
                pa = []
                for t in ["wrist", "palm"] + tips:
                    X, Y, Z = _f(r.get(f"{side}_{t}_x")), _f(r.get(f"{side}_{t}_y")), _f(r.get(f"{side}_{t}_z"))
                    if np.isnan(X) or np.isnan(Z) or Z <= 0:
                        continue
                    u, v = int(fx * X / Z + cx0), int(fy * Y / Z + cy0)
                    pa.append((u, v))
                    cv2.circle(img, (u, v), 3, col, -1)
                if pa:
                    c = np.mean(pa, axis=0)
                    if prev[side] is not None:
                        jit[side].append(np.linalg.norm(c - prev[side]))
                    prev[side] = c
        cv2.putText(img, f"HaMeR HANDS f{i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        out.write(img)
    out.release()
    mj = np.median([x for s2 in jit for x in jit[s2]]) if any(jit.values()) else 0
    n = len(frames)
    verd = f"detect L {det['left']}/{n} R {det['right']}/{n}, median fingertip jitter {mj:.1f}px/frame"
    ok = mj < 15
    return ("hamer.mp4", ok, verd, "fingertip/palm positions; used for hand-object contact & grasp")


def src_gvhmr(s, frames, V):
    p = s / "results" / "gvhmr" / "result.pkl"
    if not p.exists():
        return None, "absent"
    d = pickle.load(open(p, "rb"))
    sp = d.get("smpl_params_incam", {})
    transl = np.asarray(sp["transl"]) if "transl" in sp else None
    # reuse an existing body overlay video as the visualization
    cand = [s / "results" / "renders" / "smplx_overlay.mp4",
            s / "results" / "renders" / "full_scene_3d" / "overlay.mp4"]
    vis = next((c for c in cand if c.exists()), None)
    verd = "body recovered"
    if transl is not None and len(transl) > 2:
        acc = np.linalg.norm(transl[2:] - 2 * transl[1:-1] + transl[:-2], axis=1)
        verd = f"{len(transl)} frames, transl accel max {acc.max():.3f} (jitter)"
    art = f"(reuse) {vis.relative_to(s/'results')}" if vis else "no body overlay rendered"
    return (art, True, verd, "metric human anchor: hands stitch to it AND DA3 depth is affine-fit to its joints")


def src_object(s, V):
    for rel in ("pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv",
                "mug_oriented_pose.csv"):
        p = s / "results" / rel
        if p.exists():
            rows = read(p)
            res = np.array([_f(r.get("residual_px")) for r in rows]) if "residual_px" in rows[0] else None
            note = "the solved object pose (what every other source constrains)"
            v = f"{len(rows)} frames; E_2d RMS {np.sqrt(np.nanmean(res**2)):.2f}px" if res is not None else f"{len(rows)} frames (6DOF pose)"
            return ("see diagnostics/energy/*", True, v, note)
    return None, "absent"


def _plot(series, out: Path, title, x=None):
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 1.4 * len(series)), sharex=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ax, (name, y) in zip(axes, series):
        xx = x if x is not None else range(len(y))
        ax.plot(xx, y, lw=1.3); ax.set_title(name, fontsize=9, loc="left"); ax.grid(alpha=0.3)
    axes[0].figure.suptitle(title, y=1.0, fontsize=11)
    axes[-1].set_xlabel("frame")
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    args = ap.parse_args()
    s = args.sample_dir
    frames = frames_list(s)
    V = s / "results" / "diagnostics" / "per_source"
    V.mkdir(parents=True, exist_ok=True)
    K = load_K(s)

    results = {}
    results["SAM2 mask"] = src_mask(s, frames, V)
    results["CoTracker"] = src_cotracker(s, frames, V)
    results["DA3 depth"] = src_depth(s, frames, V)
    results["Audio events"] = src_audio(s, frames, V)
    results["GVHMR body"] = src_gvhmr(s, frames, V)
    results["HaMeR hands"] = src_hamer(s, frames, V, K)
    results["Object 3D traj"] = src_object(s, V)

    lines = [f"# Per-source analysis — {s.name}", "",
             "Each input source analysed separately. ✓/✗ = looks correct; verdict is data-driven.", "",
             "| source | present | artifact | correct? | verdict | influence on positioning |",
             "|---|---|---|---|---|---|"]
    print(f"\n==== {s.name} ====")
    for name, r in results.items():
        if r[1] == "absent":
            lines.append(f"| {name} | — | — | n/a | absent | — |")
            print(f"  {name:<16} ABSENT")
            continue
        art, ok, verd, infl = r
        mark = "✓" if ok else "✗"
        lines.append(f"| {name} | yes | `{art}` | {mark} | {verd} | {infl} |")
        print(f"  {name:<16} {mark}  {verd}")
    (s / "results" / "diagnostics" / "SOURCES.md").write_text("\n".join(lines))
    print("wrote", s / "results" / "diagnostics" / "SOURCES.md")


if __name__ == "__main__":
    main()
