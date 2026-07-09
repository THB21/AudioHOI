"""VLM-part-selected, full-3D contact constraint.

  1. an LLM/VLM decides WHICH body part(s) make contact (foot for football, fingers/palm for
     basketball, palm for mug) from the clip prompt + object;
  2. at the audio+visual contact frames the object is pulled in FULL 3D (x,y,z) onto the nearest
     selected part (GVHMR foot joints / HaMeR finger keypoints), offset by the object radius so
     its surface touches; the correction is anchor-interpolated and smoothed between contacts.

Unlike the old depth-only anchor, this also fixes the LATERAL offset, so the ball actually sits
on the foot at the touch instant.

Outputs under results/pose6d_contact3d/: contact_profile.json, corrected trajectory, before/after
gap plot + report.

Run: python scripts/shared/human_ball/contact/contact_3d_pipeline.py --sample-dir samples/football_10 [--llm gemini]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
da3spec = importlib.util.spec_from_file_location(
    "da3", HERE.parents[2] / "depth" / "run_depth_anything_v3.py")
da3 = importlib.util.module_from_spec(da3spec); da3spec.loader.exec_module(da3)

# part vocabulary -> source. body=SMPL-X joint idx, hand=HaMeR keypoint name (resolved L/R)
PART_SOURCES = {
    "L_foot": ("body", 10), "R_foot": ("body", 11), "L_ankle": ("body", 7), "R_ankle": ("body", 8),
    "L_knee": ("body", 4), "R_knee": ("body", 5), "L_wrist": ("body", 20), "R_wrist": ("body", 21),
    "thumb_tip": ("hand", "thumb_tip"), "index_tip": ("hand", "index_tip"),
    "middle_tip": ("hand", "middle_tip"), "ring_tip": ("hand", "ring_tip"),
    "pinky_tip": ("hand", "pinky_tip"), "palm": ("hand", "palm"),
}
# generic LLM label -> concrete parts
EXPAND = {
    "foot": ["L_foot", "R_foot", "L_ankle", "R_ankle"], "ankle": ["L_ankle", "R_ankle"],
    "knee": ["L_knee", "R_knee"], "wrist": ["L_wrist", "R_wrist"],
    "hand": ["palm", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
    "fingers": ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
    "thumb": ["thumb_tip"], "index": ["index_tip"], "middle": ["middle_tip"],
    "ring": ["ring_tip"], "pinky": ["pinky_tip"], "palm": ["palm"],
}
FALLBACK = {"football": ["foot"], "basketball": ["hand"], "mug": ["palm", "thumb", "index"]}


def _f(v):
    try:
        return float(v) if v not in (None, "", "nan") else np.nan
    except (ValueError, TypeError):
        return np.nan


def read(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []


def get_prompt(s):
    m = s / "metadata.json"
    if m.exists():
        p = json.loads(m.read_text()).get("prompt", "")
        if p:
            return p
    ps = Path("video_sample/prompts.json")
    if ps.exists():
        nm = s.name.split("_")[-1]
        for e in json.loads(ps.read_text()):
            if e.get("name") in (nm, s.name):
                return e.get("prompt", "")
    return ""


def select_parts_llm(obj, prompt, backend):
    """Ask the LLM which generic parts make contact. Falls back to FALLBACK on any failure."""
    labels = list(EXPAND)
    q = (f"A person interacts with a {obj}. Clip: {prompt[:400]}\n"
         f"Which human body parts physically touch the {obj}? Choose only from: {labels}. "
         'Return JSON {"parts":[...]} with the 1-3 most important.')
    try:
        if backend == "gemini":
            key = os.environ["GEMINI_API_KEY"]
            body = {"contents": [{"parts": [{"text": q}]}],
                    "generationConfig": {"responseMimeType": "application/json"}}
            req = urllib.request.Request(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + key,
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = json.load(r)["candidates"][0]["content"]["parts"][0]["text"]
            parts = json.loads(txt).get("parts", [])
            parts = [p for p in parts if p in EXPAND]
            if parts:
                return parts, f"gemini-2.5-flash"
    except Exception as e:
        print(f"  LLM part selection failed ({e}); using fallback")
    return FALLBACK.get(obj, ["hand"]), "fallback_map"


def part_positions(s, frames, parts, body_model_root):
    """Return dict frame_index -> list of candidate 3D positions for the selected parts."""
    concrete = sorted({c for p in parts for c in EXPAND.get(p, [])})
    need_body = any(PART_SOURCES[c][0] == "body" for c in concrete)
    need_hand = any(PART_SOURCES[c][0] == "hand" for c in concrete)
    joints = None
    if need_body:
        gv = da3.load_gvhmr(s / "results")
        joints = da3.build_smplx_joints(body_model_root, gv)
    hk = {int(_f(r["frame"])): r for r in read(s / "results" / "hands" / "hand_keypoints_3d.csv")} if need_hand else {}
    f0 = frames[0]
    out = {}
    for f in frames:
        cands = []
        for c in concrete:
            src, key = PART_SOURCES[c]
            if src == "body" and joints is not None:
                j = f - f0
                if 0 <= j < len(joints):
                    cands.append(joints[j, key])
            else:
                r = hk.get(f)
                if r:
                    for side in ("left", "right"):
                        p = np.array([_f(r.get(f"{side}_{key}_x")), _f(r.get(f"{side}_{key}_y")),
                                      _f(r.get(f"{side}_{key}_z"))])
                        if not np.any(np.isnan(p)):
                            cands.append(p)
        out[f] = cands
    return out, concrete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--object-name", default=None)
    ap.add_argument("--llm", choices=("gemini", "none"), default="gemini")
    ap.add_argument("--parts", default=None,
                    help="comma list of generic parts to OVERRIDE the LLM (e.g. foot | hand | palm,thumb,index). "
                         "Use this to supply Claude's own contact-part analysis when no API key is available.")
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--smooth-sigma", type=float, default=2.0)
    args = ap.parse_args()
    s = args.sample_dir
    obj = args.object_name or json.loads((s / "metadata.json").read_text()).get("name", s.name.split("_")[-1]) \
        if (s / "metadata.json").exists() else s.name.split("_")[-1]

    # object trajectory
    tp = None
    for rel in ("pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv",
                "pose6d_sharedcam_depthv3/ball_pose6d_sharedcam_trajectory.csv"):
        if (s / "results" / rel).exists():
            tp = s / "results" / rel; break
    rows = read(tp)
    fr = [int(_f(r["frame"])) for r in rows]
    obj_xyz = np.array([[_f(r["tx"]), _f(r["ty"]), _f(r["tz"])] for r in rows])
    radius = float(np.nanmedian([_f(r.get("radius_m")) for r in rows if _f(r.get("radius_m")) > 0]) or 0.12)
    contact = np.array([(_f(r.get("contact_frame")) or 0) >= 1 or (_f(r.get("audio_contact_frame")) or 0) >= 1
                        for r in rows])

    # 1. part selection: explicit override (Claude's analysis) > LLM API > semantic fallback
    prompt = get_prompt(s)
    if args.parts:
        parts, src = [p.strip() for p in args.parts.split(",") if p.strip() in EXPAND], "claude_analysis"
    elif args.llm != "none":
        parts, src = select_parts_llm(obj, prompt, args.llm)
    else:
        parts, src = FALLBACK.get(obj, ["hand"]), "fallback_map"
    pos_by_frame, concrete = part_positions(s, fr, parts, args.body_model_root)
    print(f"object={obj}  contact parts={parts} -> {concrete}  (via {src})")

    # 2. full-3D anchor at contact frames -> nearest selected part (surface-offset by radius)
    corr = np.full((len(fr), 3), np.nan)
    gap_before = np.full(len(fr), np.nan)
    for i, f in enumerate(fr):
        cands = pos_by_frame.get(f, [])
        if not cands or np.any(np.isnan(obj_xyz[i])):
            continue
        d = [np.linalg.norm(obj_xyz[i] - c) for c in cands]
        k = int(np.argmin(d))
        part = cands[k]
        gap_before[i] = d[k]
        if contact[i]:
            ray = obj_xyz[i] - part
            ray = ray / (np.linalg.norm(ray) + 1e-9)
            target = part + radius * ray            # object surface touches the part
            corr[i] = target - obj_xyz[i]

    # interpolate the correction across all frames (hold at ends), then smooth
    idx = np.where(np.isfinite(corr[:, 0]))[0]
    corr_full = np.zeros((len(fr), 3))
    if len(idx):
        for a in range(3):
            corr_full[:, a] = np.interp(np.arange(len(fr)), idx, corr[idx, a])
        try:
            from scipy.ndimage import gaussian_filter1d
            corr_full = gaussian_filter1d(corr_full, sigma=args.smooth_sigma, axis=0, mode="nearest")
        except Exception:
            pass
    obj_new = obj_xyz + corr_full

    # gap after
    gap_after = np.full(len(fr), np.nan)
    for i, f in enumerate(fr):
        cands = pos_by_frame.get(f, [])
        if cands and not np.any(np.isnan(obj_new[i])):
            gap_after[i] = min(np.linalg.norm(obj_new[i] - c) for c in cands)

    out = s / "results" / "pose6d_contact3d"
    out.mkdir(parents=True, exist_ok=True)
    (out / "contact_profile.json").write_text(json.dumps(
        {"object": obj, "contact_parts_generic": parts, "contact_parts_concrete": concrete,
         "source": src, "prompt": prompt[:300]}, indent=2))

    # write corrected trajectory
    new_rows = []
    for i, r in enumerate(rows):
        nr = dict(r)
        nr["tx"], nr["ty"], nr["tz"] = (f"{obj_new[i,0]:.6f}", f"{obj_new[i,1]:.6f}", f"{obj_new[i,2]:.6f}")
        new_rows.append(nr)
    with open(out / "ball_pose6d_contact3d_trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(new_rows)

    # plot
    con = contact
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(fr, gap_before, color="tab:red", alpha=0.7, label="before (object→part)")
    ax.plot(fr, gap_after, color="tab:green", lw=1.8, label="after (full-3D contact)")
    ax.axhline(radius, color="gray", ls="--", lw=1, label=f"radius {radius:.2f}m (touching)")
    for f in np.array(fr)[con]:
        ax.axvline(f, color="red", alpha=0.18)
    ax.set_xlabel("frame"); ax.set_ylabel("object → nearest contact part (m)")
    ax.set_title(f"Full-3D contact ({obj}, parts={parts}) — {s.name}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "contact_gap_before_after.png", dpi=120); plt.close(fig)

    cb, ca = np.abs(gap_before[con]), np.abs(gap_after[con])
    rep = [f"# Full-3D contact — {s.name}", "",
           f"- object: {obj} | parts: {parts} -> {concrete} (via {src})",
           f"- contact frames: {int(con.sum())} | object radius: {radius:.3f} m", "",
           "| metric | before | after |", "|---|---|---|",
           f"| mean object→part at contacts (m) | {np.nanmean(cb):.3f} | {np.nanmean(ca):.3f} |",
           f"| max object→part at contacts (m) | {np.nanmax(cb):.3f} | {np.nanmax(ca):.3f} |",
           f"| contacts within 1.5×radius | {int((cb<1.5*radius).sum())}/{len(cb)} | {int((ca<1.5*radius).sum())}/{len(ca)} |"]
    (out / "contact3d_report.md").write_text("\n".join(rep))
    print(f"  contacts: object→part mean {np.nanmean(cb):.3f}->{np.nanmean(ca):.3f} m,  "
          f"within-touch {int((cb<1.5*radius).sum())}->{int((ca<1.5*radius).sum())}/{len(cb)}")
    print("  wrote", out / "contact_gap_before_after.png")


if __name__ == "__main__":
    main()
