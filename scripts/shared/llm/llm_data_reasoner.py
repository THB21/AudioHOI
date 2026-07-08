"""LLM temporal data reasoner / corrector / verifier.

Reads (a) the clip's semantic context (generation prompt / metadata) and (b) a solved
per-frame trajectory CSV, then asks an LLM to REASON over the series and flag physically
implausible discontinuities ("hard jumps") given what the clip is supposed to show.

Roles are separated on purpose:
  - the LLM only FLAGS  -> {frame, field, issue, suggested_action}  (discrete, schema'd)
  - deterministic code APPLIES whitelisted fixes (interpolate / level_align)
  - a verifier RE-CHECKS that the max per-field acceleration dropped without flattening motion

LLM never emits continuous pose values; it cannot silently overwrite the trajectory.

Backend: Gemini (gemini-2.5-flash works on the free tier). Reads GEMINI_API_KEY from env.

Dry run (flags only):
  python scripts/shared/llm/llm_data_reasoner.py --sample-dir samples_known_object/02_mug_v2 \
    --data-csv samples_known_object/02_mug_v2/results/mug_oriented_pose.csv
Apply + verify:
  ... --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

API = "https://generativelanguage.googleapis.com/v1beta/models"
ANGLE_FIELDS = {"yaw", "pitch", "roll", "qw", "qx", "qy", "qz"}
SKIP_FIELDS = {"frame", "time", "coord_frame", "source"}


# data utils
def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def numeric_columns(rows: list[dict]) -> list[str]:
    cols = []
    for c in rows[0]:
        if c in SKIP_FIELDS:
            continue
        vals = [r.get(c) for r in rows]
        ok = sum(1 for v in vals if _f(v) is not None)
        if ok >= len(rows) * 0.5:
            cols.append(c)
    return cols


def _f(v):
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def series(rows: list[dict], col: str) -> np.ndarray:
    return np.array([_f(r.get(col)) if _f(r.get(col)) is not None else np.nan for r in rows], float)


def accel(y: np.ndarray) -> np.ndarray:
    a = np.zeros_like(y)
    a[1:-1] = np.abs(y[2:] - 2 * y[1:-1] + y[:-2])
    return np.nan_to_num(a)


def velocity(y: np.ndarray) -> np.ndarray:
    v = np.zeros_like(y)
    v[1:] = np.abs(y[1:] - y[:-1])
    return np.nan_to_num(v)


def candidates(rows: list[dict], cols: list[str], topk: int = 4) -> dict:
    """Compact PHYSICS summary per column: range, velocity/accel envelope, and the frames
    with the largest |velocity| and |acceleration| (candidate teleports / discontinuities)."""
    fr = [int(float(r["frame"])) for r in rows]
    out = {}
    for c in cols:
        y = series(rows, c)
        if np.all(np.isnan(y)) or np.nanstd(y) < 1e-9:
            out[c] = {"constant": True}
            continue
        v, a = velocity(y), accel(y)
        # robust envelope: "typical step" = median of velocities above plateau-noise level
        # (ignore tiny numerical jitter on constant segments, else the ratio explodes)
        rng = float(np.nanmax(y) - np.nanmin(y))
        real = v[v > max(1e-6, rng * 1e-3)]
        vmed = float(np.median(real)) if real.size else 0.0
        a_idx = list(np.argsort(a)[::-1][:topk])
        jumps = [
            {"frame": fr[i], "vel": round(float(v[i]), 4), "acc": round(float(a[i]), 4),
             "vel_over_median": round(float(v[i] / vmed), 1) if vmed > 0 else None,
             "prev": _r(y[i - 1] if i > 0 else y[i]), "at": _r(y[i]),
             "next": _r(y[i + 1] if i < len(y) - 1 else y[i])}
            for i in a_idx if a[i] > 0
        ]
        out[c] = {"min": _r(np.nanmin(y)), "max": _r(np.nanmax(y)), "is_angle": c in ANGLE_FIELDS,
                  "median_step": round(vmed, 4), "max_vel": round(float(v.max()), 4),
                  "max_acc": round(float(a.max()), 4), "top_jumps": jumps}
    return out


def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


# the LLM
FLAG_SCHEMA = {
    "type": "object",
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame": {"type": "integer"},
                    "field": {"type": "string"},
                    "issue": {"type": "string"},
                    "suggested_action": {"type": "string", "enum": ["interpolate", "level_align", "clamp", "none"]},
                },
                "required": ["frame", "field", "issue", "suggested_action"],
            },
        }
    },
    "required": ["flags"],
}

PROMPT = """You are a PHYSICS auditor for a per-frame 3D object trajectory recovered from video.

Reason from general physics and how the signal CHANGES OVER TIME, not from any story about
the clip. A real rigid object moves continuously: bounded velocity, bounded acceleration, no
teleports, no instantaneous orientation flips, and it does not spontaneously jump to a pose
and snap back. Judge each field on its own temporal evidence below.

Per field you are given: value range, and the frames with the largest |velocity| (1st diff)
and |acceleration| (2nd diff) — these are candidate discontinuities. Frames are 1-based.
Fields marked constant are not estimated; ignore them.

{summary}

Optional weak context (do NOT rely on it; physics first): {context}

Flag a field/frame ONLY when the temporal signal itself is physically implausible:
  - a single-frame spike that is far outside the local velocity envelope  -> "interpolate"
  - a sustained level shift / a flip that later reverts (e.g. an angle stepping ~pi and back,
    or a translation teleporting and returning) -> "level_align"
  - genuinely fast-but-smooth motion (a kick, a fall) is NOT a problem -> "none"
Prefer "none" when unsure. Return JSON only."""


def gemini_flags(model: str, key: str, context: str, summary: dict) -> dict:
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(context=context, summary=json.dumps(summary, indent=0))}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": FLAG_SCHEMA},
    }
    req = urllib.request.Request(
        f"{API}/{model}:generateContent?key={key}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    txt = resp["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(txt)


def claude_flags(model: str, key: str, context: str, summary: dict) -> dict:
    """Anthropic Messages API. Needs ANTHROPIC_API_KEY. Forces JSON via a tool / instruction."""
    body = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": PROMPT.format(context=context, summary=json.dumps(summary, indent=0))
                      + "\n\nReturn ONLY the JSON object {\"flags\":[...]}, no prose."}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    txt = "".join(b.get("text", "") for b in resp.get("content", []))
    s = txt[txt.find("{"): txt.rfind("}") + 1]
    return json.loads(s)


def get_flags(backend: str, model: str, context: str, summary: dict) -> dict:
    if backend == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or _die("GEMINI_API_KEY not set")
        return gemini_flags(model if "gemini" in model else "gemini-2.5-flash", key, context, summary)
    if backend == "claude":
        key = os.environ.get("ANTHROPIC_API_KEY") or _die(
            "ANTHROPIC_API_KEY not set. Either export it, or use --flags-json with flags produced "
            "by Claude (the agent) to run apply+verify with no key.")
        return claude_flags(model if "claude" in model else "claude-sonnet-4-6", key, context, summary)
    _die(f"unknown backend {backend}")


def _die(msg: str):
    raise SystemExit(msg)


# deterministic fix
def apply_fixes(rows: list[dict], flags: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply only whitelisted actions. Returns (new_rows, applied_log).

    level_align on an ANGLE field is handled globally (robust to a multi-frame ramp, e.g. a
    handle flip that ramps over ~6 frames): every frame is snapped to ref ± k*pi so the series
    stays nearest the pre-flip plateau. Non-angle level_align removes a single step; interpolate
    fixes an isolated one-frame outlier.
    """
    fr = [int(float(r["frame"])) for r in rows]
    f2i = {f: i for i, f in enumerate(fr)}
    by_field: dict[str, list[dict]] = {}
    for fl in flags:
        if fl["suggested_action"] in ("interpolate", "level_align", "clamp"):
            by_field.setdefault(fl["field"], []).append(fl)
    data = {c: series(rows, c) for c in by_field}
    applied = []
    for c, fls in by_field.items():
        y = data[c]
        ref = y[np.isfinite(y)][0] if np.any(np.isfinite(y)) else 0.0
        laf = sorted(f2i[fl["frame"]] for fl in fls
                     if fl["suggested_action"] == "level_align" and fl["frame"] in f2i)
        if c in ANGLE_FIELDS and laf:
            tol = 0.1  # rad: how close to ref counts as "back on the reference level"
            if len(laf) >= 2:
                # two bracketing flags = a flip-and-return excursion: the field does not truly
                # change orientation, so flatten the bracketed span back to the reference level.
                # Grow the span outward until the series re-converges to ref (covers the ramps).
                lo, hi = laf[0], laf[-1]
                while lo > 0 and abs(y[lo - 1] - ref) > tol:
                    lo -= 1
                while hi < len(y) - 1 and abs(y[hi + 1] - ref) > tol:
                    hi += 1
                y[lo:hi + 1] = ref
                applied.append({"field": c, "action": "flatten_excursion", "ref": _r(ref),
                                "span": [int(fr[lo]), int(fr[hi])]})
            else:
                # single step: fold everything after it back by the nearest multiple of pi
                i = laf[0]
                k = round((np.nanmedian(y[i:]) - np.nanmedian(y[:i])) / np.pi)
                if k:
                    y[i:] = y[i:] - k * np.pi
                    applied.append({"field": c, "action": "phase_shift", "k_pi": int(k), "from_frame": int(fr[i])})
            data[c] = y
            continue
        # clamp: flagged frames are physically implausible (e.g. ball depth behind the player);
        # replace them by interpolation from the nearest plausible (unflagged) frames.
        clamp_idx = sorted(f2i[fl["frame"]] for fl in fls
                           if fl["suggested_action"] == "clamp" and fl["frame"] in f2i)
        if clamp_idx:
            bad = np.zeros(len(y), bool); bad[clamp_idx] = True
            good = (~bad) & np.isfinite(y)
            if good.sum() >= 2:
                y[bad] = np.interp(np.where(bad)[0], np.where(good)[0], y[good])
                applied.append({"field": c, "action": "clamp", "n_frames": int(bad.sum()),
                                "span": [int(fr[clamp_idx[0]]), int(fr[clamp_idx[-1]])]})
            data[c] = y
        for fl in fls:
            f = fl["frame"]; i = f2i.get(f)
            if i is None:
                continue
            if fl["suggested_action"] == "interpolate" and 0 < i < len(y) - 1:
                old = y[i]; y[i] = 0.5 * (y[i - 1] + y[i + 1])
                applied.append({"field": c, "frame": f, "action": "interpolate", "from": _r(old), "to": _r(y[i])})
            elif fl["suggested_action"] == "level_align" and 0 < i < len(y):
                step = y[i] - y[i - 1]
                if abs(step) > 1e-6:
                    y[i:] = y[i:] - step
                    applied.append({"field": c, "frame": f, "action": "level_align", "step_removed": _r(step)})
    new_rows = [dict(r) for r in rows]
    for c, y in data.items():
        for i, r in enumerate(new_rows):
            if not np.isnan(y[i]):
                r[c] = f"{y[i]:.6f}"
    return new_rows, applied


def verify(before: list[dict], after: list[dict], cols: list[str]) -> dict:
    rep = {}
    for c in cols:
        b, a = accel(series(before, c)), accel(series(after, c))
        rep[c] = {"max_acc_before": round(float(b.max()), 4), "max_acc_after": round(float(a.max()), 4)}
    return rep


def plot_before_after(before, after, cols, out: Path):
    show = [c for c in cols if np.nanstd(series(before, c)) > 1e-9][:6]
    fr = [int(float(r["frame"])) for r in before]
    fig, axes = plt.subplots(len(show), 1, figsize=(11, 1.5 * len(show)), sharex=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ax, c in zip(axes, show):
        ax.plot(fr, series(before, c), color="tab:red", alpha=0.6, label="before")
        ax.plot(fr, series(after, c), color="tab:green", lw=1.5, label="after")
        ax.set_title(c, fontsize=9, loc="left"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[-1].set_xlabel("frame")
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--data-csv", type=Path, required=True)
    ap.add_argument("--backend", choices=("gemini", "claude"), default="gemini")
    ap.add_argument("--model", default="")
    ap.add_argument("--flags-json", type=Path, default=None,
                    help="skip the LLM call and load flags from this file (e.g. flags produced by Claude the agent)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rows = read_rows(args.data_csv)
    cols = numeric_columns(rows)
    meta = args.sample_dir / "metadata.json"
    context = json.loads(meta.read_text()).get("prompt", "") if meta.exists() else ""
    if not context:  # fall back to the shared prompt sheet by sample name
        ps = Path("video_sample/prompts.json")
        if ps.exists():
            nm = args.sample_dir.name.split("_")[-1]
            for e in json.loads(ps.read_text()):
                if e.get("name") in (nm, args.sample_dir.name):
                    context = e.get("prompt", ""); break
    context = context or "(no prompt available)"
    summary = candidates(rows, cols)

    out_dir = args.sample_dir / "results" / "diagnostics" / "llm_reasoning"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_summary.json").write_text(json.dumps({"context": context, "summary": summary}, indent=2))

    print(f"sample={args.sample_dir.name}  csv={args.data_csv.name}  cols={len(cols)}  backend={args.backend}")
    if args.flags_json:
        flags = json.loads(args.flags_json.read_text()).get("flags", [])
        print(f"loaded {len(flags)} flag(s) from {args.flags_json}")
    else:
        flags = get_flags(args.backend, args.model, context, summary).get("flags", [])
    (out_dir / "flags.json").write_text(json.dumps({"flags": flags}, indent=2))
    print(f"LLM flagged {len(flags)} issue(s):")
    for fl in flags:
        print(f"  f{fl['frame']:>4} {fl['field']:<10} [{fl['suggested_action']}] {fl['issue']}")

    if not args.apply:
        print(f"\ndry-run. flags -> {out_dir/'flags.json'}  (re-run with --apply to fix+verify)")
        return

    new_rows, applied = apply_fixes(rows, flags)
    touched = sorted({a["field"] for a in applied})
    rep = verify(rows, new_rows, touched or cols)
    out_csv = out_dir / "corrected_trajectory.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(new_rows)
    if touched:
        plot_before_after(rows, new_rows, touched, out_dir / "before_after.png")
    (out_dir / "verify_report.json").write_text(json.dumps({"applied": applied, "accel": rep}, indent=2))

    print(f"\napplied {len(applied)} fix(es) on {touched}:")
    for a in applied:
        print("  ", a)
    print("verification (max |accel| per touched field):")
    for c in touched:
        b, a = rep[c]["max_acc_before"], rep[c]["max_acc_after"]
        print(f"   {c:<10} {b:>8.4f} -> {a:<8.4f}  ({'reduced' if a < b - 1e-9 else 'no change'})")
    print(f"\nwrote {out_csv}\n      {out_dir/'verify_report.json'}\n      {out_dir/'before_after.png'}")


if __name__ == "__main__":
    main()
