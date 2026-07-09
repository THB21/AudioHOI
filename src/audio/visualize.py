"""Visual demo — annotate the video frames at each detected audio event.

Builds one PNG: a waveform strip with onsets colored by fused event type, plus a grid
of the event frames each annotated with the object position, the nearest body part, and
the fused semantic label (what / to whom / how-constrained). Makes the semantic sheet
tangible.

  python -m src.audio.visualize --sample-dir samples/basketball_01
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .extract import extract_wav, load_wav

_COLORS = {
    "strike": "#d62728", "bounce": "#1f77b4", "tap": "#ff7f0e", "slide": "#9467bd",
    "roll": "#17becf", "ring": "#bcbd22", "rattle": "#e377c2", "swish": "#2ca02c",
    "none": "#cccccc", "unknown": "#7f7f7f",
}


def _load_sheet(sample_dir: Path, classifier: str) -> list[dict]:
    p = sample_dir / "results" / "audio_semantics" / f"semantic_sheet_{classifier}.csv"
    with p.open() as f:
        return list(csv.DictReader(f))


def _pick(rows: list[dict], k: int) -> list[dict]:
    """Pick up to k events spanning distinct fused types, then by confidence."""
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["fused_event_type"], []).append(r)
    picked, i = [], 0
    keys = list(by_type)
    while len(picked) < k and any(by_type.values()):
        t = keys[i % len(keys)]
        if by_type[t]:
            by_type[t].sort(key=lambda r: -float(r["fused_conf"]))
            picked.append(by_type[t].pop(0))
        i += 1
        if i > 1000:
            break
    return sorted(picked, key=lambda r: int(r["frame"]))


def make_demo(sample_dir: Path, classifier: str = "rule", k: int = 6) -> Path:
    import cv2

    sample_dir = Path(sample_dir)
    rows = _load_sheet(sample_dir, classifier)
    sr, x = load_wav(extract_wav(sample_dir))
    t = np.arange(len(x)) / sr
    sel = _pick(rows, k)

    files = sorted((sample_dir / "frames").glob("*.png")) or sorted((sample_dir / "frames").glob("*.jpg"))

    ncol = 3
    nrow = 1 + int(np.ceil(len(sel) / ncol))
    fig = plt.figure(figsize=(13, 3.2 * nrow))
    gs = fig.add_gridspec(nrow, ncol, height_ratios=[1.1] + [2] * (nrow - 1))

    # --- waveform strip ---
    axw = fig.add_subplot(gs[0, :])
    axw.plot(t, x, color="#444", lw=0.4)
    for r in rows:
        c = _COLORS.get(r["fused_event_type"], "#7f7f7f")
        axw.axvline(float(r["time"]), color=c, alpha=0.85, lw=1.4)
    axw.set_xlim(0, t[-1])
    axw.set_yticks([])
    axw.set_xlabel("time (s)")
    axw.set_title(f"{sample_dir.name} — audio waveform with detected events "
                  f"(classifier={classifier}, {len(rows)} events)")
    handles = [plt.Line2D([0], [0], color=c, lw=3) for c in _COLORS.values()]
    axw.legend(handles, list(_COLORS), ncol=7, fontsize=7, loc="upper right")

    # --- annotated event frames ---
    for i, r in enumerate(sel):
        ax = fig.add_subplot(gs[1 + i // ncol, i % ncol])
        fr = int(r["frame"])
        idx = int(np.clip(fr - 1, 0, len(files) - 1))
        img = cv2.cvtColor(cv2.imread(str(files[idx])), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        c = _COLORS.get(r["fused_event_type"], "#7f7f7f")
        ou, ov = float(r["obj_u"]), float(r["obj_v"])
        if ou > 0 or ov > 0:
            ax.add_patch(plt.Circle((ou, ov), 16, color=c, fill=False, lw=2.5))
        # nearest body part marker (object - part_dist along the line is unknown; mark text)
        part = r["nearest_part"]
        ax.set_title(f"f{fr}  [{r.get('source','?')}]  {r['fused_event_type']} → {r['target_entity']}\n"
                     f"target={r['contact_target']}  conf={float(r['fused_conf']):.2f}  "
                     f"cue={r['visual_cue']}", fontsize=8, color=c)
        ax.text(0.02, 0.04, f"nearest: {part}  d={r['part_dist_px']}px",
                transform=ax.transAxes, fontsize=7, color="white",
                bbox=dict(facecolor="black", alpha=0.5, pad=1))
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    out = sample_dir / "results" / "audio_semantics" / f"demo_{classifier}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def make_loss_timeline(sample_dirs, out_path: Path) -> Path:
    """One figure: per sample, the loss-ready events on a (time × entity) timeline.
    color=event_type, size=manip_weight, ring=promote_anchor, grey ✕=filtered out."""
    import csv as _csv

    ents = ["support", "left_foot", "right_foot", "left_hand", "right_hand", "object", "none"]
    sample_dirs = [Path(s) for s in sample_dirs]
    fig, axes = plt.subplots(len(sample_dirs), 1, figsize=(13, 3.0 * len(sample_dirs)), squeeze=False)
    for ax, sd in zip(axes[:, 0], sample_dirs):
        p = sd / "results" / "audio_semantics" / "semantic_sheet_vlm.csv"
        rows = list(_csv.DictReader(p.open()))
        tmax = max((float(r["refined_time"]) for r in rows), default=1.0)
        for r in rows:
            t = float(r["refined_time"])
            ent = r["target_entity"] if r["target_entity"] in ents else "none"
            y = ents.index(ent)
            et = r["fused_event_type"]
            keep = int(r["vlm_relevant"])
            wt = float(r["manip_weight"])
            if not keep:
                ax.scatter([t], [y], marker="x", c="#bbbbbb", s=40, zorder=2)
                continue
            c = _COLORS.get(et, "#7f7f7f")
            edge = "black" if int(r["promote_anchor"]) else c
            lw = 2.2 if int(r["promote_anchor"]) else 0.6
            ax.scatter([t], [y], c=c, s=60 + 180 * wt, edgecolors=edge, linewidths=lw, alpha=0.9, zorder=3)
            ax.annotate(f"{et}\nγ{r['manip_gamma']} w{r['manip_weight']}", (t, y),
                        fontsize=6, ha="center", va="bottom", xytext=(0, 6),
                        textcoords="offset points", color=c)
            ax.annotate(f"[{r['evidence'][:3]}]", (t, y), fontsize=5, ha="center", va="top",
                        xytext=(0, -8), textcoords="offset points", color="#555")
        ax.set_yticks(range(len(ents)))
        ax.set_yticklabels(ents, fontsize=8)
        ax.set_ylim(-0.6, len(ents) - 0.4)
        ax.set_xlim(-0.2, tmax + 0.4)
        ax.set_xlabel("refined time (s)  —  audio-snapped contact instant")
        ax.grid(axis="y", alpha=0.2)
        kept = sum(int(r["vlm_relevant"]) for r in rows)
        ax.set_title(f"{sd.name} — {kept}/{len(rows)} loss-ready events "
                     f"(what=color/label · where=row · when=x · size=weight · ring=anchor · ✕=filtered)",
                     fontsize=9)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=c, label=k) for k, c in _COLORS.items()]
    axes[0, 0].legend(handles=handles, ncol=10, fontsize=6, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def make_audio_comparison(sample_dirs, out_path: Path) -> Path:
    """Per sample: waveform with color-coded audio events (left) + normalized mean acoustic
    feature signature (right). One row per sample for side-by-side comparison."""
    import csv as _csv

    from .extract import extract_wav, load_wav

    feat_names = ["attack", "decay_ms", "centroid", "flatness", "hf_ratio", "rms", "hpr"]
    norm = {"attack": 1.0, "decay_ms": 150.0, "centroid": 3000.0, "flatness": 1.0,
            "hf_ratio": 1.0, "rms": 1.0, "hpr": 1.0}
    sample_dirs = [Path(s) for s in sample_dirs]
    n = len(sample_dirs)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.6 * n),
                             gridspec_kw={"width_ratios": [4.2, 1.0]}, squeeze=False)
    for i, sd in enumerate(sample_dirs):
        rows = list(_csv.DictReader((sd / "results" / "audio_semantics" / "semantic_sheet_rule.csv").open()))
        aud = [r for r in rows if r["source"] in ("audio", "audio+visual")]
        sr, x = load_wav(extract_wav(sd))
        t = np.arange(len(x)) / sr

        axw = axes[i, 0]
        axw.plot(t[::8], x[::8], color="#555", lw=0.4)
        seen = set()
        for r in aud:
            et = r["audio_event_type"]
            c = _COLORS.get(et, "#7f7f7f")
            axw.axvline(float(r["time"]), color=c, alpha=0.85, lw=1.5,
                        label=et if et not in seen else None)
            seen.add(et)
        axw.set_xlim(0, t[-1]); axw.set_yticks([])
        axw.set_ylabel(sd.name, fontsize=9, fontweight="bold")
        if i == n - 1:
            axw.set_xlabel("time (s)")
        axw.legend(ncol=6, fontsize=7, loc="upper right")
        axw.set_title(f"{sd.name}: {len(aud)} audio events  "
                      f"({', '.join(f'{k}×{v}' for k, v in _count(aud))})", fontsize=9)

        axf = axes[i, 1]
        vals = [np.mean([min(1.0, float(r[k]) / norm[k]) for r in aud]) if aud else 0 for k in feat_names]
        colors = ["#d62728" if k in ("attack", "hf_ratio", "rms") else "#1f77b4" for k in feat_names]
        axf.barh(range(len(feat_names)), vals, color=colors, alpha=0.8)
        axf.set_yticks(range(len(feat_names)))
        axf.set_yticklabels([k.replace("_ms", "").replace("_ratio", "") for k in feat_names], fontsize=7)
        axf.set_xlim(0, 1); axf.invert_yaxis()
        axf.set_title("acoustic signature\n(normalized mean)", fontsize=7)
        for sp in ("top", "right"):
            axf.spines[sp].set_visible(False)
    fig.suptitle("Audio extraction comparison — sharp/short/loud = hard impact · dull/long/quiet = soft contact",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def _count(rows):
    from collections import Counter
    return Counter(r["audio_event_type"] for r in rows).most_common()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    ap.add_argument("--classifier", default="rule")
    ap.add_argument("-k", type=int, default=6)
    args = ap.parse_args()
    make_demo(args.sample_dir, args.classifier, args.k)


if __name__ == "__main__":
    main()
