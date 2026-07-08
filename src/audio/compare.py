"""Comparison harness — run every detector × classifier on every sample with audio.

No human labels, so "what works best" is scored by cross-modal + cross-method agreement:
  - detector overlap          how much the 3 detectors agree on onset timing
  - grounding rate            fraction of onsets with a corroborating visual cue
  - classifier agreement      pairwise label agreement on the same (combined) onsets
  - cluster silhouette / AST confidence
Writes a per-sample report and an aggregate report (analysis/audio_compare_report.md) so we
can see whether the ranking generalizes across examples.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from . import classify, detect, features as featmod
from .extract import extract_wav, load_wav
from .visual_context import build_visual_contexts


def _samples_with_audio(root: Path = Path("samples")) -> list[Path]:
    out = []
    for d in sorted(root.glob("*")):
        if (d / "audio.wav").exists() or (d / "video.mp4").exists():
            out.append(d)
    return out


def _n_frames(sample_dir: Path) -> int:
    fdir = sample_dir / "frames"
    return len(list(fdir.glob("*.png")) or list(fdir.glob("*.jpg")))


def _match_count(a: list[float], b: list[float], tol: float) -> int:
    bb = sorted(b)
    used = [False] * len(bb)
    m = 0
    for t in a:
        for j, u in enumerate(bb):
            if not used[j] and abs(u - t) <= tol:
                used[j] = True
                m += 1
                break
    return m


def analyze_sample(sample_dir: Path, fps: float = 0.0) -> dict:
    wav = extract_wav(sample_dir)
    sr, x = load_wav(wav)
    n_frames = _n_frames(sample_dir) or (int(len(x) / sr * fps) + 1)
    tol = 1.5 / fps  # onset match tolerance ≈ 1.5 frames

    # detectors
    det_onsets = {name: detect.detect(x, name) for name in detect.DETECTORS}
    det_onsets["combined"] = detect.detect(x, "combined")
    det_times = {k: [o.time_s for o in v] for k, v in det_onsets.items()}
    det_counts = {k: len(v) for k, v in det_times.items()}
    # pairwise detector overlap (fraction of the smaller set that matches)
    names = list(detect.DETECTORS)
    overlap = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = det_times[names[i]], det_times[names[j]]
            denom = min(len(a), len(b)) or 1
            overlap[f"{names[i]}∩{names[j]}"] = round(_match_count(a, b, tol) / denom, 2)

    # canonical event set = combined
    onsets = det_onsets["combined"]
    times = [o.time_s for o in onsets]
    frames = [int(np.clip(round(t * fps) + 1, 1, n_frames)) for t in times]
    feats = featmod.compute_features(x, times, sr)
    vctx = build_visual_contexts(sample_dir, frames, n_frames, fps=fps)
    grounded = sum(1 for vc in vctx if vc.visual_cue != "none")
    grounding_rate = round(grounded / max(1, len(vctx)), 2)
    cue_counts = dict(Counter(vc.visual_cue for vc in vctx))

    # classifiers on the same onsets
    labels = {"rule": [l.event_type for l in classify.rule_classify(feats, frames)],
              "cluster": [l.event_type for l in classify.cluster_classify(feats, frames)]}
    sil = None
    cl = classify.cluster_classify(feats, frames)
    if cl and "silhouette" in cl[0].extra:
        sil = cl[0].extra["silhouette"]
    ast_labels = classify.pretrained_classify(x, times, sr, feats, frames)
    if ast_labels is not None:
        labels["pretrained"] = [l.event_type for l in ast_labels]

    # pairwise classifier agreement
    agree = {}
    keys = list(labels)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = labels[keys[i]], labels[keys[j]]
            agree[f"{keys[i]}~{keys[j]}"] = round(np.mean([x == y for x, y in zip(a, b)]), 2) if a else 0.0

    dist = {k: dict(Counter(v)) for k, v in labels.items()}
    return {
        "sample": sample_dir.name, "n_events": len(onsets),
        "det_counts": det_counts, "det_overlap": overlap,
        "grounding_rate": grounding_rate, "cue_counts": cue_counts,
        "label_dist": dist, "classifier_agreement": agree,
        "silhouette": sil, "ast_available": ast_labels is not None,
    }


def _fmt(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def run_comparison(fps: float = 0.0) -> None:
    samples = _samples_with_audio()
    results = [analyze_sample(s, fps=fps) for s in samples]

    lines = ["# Audio→semantic approach comparison\n",
             f"samples: {', '.join(r['sample'] for r in results)}  |  AST available: "
             f"{any(r['ast_available'] for r in results)}\n"]
    for r in results:
        lines += [
            f"\n## {r['sample']}  ({r['n_events']} events)",
            f"- detector counts: {_fmt(r['det_counts'])}",
            f"- detector overlap: {_fmt(r['det_overlap'])}",
            f"- visual grounding rate: **{r['grounding_rate']}**  (cues: {_fmt(r['cue_counts'])})",
            f"- classifier label dist: " + " | ".join(f"{k}: {_fmt(v)}" for k, v in r["label_dist"].items()),
            f"- classifier agreement: {_fmt(r['classifier_agreement'])}",
            f"- cluster silhouette: {r['silhouette']}",
        ]

    # aggregate generalization view
    lines.append("\n## Aggregate (generalization)")
    all_agree_keys = set().union(*[set(r["classifier_agreement"]) for r in results]) if results else set()
    for key in sorted(all_agree_keys):
        vals = [r["classifier_agreement"][key] for r in results if key in r["classifier_agreement"]]
        if vals:
            lines.append(f"- {key} agreement across samples: mean={round(float(np.mean(vals)),2)} "
                         f"(min={min(vals)}, max={max(vals)})")
    gr = [r["grounding_rate"] for r in results]
    lines.append(f"- grounding rate across samples: mean={round(float(np.mean(gr)),2)} "
                 f"(min={min(gr)}, max={max(gr)})")
    lines.append("\nInterpretation: a higher grounding rate + higher cross-sample classifier "
                 "agreement = a more trustworthy, generalizable event source for the loss.")

    report = "\n".join(lines) + "\n"
    out = Path("analysis") / "audio_compare_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    for r in results:
        per = Path("samples") / r["sample"] / "results" / "audio_semantics" / "compare_report.md"
        per.parent.mkdir(parents=True, exist_ok=True)
        per.write_text(report)
    print(report)
    print(f"\nwrote {out}")
