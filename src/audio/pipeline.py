"""Orchestrator — run the full audio→semantic pipeline for one sample and write the sheet.

  python -m src.audio.pipeline --sample-dir samples/basketball_01 --classifier rule
  python -m src.audio.pipeline --sample-dir samples/basketball_01 --classifier pretrained
  python -m src.audio.pipeline --compare           # all samples, all approaches → report
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import classify, detect, features as featmod, seed
from .extract import detect_fps, extract_wav, load_wav
from .fuse import FusedEvent, fuse
from .visual_context import VisualContext, build_visual_contexts


def _n_frames(sample_dir: Path) -> int:
    fdir = sample_dir / "frames"
    files = list(fdir.glob("*.png")) or list(fdir.glob("*.jpg"))
    return len(files)


def run_sample(sample_dir: Path, *, detector: str = "combined", classifier: str = "rule",
               fps: float = 0.0, min_gap_s: float = 0.12, write: bool = True):
    """Run detect→features→classify→visual→fuse for one sample. Returns the records.

    ``fps=0`` auto-detects from video.mp4 — onset-time→frame conversion breaks on
    clips whose rate differs from a hardcoded default (football_10 is 30 fps).
    """
    sample_dir = Path(sample_dir)
    fps = fps or detect_fps(sample_dir)
    wav = extract_wav(sample_dir)
    sr, x = load_wav(wav)

    n_frames = _n_frames(sample_dir) or (int(len(x) / sr * fps) + 1)

    # seed events from BOTH modalities: audio onsets (loud contacts) + visual hand-object
    # proximity minima (silent contacts the mic never heard).
    onsets = detect.detect(x, detector, min_gap_s=min_gap_s)
    vcontacts = seed.detect_visual_contacts(sample_dir, n_frames, fps=fps)
    seeds = seed.merge_seeds(onsets, vcontacts, n_frames, fps=fps, tol_s=min_gap_s)

    times = [s.time for s in seeds]
    frames = [s.frame for s in seeds]

    feats = featmod.compute_features(x, times, sr)

    # audio label source
    if classifier == "rule":
        labels = classify.rule_classify(feats, frames)
    elif classifier == "cluster":
        labels = classify.cluster_classify(feats, frames)
    elif classifier == "pretrained":
        labels = classify.pretrained_classify(x, times, sr, feats, frames)
        if labels is None:
            print(f"[pretrained] AST unavailable ({classify.ast_error()[:80]}); falling back to rule")
            labels = classify.rule_classify(feats, frames)
    else:
        raise ValueError(f"unknown classifier {classifier!r}")

    vctx = build_visual_contexts(sample_dir, frames, n_frames, fps=fps)

    records = []
    for sd, fr, t, f, lab, vc in zip(seeds, frames, times, feats, labels, vctx):
        fe = fuse(fr, t, sd.detector, sd.score, lab, vc,
                  source=sd.source, kind=sd.kind, entity_hint=sd.entity_hint)
        records.append((fe, f, vc))

    if write:
        out_dir = sample_dir / "results" / "audio_semantics"
        write_sheet(out_dir / f"semantic_sheet_{classifier}.csv", records)
        write_sheet_json(out_dir / f"semantic_sheet_{classifier}.json", records, sample_dir.name, classifier)
        print(f"[{sample_dir.name}] {classifier}: {len(records)} events → {out_dir}/semantic_sheet_{classifier}.csv")
    return records


# Unified semantic sheet I/O

_FUSED_COLS = ["frame", "time", "source", "detector", "audio_score", "audio_event_type", "audio_conf",
               "fused_event_type", "interaction_mode", "contact_quality", "contact_constraint",
               "allow_slide", "fused_conf", "contact_target", "target_entity",
               "manip_gamma", "manip_weight", "promote_anchor"]
_VC_COLS = ["obj_u", "obj_v", "obj_speed", "obj_accel", "vel_reversal", "nearest_part",
            "part_dist_px", "part_speed", "flow_mag", "proximity_min", "flow_spike", "visual_cue"]


def _row(fe: FusedEvent, f: featmod.EventFeatures, vc: VisualContext) -> dict:
    d = asdict(fe)
    d["promote_anchor"] = int(d["promote_anchor"])
    d["allow_slide"] = int(d["allow_slide"])
    for k, v in vc.as_dict().items():
        d[k] = v
    for k, v in f.as_dict().items():
        d[k] = v
    return d


def write_sheet(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = _FUSED_COLS + _VC_COLS + featmod.EventFeatures.names()
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for fe, f, vc in records:
            row = _row(fe, f, vc)
            w.writerow({k: (f"{row[k]:.5f}" if isinstance(row[k], float) else row[k]) for k in cols})


def write_sheet_json(path: Path, records, sample: str, classifier: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sample": sample, "classifier": classifier, "n_events": len(records),
               "events": [_row(fe, f, vc) for fe, f, vc in records]}
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, default=None)
    ap.add_argument("--detector", default="combined", choices=list(detect.DETECTORS) + ["combined"])
    ap.add_argument("--classifier", default="rule", choices=list(classify.CLASSIFIERS))
    ap.add_argument("--fps", type=float, default=0.0, help="0 = auto-detect from video.mp4")
    ap.add_argument("--compare", action="store_true", help="run all samples × approaches → report")
    args = ap.parse_args()

    if args.compare:
        from .compare import run_comparison
        run_comparison(fps=args.fps)
        return
    if args.sample_dir is None:
        ap.error("provide --sample-dir or --compare")
    run_sample(args.sample_dir, detector=args.detector, classifier=args.classifier, fps=args.fps)


if __name__ == "__main__":
    main()
