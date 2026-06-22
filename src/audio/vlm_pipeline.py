"""VLM analysis pipeline — windows + audio-refined timing + relevance gating.

  python -m src.audio.vlm_pipeline --sample-dir samples/basketball_01            # heuristic backend
  python -m src.audio.vlm_pipeline --sample-dir samples/basketball_01 --backend moondream

For each audio/visual event: cut a before→event→after frame window (saved as the VLM input),
let the backend analyze what happens + impact, refine the timing with the local audio peak,
and gate on relevance (drop audio that is not a real interaction; keep silent visual contacts).
Writes ``semantic_sheet_vlm.csv`` + ``vlm_windows/*.png`` + ``vlm_summary.md``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from . import vlm
from .extract import extract_wav, load_wav
from .pipeline import run_sample

_COLS = ["frame", "refined_frame", "time", "refined_time", "evidence", "audio_support",
         "fused_event_type", "interaction_mode", "contact_quality", "contact_target",
         "target_entity", "vlm_relevant", "vlm_interaction", "vlm_impact", "vlm_entities",
         "vlm_confidence", "vlm_hand_side", "vlm_object_part", "vlm_object_region",
         "vlm_grasp_type", "vlm_primary_finger", "vlm_fingers",
         "manip_gamma", "manip_weight", "promote_anchor", "vlm_description", "window_png"]


def run_vlm(sample_dir: Path, *, classifier: str = "rule", backend: str = "heuristic",
            model_path: str | None = None, fps: float = 24.0, write: bool = True):
    sample_dir = Path(sample_dir)
    records = run_sample(sample_dir, classifier=classifier, fps=fps, write=False)
    sr, x = load_wav(extract_wav(sample_dir))
    be = vlm.get_backend(backend, model_path)
    win_dir = sample_dir / "results" / "audio_semantics" / "vlm_windows"

    rows = []
    for fe, feat, vc in records:
        obj_uv = (vc.obj_u, vc.obj_v) if (vc.obj_u or vc.obj_v) else None
        png = win_dir / f"event_{fe.frame:04d}.png"
        montage, _ = vlm.extract_window(sample_dir, fe.frame, obj_uv=obj_uv, out_path=png)
        rframe, rtime, support = vlm.refine_timing(x, sr, fe.frame, fps)

        ctx = {"fused_event_type": fe.fused_event_type, "target_entity": fe.target_entity,
               "contact_target": fe.contact_target, "manip_weight": fe.manip_weight,
               "fused_conf": fe.fused_conf, "visual_cue": vc.visual_cue, "audio_support": support}
        an = be.analyze(montage, ctx) if montage is not None else \
            vlm.VLMAnalysis(False, "none", [], "none", "no frames", 0.0)

        # audio refines TIMING for silent visual contacts (audio is temporally precise)
        refined_frame = rframe if (fe.source == "visual" and support >= 0.10) else fe.frame
        # relevance gate: drop audio with no visual corroboration AND weak energy AND VLM says nothing
        relevant = bool(an.relevant and not (
            fe.source == "audio" and vc.visual_cue == "none" and support < 0.10))

        rows.append({
            "frame": fe.frame, "refined_frame": refined_frame,
            "time": round(fe.time, 4), "refined_time": round(rtime, 4),
            "evidence": fe.source, "audio_support": round(support, 3),
            "fused_event_type": fe.fused_event_type, "interaction_mode": fe.interaction_mode,
            "contact_quality": fe.contact_quality, "contact_target": fe.contact_target,
            "target_entity": fe.target_entity, "vlm_relevant": int(relevant),
            "vlm_interaction": an.interaction, "vlm_impact": an.impact,
            "vlm_entities": "|".join(an.entities), "vlm_confidence": round(an.confidence, 3),
            "vlm_hand_side": an.extra.get("hand_side", ""), "vlm_object_part": an.extra.get("object_part", ""),
            "vlm_object_region": an.extra.get("object_region", ""), "vlm_grasp_type": an.extra.get("grasp_type", ""),
            "vlm_primary_finger": an.extra.get("primary_finger", ""),
            "vlm_fingers": "|".join(an.extra.get("contact_fingers", []) or []),
            "manip_gamma": fe.manip_gamma, "manip_weight": fe.manip_weight,
            "promote_anchor": int(fe.promote_anchor), "vlm_description": an.description,
            "window_png": str(png.relative_to(sample_dir)),
        })

    if write:
        out_dir = sample_dir / "results" / "audio_semantics"
        csv_path = out_dir / "semantic_sheet_vlm.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        _write_summary(out_dir / "vlm_summary.md", sample_dir.name, backend, rows)
        kept = sum(r["vlm_relevant"] for r in rows)
        print(f"[{sample_dir.name}] vlm({backend}): {len(rows)} events, {kept} relevant "
              f"→ {csv_path}  (+{len(rows)} window montages)")
    return rows


def _write_summary(path: Path, sample: str, backend: str, rows: list[dict]) -> None:
    from collections import Counter

    kept = [r for r in rows if r["vlm_relevant"]]
    dropped = [r for r in rows if not r["vlm_relevant"]]
    lines = [f"# VLM event analysis — {sample}  (backend={backend})\n",
             f"{len(rows)} candidate events · **{len(kept)} relevant** · {len(dropped)} filtered out\n",
             f"- evidence: {dict(Counter(r['evidence'] for r in kept))}",
             f"- interactions: {dict(Counter(r['fused_event_type'] for r in kept))}",
             f"- impact: {dict(Counter(r['vlm_impact'] for r in kept))}\n",
             "| frame→refined | evidence | event→entity | impact | audio | relevant | description |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        arrow = f"{r['frame']}" if r['frame'] == r['refined_frame'] else f"{r['frame']}→{r['refined_frame']}"
        lines.append(f"| {arrow} | {r['evidence']} | {r['fused_event_type']}→{r['target_entity']} | "
                     f"{r['vlm_impact']} | {r['audio_support']:.2f} | {'keep' if r['vlm_relevant'] else 'drop'} "
                     f"| {r['vlm_description']} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--classifier", default="rule")
    ap.add_argument("--backend", default="heuristic", help="heuristic | moondream")
    ap.add_argument("--vlm-model", default=None, help="local model path for the VLM backend")
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()
    run_vlm(args.sample_dir, classifier=args.classifier, backend=args.backend,
            model_path=args.vlm_model, fps=args.fps)


if __name__ == "__main__":
    main()
