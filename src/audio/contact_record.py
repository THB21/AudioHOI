"""Unified contact record — fuses our audio/visual events with the teammate's object-side
contact (object-local point + semantic part) and a persistence state machine.

One schema both pipelines feed, so a single loss reads it for any object:
  WHEN   frame, time, refined_time, source, evidence
  WHAT   event_type, interaction_mode, contact_quality, impact, relevant
  WHERE  target_entity, contact_target, contact_constraint,
         object_local_x/y/z, semantic_contact_part            ← adopted from teammate
  HOLD   contact_state, interval_id, stable_entity            ← adopted state machine
  LOSS   manip_gamma, manip_weight, promote_anchor, allow_slide, audio_support, confidence

Object-side fields come from her ``mug_articraft_contact_points.csv`` (per-frame, object-local)
and her ``mug_grasp_anchor_state.csv`` (per-frame grasp mode) when present; otherwise they are
empty and the generic persistence machine runs over our events.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from . import persistence
from .pipeline import run_sample
from .vlm_pipeline import run_vlm

_COLS = ["frame", "time", "refined_time", "source", "evidence",
         "event_type", "interaction_mode", "contact_quality", "impact", "relevant",
         "target_entity", "contact_target", "contact_constraint",
         "object_local_x", "object_local_y", "object_local_z", "semantic_contact_part",
         "vlm_hand_part", "vlm_handle_contact", "vlm_rim_contact", "vlm_constraint", "vlm_object_conf",
         "contact_state", "interval_id", "stable_entity",
         "manip_gamma", "manip_weight", "promote_anchor", "allow_slide",
         "audio_support", "confidence"]


def _load_csv(path: Path) -> list[dict]:
    if path and Path(path).exists():
        with Path(path).open() as f:
            return list(csv.DictReader(f))
    return []


def _nearest(frame: int, by_frame: dict[int, dict], tol: int = 3) -> dict:
    if frame in by_frame:
        return by_frame[frame]
    cands = [f for f in by_frame if abs(f - frame) <= tol]
    return by_frame[min(cands, key=lambda f: abs(f - frame))] if cands else {}


def build_records(sample_dir: Path, *, classifier: str = "rule", backend: str = "heuristic",
                  object_contact_csv: Path | None = None, grasp_state_csv: Path | None = None,
                  fps: float = 24.0, write: bool = True):
    sample_dir = Path(sample_dir)
    res = sample_dir / "results"

    # our events (use the VLM sheet if a real backend was requested, else the fused sheet)
    if backend not in ("none",):
        ev = run_vlm(sample_dir, classifier=classifier, backend=backend, fps=fps, write=False)
        events = [{"frame": int(r["frame"]), "time": float(r["time"]),
                   "refined_time": float(r["refined_time"]), "source": r["evidence"],
                   "event_type": r["fused_event_type"], "interaction_mode": r["interaction_mode"],
                   "contact_quality": r["contact_quality"], "impact": r["vlm_impact"],
                   "relevant": int(r["vlm_relevant"]), "target_entity": r["target_entity"],
                   "contact_target": r["contact_target"], "audio_support": float(r["audio_support"]),
                   "manip_gamma": float(r["manip_gamma"]), "manip_weight": float(r["manip_weight"]),
                   "promote_anchor": int(r["promote_anchor"]), "confidence": float(r["vlm_confidence"])}
                  for r in ev]
    else:
        recs = run_sample(sample_dir, classifier=classifier, fps=fps, write=False)
        events = [{"frame": fe.frame, "time": fe.time, "refined_time": fe.time, "source": fe.source,
                   "event_type": fe.fused_event_type, "interaction_mode": fe.interaction_mode,
                   "contact_quality": fe.contact_quality, "impact": "n/a", "relevant": 1,
                   "target_entity": fe.target_entity, "contact_target": fe.contact_target,
                   "audio_support": 0.0, "manip_gamma": fe.manip_gamma, "manip_weight": fe.manip_weight,
                   "promote_anchor": int(fe.promote_anchor), "confidence": fe.fused_conf}
                  for fe, _, _ in recs]

    constraint = {"impulsive": "point", "repetitive": "point", "resonant": "point",
                  "continuous": "interval", "none": "none"}

    # --- adopt teammate object-side contact (object-local point + semantic part) ---
    oc = object_contact_csv or (res / "mug_articraft_contact_points" / "mug_articraft_contact_points.csv")
    obj_rows = _load_csv(oc)
    obj_by_frame = {int(float(r["frame"])): r for r in obj_rows if r.get("frame")}

    # --- adopt teammate grasp-anchor state machine (per-frame mode) ---
    gs = grasp_state_csv or (res / "mug_grasp_anchor_state" / "mug_grasp_anchor_state.csv")
    gs_rows = _load_csv(gs)
    gs_by_frame = {int(float(r["frame"])): r for r in gs_rows if r.get("frame")}

    # persistence: her per-frame state if present, else our generic machine over our events
    if gs_by_frame:
        persist = []
        interval = -1
        prev = None
        for e in sorted(events, key=lambda r: r["frame"]):
            g = _nearest(e["frame"], gs_by_frame)
            state = persistence.from_teammate_state(g.get("frame_mode", "")) if g else "no_contact"
            if state == "direct_contact" and prev != "direct_contact":
                interval += 1
            persist.append(persistence.Persistence(state, interval if state in ("direct_contact", "keep_grasp") else -1,
                                                    g.get("stable_grasp_hand_side", "") or e["target_entity"]))
            prev = state
        pmap = {id(e): p for e, p in zip(sorted(events, key=lambda r: r["frame"]), persist)}
    else:
        plist = persistence.assign_persistence(events)
        pmap = {id(e): p for e, p in zip(sorted(events, key=lambda r: r["frame"]), plist)}

    rows = []
    for e in events:
        p = pmap[id(e)]
        o = _nearest(e["frame"], obj_by_frame)
        rows.append({
            "frame": e["frame"], "time": round(e["time"], 4), "refined_time": round(e["refined_time"], 4),
            "source": e["source"], "evidence": e["source"], "event_type": e["event_type"],
            "interaction_mode": e["interaction_mode"], "contact_quality": e["contact_quality"],
            "impact": e["impact"], "relevant": e["relevant"], "target_entity": e["target_entity"],
            "contact_target": e["contact_target"], "contact_constraint": constraint.get(e["interaction_mode"], "point"),
            "object_local_x": o.get("mug_local_x", ""), "object_local_y": o.get("mug_local_y", ""),
            "object_local_z": o.get("mug_local_z", ""),
            "semantic_contact_part": o.get("semantic_contact_part", o.get("nearest_articraft_part", "")),
            "vlm_hand_part": o.get("vlm_hand_contact_part", ""), "vlm_handle_contact": o.get("vlm_handle_contact", ""),
            "vlm_rim_contact": o.get("vlm_rim_contact", ""), "vlm_constraint": o.get("vlm_constraint", ""),
            "vlm_object_conf": o.get("vlm_confidence", ""),
            "contact_state": p.contact_state, "interval_id": p.interval_id, "stable_entity": p.stable_entity,
            "manip_gamma": e["manip_gamma"], "manip_weight": e["manip_weight"],
            "promote_anchor": e["promote_anchor"], "allow_slide": int(e["contact_quality"] == "friction"),
            "audio_support": round(e["audio_support"], 3), "confidence": round(e["confidence"], 3),
        })
    rows.sort(key=lambda r: r["frame"])

    if write:
        out = res / "audio_semantics" / "contact_records.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        n_obj = sum(1 for r in rows if r["object_local_x"] != "")
        n_grasp = sum(1 for r in rows if r["contact_state"] in ("direct_contact", "keep_grasp"))
        print(f"[{sample_dir.name}] {len(rows)} contact records → {out}  "
              f"({n_obj} with object-local point, {n_grasp} in a grasp interval)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--classifier", default="rule")
    ap.add_argument("--backend", default="heuristic", help="heuristic | qwen | none")
    ap.add_argument("--object-contact-csv", type=Path, default=None)
    ap.add_argument("--grasp-state-csv", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()
    build_records(args.sample_dir, classifier=args.classifier, backend=args.backend,
                  object_contact_csv=args.object_contact_csv, grasp_state_csv=args.grasp_state_csv,
                  fps=args.fps)


if __name__ == "__main__":
    main()
