#!/usr/bin/env python3
"""LLM/VLM correction audit — final sanity gate over the 4D HOI reconstruction.

Runs at the END of the pipeline (after the contact-phase solver and the full-scene
render) and answers "does the reconstruction hold together, and if not, which
bounded re-run fixes it". Two parts:

1. Deterministic table audits (no model) over the contact-phase trajectory,
   the audio-semantic contact records and the DA3 object-depth table:
     depth_outlier           |Δtz| > --depth-jump-m between consecutive frames
     boundary_drift          tz std over the first/last 10 frames > --boundary-std-m
                             (solver boundary frames have no anchor on one side)
     contact_gap_violation   at anchor frames (contact_frame==1 or audio_anchor==1)
                             |tz − active_part_z| > --contact-gap-m
     part_flip               active_part switches side (left↔right) inside one
                             contiguous human_contact_state==1 interval
     audio_visual_mismatch   relevant audio records with audio_support >= 0.4 that
                             have NO anchor (contact_frame/audio_anchor) within
                             ±--mismatch-window frames — audio heard, solver ignored
     low_conf_unsupported    runs of >= --low-conf-run frames with depth_conf <
                             --low-conf AND audio_support == 0 — the solver is
                             blindly interpolating there; verify visually

2. VLM render check: samples up to 6 frames from the full-scene overlay render
   (first, last, up to 4 contact anchors spread over the clip), saves them as PNGs
   for human review, and — when the qwen backend loads — runs src.audio.vlm's
   QwenVLBackend on each. NB the backend's prompt is FIXED (contact/part/impact
   forced-choice JSON), not free-form, so at contact frames we can only check that
   the VLM sees a contact consistent with the solver's active part (pass/fail/
   unclear); boundary frames stay 'for_human_review'. If the backend fails to
   load, all sampled frames are marked 'for_human_review' with a note.

Outputs (results/llm_correction/ unless --out-dir):
  llm_correction_report.md   human-readable findings + VLM results + PASS/WARN/FAIL
  corrections.json           bounded RE-RUN RECIPES, never raw pose edits
  frames/frame_NNNN.png      the sampled render frames

Recipe vocabulary (kept deliberately small):
  {"type": "rerun_solver", "flags": {"--w-temp": 8.0}}
      depth_outlier / boundary_drift → re-run the contact-phase solver with a
      stronger temporal smoothness weight.
  {"type": "demote_anchor", "frame": N}
      contact_gap_violation at an audio anchor whose audio_support < --demote-support:
      the anchor is weakly supported and pins the object away from the part —
      drop its promotion and re-solve.
  {"type": "rerun_body_refine", "flags": {}}
      penetration-type contact gap (object center closer to camera than the
      active part, i.e. the part likely sits inside the object) → re-run
      refine_body_pose_contact.py.
  {"type": "visual_review", "frames": [...]}
      low_conf_unsupported / part_flip / audio_visual_mismatch / remaining
      contact gaps — a human (or a stronger VLM) must look before anything reruns.

Verdict: FAIL when high-severity findings (depth_outlier, contact_gap_violation)
cover > 2% of frames or number >= 5; WARN on any finding or any VLM 'fail';
PASS otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]

SEVERITY = {
    "depth_outlier": "high",
    "contact_gap_violation": "high",
    "boundary_drift": "medium",
    "part_flip": "medium",
    "audio_visual_mismatch": "medium",
    "low_conf_unsupported": "low",
}


def read_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(row: dict, key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def frames_to_ranges(frames: list[int]) -> str:
    """Compact '3-5, 12, 40-41' rendering of a sorted frame list."""
    if not frames:
        return "-"
    frames = sorted(set(frames))
    spans, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        spans.append((start, prev))
        start = prev = f
    spans.append((start, prev))
    return ", ".join(f"{a}-{b}" if b > a else f"{a}" for a, b in spans)


def _side(part: str) -> str:
    return "left" if part.startswith("left") else ("right" if part.startswith("right") else "")


# Part 1 — deterministic table audits

def run_table_audits(traj: list[dict], records: list[dict] | None, depth: list[dict] | None,
                     args: argparse.Namespace) -> list[dict]:
    """Each finding: {label, severity, frames, evidence}."""
    findings: list[dict] = []
    frames = [int(r["frame"]) for r in traj]
    tz = np.array([_f(r, "tz") for r in traj])
    n = len(traj)

    # depth_outlier — flag BOTH endpoints of each violating transition
    bad: list[int] = []
    jumps: list[str] = []
    for i in range(n - 1):
        d = abs(tz[i + 1] - tz[i])
        if d > args.depth_jump_m:
            bad += [frames[i], frames[i + 1]]
            jumps.append(f"f{frames[i]}→f{frames[i + 1]}: Δtz={d:.2f} m")
    if bad:
        findings.append({"label": "depth_outlier", "frames": sorted(set(bad)),
                         "evidence": "; ".join(jumps[:8]) + (" …" if len(jumps) > 8 else "")})

    # boundary_drift — first/last 10 frames tz std
    k = min(10, n)
    for name, idx in (("first", slice(0, k)), ("last", slice(n - k, n))):
        s = float(np.std(tz[idx]))
        if s > args.boundary_std_m:
            findings.append({"label": "boundary_drift", "frames": frames[idx],
                             "evidence": f"{name} {k} frames tz std {s:.3f} m > {args.boundary_std_m} m"})

    # contact_gap_violation — anchor frames where object depth misses the part depth
    gap_bad, gap_ev = [], []
    for r in traj:
        if not (_i(r, "contact_frame") == 1 or _i(r, "audio_anchor") == 1):
            continue
        pz = _f(r, "active_part_z")
        if not np.isfinite(pz):
            continue
        gap = _f(r, "tz") - pz
        if abs(gap) > args.contact_gap_m:
            gap_bad.append(int(r["frame"]))
            gap_ev.append({"frame": int(r["frame"]), "gap_m": round(gap, 3),
                           "part": r.get("active_part", ""),
                           "audio_anchor": _i(r, "audio_anchor") == 1,
                           "audio_support": _f(r, "audio_support", 0.0)})
    if gap_bad:
        ev = "; ".join(f"f{e['frame']}: {e['gap_m']:+.3f} m vs {e['part']}" for e in gap_ev[:8])
        findings.append({"label": "contact_gap_violation", "frames": gap_bad,
                         "evidence": ev, "details": gap_ev})

    # part_flip — side change inside a contiguous human_contact_state==1 interval
    flip_frames, flip_ev = [], []
    run_side, run_start = "", None
    for r in traj:
        if _i(r, "human_contact_state") == 1:
            side = _side(str(r.get("active_part", "")))
            if run_start is None:
                run_start, run_side = int(r["frame"]), side
            elif side and run_side and side != run_side:
                flip_frames.append(int(r["frame"]))
                flip_ev.append(f"f{int(r['frame'])}: {run_side}→{side} (interval from f{run_start})")
                run_side = side
        else:
            run_start, run_side = None, ""
    if flip_frames:
        findings.append({"label": "part_flip", "frames": flip_frames, "evidence": "; ".join(flip_ev[:8])})

    # audio_visual_mismatch — supported audio records with no nearby anchor
    if records is not None:
        anchor_frames = {int(r["frame"]) for r in traj
                         if _i(r, "contact_frame") == 1 or _i(r, "audio_anchor") == 1}
        miss, miss_ev = [], []
        for rec in records:
            if _i(rec, "relevant") != 1 or _f(rec, "audio_support", 0.0) < 0.4:
                continue
            f = int(rec["frame"])
            if not any(f + d in anchor_frames for d in range(-args.mismatch_window, args.mismatch_window + 1)):
                miss.append(f)
                miss_ev.append(f"f{f}: {rec.get('event_type', '?')} "
                               f"support={_f(rec, 'audio_support', 0.0):.2f}, no anchor ±{args.mismatch_window}")
        if miss:
            findings.append({"label": "audio_visual_mismatch", "frames": miss, "evidence": "; ".join(miss_ev[:8])})

    # low_conf_unsupported — long blind-interpolation runs
    if depth is not None:
        conf = {int(r["frame"]): _f(r, "depth_conf", 1.0) for r in depth}
        support = {int(r["frame"]): _f(r, "audio_support", 0.0) for r in traj}
        blind = [f for f in frames if conf.get(f, 1.0) < args.low_conf and support.get(f, 0.0) == 0.0]
        runs, cur = [], []
        for f in blind:
            if cur and f == cur[-1] + 1:
                cur.append(f)
            else:
                if len(cur) >= args.low_conf_run:
                    runs.append(cur)
                cur = [f]
        if len(cur) >= args.low_conf_run:
            runs.append(cur)
        if runs:
            all_frames = [f for run in runs for f in run]
            findings.append({"label": "low_conf_unsupported", "frames": all_frames,
                             "evidence": f"{len(runs)} run(s) of depth_conf<{args.low_conf} with no audio "
                                         f"support — blind interpolation, verify visually"})

    for fd in findings:
        fd["severity"] = SEVERITY[fd["label"]]
    return findings


# Part 2 — VLM render check

def sample_render_frames(video_path: Path, traj: list[dict], max_anchors: int = 4) -> list[dict]:
    """First frame, last frame and up to `max_anchors` contact frames spread over the clip.
    Returns [{frame, kind, expected_part, image(BGR)}]."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    part_by_frame = {int(r["frame"]): str(r.get("active_part", "") or "") for r in traj}
    contacts = [int(r["frame"]) for r in traj if _i(r, "contact_frame") == 1]
    picks: dict[int, str] = {1: "boundary_first", n_video: "boundary_last"}
    if contacts:
        idx = np.unique(np.linspace(0, len(contacts) - 1, min(max_anchors, len(contacts))).round().astype(int))
        picks.update({contacts[i]: "contact" for i in idx if contacts[i] <= n_video})  # contact wins on ties
    out = []
    for frame, kind in sorted(picks.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
        ok, img = cap.read()
        if not ok:
            continue
        out.append({"frame": frame, "kind": kind, "expected_part": part_by_frame.get(frame, ""), "image": img})
    cap.release()
    return out


def run_vlm_check(samples: list[dict], out_dir: Path, backend_name: str, model_path: str | None) -> tuple[list[dict], str]:
    """Save sampled frames as PNGs; judge contact frames with the qwen backend when it loads.
    Returns (results, backend_note)."""
    import cv2

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for s in samples:
        png = frames_dir / f"frame_{s['frame']:04d}.png"
        cv2.imwrite(str(png), s["image"])
        results.append({"frame": s["frame"], "kind": s["kind"], "expected_part": s["expected_part"],
                        "png": str(png), "judgement": "for_human_review", "vlm": None})

    if backend_name in ("none", ""):
        return results, "VLM disabled (--vlm-backend none) — frames saved for human review."

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        from src.audio import vlm
        be = vlm.get_backend(backend_name, model_path)
    except Exception as e:  # model missing / OOM / import error — degrade gracefully
        return results, (f"VLM backend {backend_name!r} failed to load ({type(e).__name__}: {e}) — "
                         f"frames saved for human review only.")

    # The backend prompt is FIXED (contact/part/impact JSON) — it cannot be asked a
    # free-form "is the render plausible" question, so we only verify agreement at
    # contact frames: does the VLM see a contact whose part matches the solver's?
    note = (f"VLM backend '{be.name}' loaded. Prompt is fixed forced-choice "
            "(contact/part/impact); contact frames judged by solver-agreement, boundary "
            "frames left for human review.")
    for r, s in zip(results, samples):
        rgb = cv2.cvtColor(s["image"], cv2.COLOR_BGR2RGB)
        # Qwen vision tokens scale with resolution — full-res render frames OOM an
        # 8 GB GPU, so cap the long side (the saved PNGs stay full resolution)
        h, w = rgb.shape[:2]
        if max(h, w) > 896:
            sc = 896.0 / max(h, w)
            rgb = cv2.resize(rgb, (int(w * sc), int(h * sc)))
        ctx = {"fused_event_type": "contact" if s["kind"] == "contact" else "none",
               "target_entity": s["expected_part"] or "none", "contact_target": "part",
               "audio_support": 0.0, "manip_weight": 0.5, "visual_cue": "none", "fused_conf": 0.5}
        try:
            a = be.analyze(rgb, ctx)
        except Exception as e:
            r["judgement"] = "unclear"
            r["vlm"] = {"error": f"{type(e).__name__}: {e}"}
            continue
        r["vlm"] = {"source": a.source, "relevant": a.relevant, "impact": a.impact,
                    "entities": a.entities, "description": a.description,
                    "contact": a.extra.get("contact")}
        if s["kind"] != "contact":
            continue  # stays for_human_review — fixed prompt can't assess "floating"
        saw_contact = bool(a.extra.get("contact", a.relevant))
        blob = " ".join(a.entities + [a.description]).lower()
        part = s["expected_part"]
        part_match = bool(part) and (part in blob or (
            _side(part) in blob and part.rsplit("_", 1)[-1] in blob))
        if saw_contact and part_match:
            r["judgement"] = "pass"
        elif not saw_contact and not a.relevant:
            r["judgement"] = "fail"
        else:
            r["judgement"] = "unclear"
    return results, note


# Part 3 — corrections + report

def build_corrections(findings: list[dict], args: argparse.Namespace) -> list[dict]:
    """Map findings to the bounded recipe vocabulary (see module docstring)."""
    corrections: list[dict] = []
    for fd in findings:
        label, frames = fd["label"], fd["frames"]
        if label in ("depth_outlier", "boundary_drift"):
            corrections.append({"issue": label, "frames": frames,
                                "recipe": {"type": "rerun_solver", "flags": {"--w-temp": 8.0}}})
        elif label == "contact_gap_violation":
            review = []
            for e in fd.get("details", []):
                if e["audio_anchor"] and e["audio_support"] < args.demote_support:
                    corrections.append({"issue": label, "frames": [e["frame"]],
                                        "recipe": {"type": "demote_anchor", "frame": e["frame"]}})
                elif e["gap_m"] < 0:  # object closer than the part → penetration-type
                    corrections.append({"issue": label, "frames": [e["frame"]],
                                        "recipe": {"type": "rerun_body_refine", "flags": {}}})
                else:
                    review.append(e["frame"])
            if review:
                corrections.append({"issue": label, "frames": review,
                                    "recipe": {"type": "visual_review", "frames": review}})
        else:  # part_flip / audio_visual_mismatch / low_conf_unsupported
            corrections.append({"issue": label, "frames": frames,
                                "recipe": {"type": "visual_review", "frames": frames}})
    return corrections


def decide_verdict(findings: list[dict], vlm_results: list[dict], n_frames: int) -> tuple[str, str]:
    high_frames = sorted({f for fd in findings if fd["severity"] == "high" for f in fd["frames"]})
    n_high = sum(1 for fd in findings if fd["severity"] == "high")
    vlm_fails = [r["frame"] for r in vlm_results if r["judgement"] == "fail"]
    if high_frames and (len(high_frames) > 0.02 * n_frames or len(high_frames) >= 5):
        return "FAIL", (f"{n_high} high-severity finding(s) covering {len(high_frames)} frames "
                        f"({100.0 * len(high_frames) / n_frames:.1f}% of {n_frames})")
    if findings or vlm_fails:
        why = []
        if findings:
            why.append(f"{len(findings)} finding(s): " + ", ".join(fd["label"] for fd in findings))
        if vlm_fails:
            why.append(f"VLM contact check failed at frames {vlm_fails}")
        return "WARN", "; ".join(why)
    return "PASS", "no findings; VLM checks did not contradict the reconstruction"


def write_report(out_dir: Path, sample_dir: Path, traj_csv: Path, findings: list[dict],
                 vlm_results: list[dict], vlm_note: str, verdict: str, reason: str,
                 n_frames: int) -> Path:
    lines = [
        "# LLM/VLM correction audit",
        "",
        f"- sample: `{sample_dir}`",
        f"- trajectory: `{traj_csv}`",
        f"- frames audited: {n_frames}",
        "",
        "## Deterministic findings",
        "",
    ]
    if findings:
        lines += ["| label | severity | frames | evidence |", "|---|---|---|---|"]
        lines += [f"| {fd['label']} | {fd['severity']} | {frames_to_ranges(fd['frames'])} | {fd['evidence']} |"
                  for fd in findings]
    else:
        lines.append("No findings — all table audits clean.")
    lines += ["", "## VLM render check", "", vlm_note, ""]
    if vlm_results:
        lines += ["| frame | kind | expected part | judgement | VLM says |", "|---|---|---|---|---|"]
        for r in vlm_results:
            v = r["vlm"]
            says = "-" if not v else (v.get("error") or
                                      f"contact={v.get('contact')} impact={v.get('impact')} "
                                      f"entities={v.get('entities')} \"{v.get('description', '')}\"")
            lines.append(f"| {r['frame']} | {r['kind']} | {r['expected_part'] or '-'} | "
                         f"{r['judgement']} | {says} |")
        lines.append("")
        lines.append(f"Sampled frames saved under `{out_dir / 'frames'}`.")
    else:
        lines.append("No render frames sampled (overlay video missing?).")
    lines += ["", "## Verdict", "", f"**{verdict}** — {reason}", ""]
    path = out_dir / "llm_correction_report.md"
    path.write_text("\n".join(lines))
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit the final HOI reconstruction and emit bounded re-run recipes.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--trajectory-csv", type=Path, default=None,
                        help="final object trajectory (default: pose6d_sharedcam_contactphase_depthv3)")
    parser.add_argument("--contact-records-csv", type=Path, default=None,
                        help="audio-semantic records (default: results/audio_semantics/contact_records.csv)")
    parser.add_argument("--object-depth-csv", type=Path, default=None,
                        help="DA3 depth table (default: results/depth/object_depth.csv)")
    parser.add_argument("--overlay-video", type=Path, default=None,
                        help="rendered overlay (default: results/renders/full_scene_3d/overlay.mp4)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output dir (default: results/llm_correction)")
    parser.add_argument("--vlm-backend", default="qwen",
                        help="src.audio.vlm backend name ('none' to skip the render check)")
    parser.add_argument("--vlm-model", default=None, help="override VLM model path")
    parser.add_argument("--depth-jump-m", type=float, default=0.35)
    parser.add_argument("--boundary-std-m", type=float, default=0.05)
    parser.add_argument("--contact-gap-m", type=float, default=0.05)
    parser.add_argument("--mismatch-window", type=int, default=3)
    parser.add_argument("--low-conf", type=float, default=0.05)
    parser.add_argument("--low-conf-run", type=int, default=10)
    parser.add_argument("--demote-support", type=float, default=0.3,
                        help="audio anchors below this support get demoted on contact-gap violations")
    args = parser.parse_args(argv)

    results_dir = args.sample_dir / "results"
    traj_csv = args.trajectory_csv or (
        results_dir / "pose6d_sharedcam_contactphase_depthv3" / "ball_pose6d_sharedcam_contactphase_trajectory.csv")
    records_csv = args.contact_records_csv or (results_dir / "audio_semantics" / "contact_records.csv")
    depth_csv = args.object_depth_csv or (results_dir / "depth" / "object_depth.csv")
    overlay_mp4 = args.overlay_video or (results_dir / "renders" / "full_scene_3d" / "overlay.mp4")
    out_dir = args.out_dir or (results_dir / "llm_correction")
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = read_csv_rows(traj_csv)
    records = read_csv_rows(records_csv) if records_csv.exists() else None
    depth = read_csv_rows(depth_csv) if depth_csv.exists() else None
    if records is None:
        print(f"note: no contact records at {records_csv} — audio_visual_mismatch skipped")
    if depth is None:
        print(f"note: no object depth at {depth_csv} — low_conf_unsupported skipped")

    findings = run_table_audits(traj, records, depth, args)

    vlm_results: list[dict] = []
    vlm_note = f"Overlay render not found at {overlay_mp4} — VLM check skipped."
    if overlay_mp4.exists():
        samples = sample_render_frames(overlay_mp4, traj)
        vlm_results, vlm_note = run_vlm_check(samples, out_dir, args.vlm_backend, args.vlm_model)

    verdict, reason = decide_verdict(findings, vlm_results, len(traj))
    corrections = build_corrections(findings, args)

    report_path = write_report(out_dir, args.sample_dir, traj_csv, findings, vlm_results,
                               vlm_note, verdict, reason, len(traj))
    corr_path = out_dir / "corrections.json"
    corr_path.write_text(json.dumps({
        "sample_dir": str(args.sample_dir), "trajectory_csv": str(traj_csv),
        "verdict": verdict, "reason": reason, "corrections": corrections,
    }, indent=2))

    for fd in findings:
        print(f"[{fd['severity']:6s}] {fd['label']}: frames {frames_to_ranges(fd['frames'])} — {fd['evidence']}")
    for r in vlm_results:
        print(f"[vlm   ] f{r['frame']} ({r['kind']}): {r['judgement']}")
    print(f"verdict: {verdict} — {reason}")
    print(f"report: {report_path}")
    print(f"corrections: {corr_path}")


if __name__ == "__main__":
    main()
