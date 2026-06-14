#!/usr/bin/env python3
"""Mug handle-phase + depth-anchor pipeline.

Stages (all CSV-only except the final render):
  1. phase      VLM temporal filter + joint phase optimizer    → results/pipe/handle_phase.csv
  2. correction Hidden-segment far-side interpolation          → results/pipe/corrected_phase.csv
  3. anchor     Z + XY camera-space depth anchor              → results/pipe/anchored_pose.csv
  4. render     Final scene render (6 videos)                 → results/renders/pipe/

Each stage is skipped if its output already exists (use --force to re-run).

Usage (from repo root):
  python src/mug/pipeline.py --sample-dir samples_known_object/02_mug
  python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --force
  python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --from correction
  python src/mug/pipeline.py --sample-dir samples_known_object/02_mug --only render --wireframe
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Stage imports (lazy — only imported when that stage is actually run)
# ---------------------------------------------------------------------------

def _stage_phase():
    from src.mug import phase as m
    return m

def _stage_correction():
    from src.mug import correction as m
    return m

def _stage_anchor():
    from src.mug import anchor as m
    return m


# ---------------------------------------------------------------------------
# Render stage — calls the existing render script as a subprocess
# (it is already rendering-only; no need to rewrite)
# ---------------------------------------------------------------------------

_RENDER_SCRIPT = REPO / "scripts" / "shared" / "radius_free_proxy" / "stage5_render" / "render_mug_articraft_camera3d_scene.py"


def _run_render(sample: Path, phase_csv: Path, pose_csv: Path, wireframe: bool) -> Path:
    out_dir = sample / "results" / "renders" / "pipe"
    cmd = [
        PYTHON, str(_RENDER_SCRIPT),
        "--sample-dir", str(sample),
        "--pose-csv", str(pose_csv),
        "--phase-csv", str(phase_csv),
        "--out-dir", str(out_dir),
    ]
    if wireframe:
        cmd.append("--wireframe")
    print(f"  $ {' '.join(str(a) for a in cmd)}")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        raise RuntimeError(f"Render stage failed (exit {result.returncode})")
    return out_dir


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

STAGE_NAMES = ["phase", "correction", "anchor", "render"]


def _pipe_dir(sample: Path) -> Path:
    return sample / "results" / "pipe"


def _default_inputs(sample: Path) -> dict[str, Path]:
    """Resolve standard default input paths (not stage outputs)."""
    proxy_dir = sample / "proxy"
    pose_csv = (proxy_dir / "mug_body_only_cylinder_pose_table_static_sequence.csv")
    if not pose_csv.exists():
        pose_csv = proxy_dir / "mug_body_only_cylinder_pose_segmented_sequence.csv"
    return {
        "pose_csv": pose_csv,
        "prior_csv": sample / "results" / "renders" / "M12_articraft_rigid_mesh_vlm" / "handle_phase_all.csv",
        "contact_csv": sample / "results" / "mug_articraft_contact_points" / "mug_articraft_contact_points.csv",
        "vlm_csv": sample / "annotations" / "vlm_handle_visibility_full" / "qwen_handle_visibility.csv",
    }


def _output_path(sample: Path, stage: str) -> Path:
    d = _pipe_dir(sample)
    if stage == "phase":
        return d / "handle_phase.csv"
    if stage == "correction":
        return d / "corrected_phase.csv"
    if stage == "anchor":
        return d / "anchored_pose.csv"
    if stage == "render":
        return sample / "results" / "renders" / "pipe"
    raise ValueError(stage)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, default=REPO / "samples_known_object" / "02_mug")
    ap.add_argument("--force", action="store_true", help="Re-run all stages even if outputs exist.")
    ap.add_argument("--from", dest="from_stage", choices=STAGE_NAMES, default=None,
                    help="Start from this stage (re-run it and all subsequent).")
    ap.add_argument("--only", choices=STAGE_NAMES, default=None,
                    help="Run only this single stage.")
    ap.add_argument("--wireframe", action="store_true",
                    help="Use wireframe camera3d in the render stage (fast preview).")
    ap.add_argument("--body-model-root", type=Path, default=None)
    ap.add_argument("--max-nfev-phase", type=int, default=120)
    ap.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for M17 correction.")

    args = ap.parse_args()
    sample = args.sample_dir.resolve()

    if args.only:
        stages_to_run = {args.only}
        force_set = stages_to_run
    elif args.from_stage:
        idx = STAGE_NAMES.index(args.from_stage)
        force_set = set(STAGE_NAMES[idx:])
        stages_to_run = set(STAGE_NAMES)
    else:
        stages_to_run = set(STAGE_NAMES)
        force_set = set(STAGE_NAMES) if args.force else set()

    inputs = _default_inputs(sample)
    pipe = _pipe_dir(sample)
    pipe.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Mug pipeline  —  {sample.name}")
    print(f"{'='*62}\n")

    # ── Stage 1: phase ───────────────────────────────────────────────────────
    phase_out = _output_path(sample, "phase")
    if "phase" in stages_to_run:
        if not phase_out.exists() or "phase" in force_set:
            print("[1/4] phase  — VLM filter + joint phase optimization")
            _stage_phase().run(
                sample,
                inputs["pose_csv"],
                inputs["prior_csv"],
                inputs["contact_csv"],
                vlm_csv=inputs["vlm_csv"],
                out_csv=phase_out,
                max_nfev=args.max_nfev_phase,
            )
        else:
            print(f"[1/4] phase  — skip (exists: {phase_out.name})")

    if not phase_out.exists():
        sys.exit(f"ERROR: phase output missing: {phase_out}")

    # ── Stage 2: correction ──────────────────────────────────────────────────
    corrected_out = _output_path(sample, "correction")
    if "correction" in stages_to_run:
        if not corrected_out.exists() or "correction" in force_set:
            print("[2/4] correction  — hidden-segment far-side interpolation")
            _stage_correction().run(
                sample,
                phase_csv=phase_out,
                pose_csv=inputs["pose_csv"],
                vlm_csv=inputs["vlm_csv"],
                sigma=args.sigma,
                out_csv=corrected_out,
            )
        else:
            print(f"[2/4] correction  — skip (exists: {corrected_out.name})")

    if not corrected_out.exists():
        sys.exit(f"ERROR: corrected phase missing: {corrected_out}")

    # ── Stage 3: anchor ───────────────────────────────────────────────────────
    anchored_out = _output_path(sample, "anchor")
    if "anchor" in stages_to_run:
        if not anchored_out.exists() or "anchor" in force_set:
            print("[3/4] anchor  — Z + XY depth anchor refinement")
            _stage_anchor().run(
                sample,
                pose_csv=inputs["pose_csv"],
                phase_csv=corrected_out,
                body_model_root=args.body_model_root,
                out_csv=anchored_out,
            )
        else:
            print(f"[3/4] anchor  — skip (exists: {anchored_out.name})")

    if not anchored_out.exists():
        sys.exit(f"ERROR: anchored pose missing: {anchored_out}")

    # ── Stage 4: render ───────────────────────────────────────────────────────
    render_out = _output_path(sample, "render")
    if "render" in stages_to_run:
        if not render_out.exists() or "render" in force_set:
            print("[4/4] render  — final scene render (6 videos)")
            _run_render(sample, corrected_out, anchored_out, args.wireframe)
        else:
            print(f"[4/4] render  — skip (exists: {render_out.name})")

    print(f"\n{'='*62}")
    print("  Done.")
    print(f"  phase      → {phase_out.relative_to(sample)}")
    print(f"  correction → {corrected_out.relative_to(sample)}")
    print(f"  anchor     → {anchored_out.relative_to(sample)}")
    print(f"  render     → {render_out.relative_to(sample)}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
