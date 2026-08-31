#!/usr/bin/env python3
"""AKCA — Audio-Kinematic Contact Attribution (VLM-free contact-part reasoning).

Problem this solves (the entire C2 -> C3 gap in the ablation): the audio contact solver
must decide WHICH body part an audio onset belongs to. The cheap geometric heuristic picks
the part *nearest* the object in the image — which on a football juggle is the dangling
hand, not the kicking foot, so the audio anchor drags the ball into the wrong limb
(penetration up). The expensive fix is a 72B VLM that names the part. AKCA recovers the
correct part from KINEMATICS + AUDIO alone, no VLM:

At each audio onset frame f, score each candidate part p in {left/right hand, left/right
foot} by three physically-motivated cues and pick the argmax (the *striker*):
  1. PROXIMITY  the part must actually be near the object surface at f (gate).
  2. SYNCHRONY  the striker's 3D acceleration spikes AT the onset (an impact is a velocity
                discontinuity synchronized with the sound); a dangling limb does not kink.
  3. RECOIL     the striker approaches the object just before f and recedes just after
                (strike-and-bounce radial-velocity signature).
  score(p,f) = prox(p,f) * (w_sync*z_sync(p,f) + w_recoil*recoil(p,f))

This is loop_plan §5.5 import #3 ("onset<->kinematic-discontinuity attribution", flagged
NOVEL / unpublished). It reuses the existing contact-records interface: we copy the audio
records container and overwrite `target_entity`/`stable_entity` (the part) with the AKCA
striker for part-contact rows, leaving floor/support rows and all promotion/support fields
untouched. So a solver run with `--audio-records-csv <akca_records>` isolates exactly the
attribution method (geometric C2 vs AKCA vs VLM C3).

Usage:
  conda run -n gvhmr python scripts/shared/human_ball/contact/audio_kinematic_attribution.py \
      --sample-dir samples/football_10 \
      --records-csv samples/football_10/results/audio_semantics/contact_records.csv \
      --object-trajectory samples/football_10/results/pose6d_sharedcam_depthv3/ball_pose6d_sharedcam_trajectory.csv \
      --out-csv overnight/c2improve/akca_records/football.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_human_ball_contact_phase_calibration_anchorinterp_generic import (  # noqa: E402
    read_human_result, build_body_joints)
from contact_part_utils import build_contact_part_centers  # noqa: E402

_PARTS = ("left_hand", "right_hand", "left_foot", "right_foot")


def _object_uv(traj_csv: Path, n: int) -> np.ndarray:
    """Observed object 2D center per frame (pixels) — reliable (unlike pre-solve depth)."""
    with traj_csv.open() as f:
        rows = list(csv.DictReader(f))
    by = {int(r["frame"]): r for r in rows}
    uv = np.full((n, 2), np.nan)
    for i in range(n):
        r = by.get(i + 1)
        if r is None:
            continue
        u = r.get("u_obs") or r.get("u_proj")
        v = r.get("v_obs") or r.get("v_proj")
        if u and v:
            uv[i] = [float(u), float(v)]
    for a in range(2):
        col = uv[:, a]
        idx = np.where(~np.isnan(col))[0]
        if len(idx):
            uv[:, a] = np.interp(np.arange(n), idx, col[idx])
    return uv


def _project(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project [N,3] camera-frame points to [N,2] pixels with per-frame K [N,3,3]."""
    z = np.clip(pts[:, 2], 1e-3, None)
    u = K[:, 0, 0] * (pts[:, 0] / z) + K[:, 0, 2]
    v = K[:, 1, 1] * (pts[:, 1] / z) + K[:, 1, 2]
    return np.stack([u, v], axis=1)


def attribute(sample_dir: Path, records_csv: Path, object_traj: Path, body_model_root: Path,
              w_sync: float = 1.0, w_recoil: float = 0.5, prox_sigma_px: float = 90.0,
              prox_gate_px: float = 260.0, win: int = 3,
              mode: str = "full", gate: bool = False,
              gate_floor: float = 0.1) -> tuple[list[dict], list[dict]]:
    human = read_human_result(sample_dir / "results" / "gvhmr" / "result.pkl")
    joints = build_body_joints(body_model_root, human)
    n = joints.shape[0]
    centers = build_contact_part_centers(joints)  # part -> [N,3]
    obj_uv = _object_uv(object_traj, n)
    K = np.asarray(human["K_fullimg"], dtype=np.float64)
    if K.ndim == 2:
        K = K[None].repeat(n, axis=0)
    K = K[:n]

    # per-part acceleration magnitude (3D, reliable) and its robust clip scale
    accel = {p: np.zeros(n) for p in _PARTS}
    part_uv = {}
    for p in _PARTS:
        c = centers[p][:n]
        a = np.zeros(n)
        a[1:-1] = np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1)
        accel[p] = a
        part_uv[p] = _project(c, K)
    accel_scale = {p: (np.median(accel[p][accel[p] > 0]) + 1e-6) if np.any(accel[p] > 0) else 1.0
                   for p in _PARTS}
    # 2D image distance part<->object (reliable; pre-solve depth is not)
    dist = {p: np.linalg.norm(part_uv[p] - obj_uv, axis=1) for p in _PARTS}

    def score_part(p: str, f: int) -> tuple[float, dict]:
        if not (1 <= f < n - 1):
            f = min(max(f, 1), n - 2)
        d = dist[p][f]  # pixels
        prox = float(np.exp(-max(0.0, d - 40.0) / prox_sigma_px)) if d < prox_gate_px else 0.0
        z_sync = float(accel[p][f] / accel_scale[p])
        lo = max(0, f - win); hi = min(n, f + win + 1)
        before = float(np.mean(dist[p][lo:f])) if f > lo else d
        after = float(np.mean(dist[p][f + 1:hi])) if hi > f + 1 else d
        recoil = max(0.0, (before - d)) + max(0.0, (after - d))  # approach+recede, px
        recoil_n = recoil / 100.0
        s = prox * (w_sync * z_sync + w_recoil * recoil_n)
        return s, {"dist_px": round(d, 1), "prox": round(prox, 3), "z_sync": round(z_sync, 2),
                   "recoil_n": round(recoil_n, 2), "score": round(s, 3)}

    with records_csv.open() as f:
        rec_rows = list(csv.DictReader(f))
        fields = list(rec_rows[0].keys()) if rec_rows else []

    diag = []
    for row in rec_rows:
        tgt = str(row.get("target_entity", "") or "")
        # only re-attribute rows that are body-part contacts (leave support/floor/none)
        if tgt not in _PARTS:
            continue
        try:
            f = int(round(float(row.get("frame", 0)))) - 1
        except (TypeError, ValueError):
            continue
        scores = {p: score_part(p, f) for p in _PARTS}
        full_best = max(scores, key=lambda p: scores[p][0])
        best_s = scores[full_best][0]
        # prox-only pick = the geometric nearest-part heuristic (C2 baseline proxy):
        # argmax proximity, ignoring synchrony/recoil.
        prox_only = max(_PARTS, key=lambda p: scores[p][1]["prox"])
        best = prox_only if mode == "prox_only" else full_best
        # RELEVANCE GATE (replaces the VLM's relevance gate): if NO part is both near the
        # object and kinking at the onset (best score below floor), this "part contact" is
        # not supported by kinematics — an off-object / floor / spurious sound. Demote it so
        # it is not hard-anchored to a wrong part.
        gated = gate and best_s < gate_floor
        if gated:
            row["relevant"] = "0"
            row["promote_anchor"] = "0"
            row["target_entity"] = "none"
            row["stable_entity"] = "none"
            row["contact_state"] = "no_contact"
        elif best_s > 0.05 or mode == "prox_only":
            row["target_entity"] = best
            row["stable_entity"] = best
        diag.append({"frame": f + 1, "orig": tgt,
                     "akca": ("DEMOTED" if gated else (best if best_s > 0.05 else tgt)),
                     "prox_only": prox_only, "best_score": round(best_s, 3),
                     **{f"{p}": scores[p][1]["score"] for p in _PARTS}})
    return rec_rows, diag, fields  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--records-csv", type=Path, required=True)
    ap.add_argument("--object-trajectory", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--w-sync", type=float, default=1.0)
    ap.add_argument("--w-recoil", type=float, default=0.5)
    ap.add_argument("--mode", choices=["full", "prox_only"], default="full")
    ap.add_argument("--relevance-gate", action="store_true",
                    help="demote onsets where no part is near+kinking (VLM-free relevance gate)")
    ap.add_argument("--gate-floor", type=float, default=0.1)
    args = ap.parse_args()

    rec_rows, diag, fields = attribute(args.sample_dir, args.records_csv, args.object_trajectory,
                                       args.body_model_root, args.w_sync, args.w_recoil,
                                       mode=args.mode, gate=args.relevance_gate,
                                       gate_floor=args.gate_floor)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rec_rows)
    n_changed = sum(1 for d in diag if d["orig"] != d["akca"])
    n_vs_prox = sum(1 for d in diag if d["prox_only"] != d["akca"])
    print(f"[akca] {args.sample_dir.name}: {len(diag)} part-rows, {n_changed} vs VLM, "
          f"{n_vs_prox} corrections vs geometric(prox-only)")
    for d in diag:
        fix = "  <== AKCA CORRECTS GEOMETRIC" if d["prox_only"] != d["akca"] else ""
        print(f"  f{d['frame']:>3} geometric={d['prox_only']:11s} akca={d['akca']:11s} "
              f"vlm={d['orig']:11s} score={d['best_score']}{fix}")
    (args.out_csv.parent / f"{args.sample_dir.name}_akca_diag.csv").write_text(
        "frame,orig,akca,best_score," + ",".join(_PARTS) + "\n" +
        "\n".join(f"{d['frame']},{d['orig']},{d['akca']},{d['best_score']}," +
                  ",".join(str(d[p]) for p in _PARTS) for d in diag) + "\n")


if __name__ == "__main__":
    main()
