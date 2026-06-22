#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import smplx
import torch

from contact_part_utils import (
    build_contact_identity,
    choose_active_contact_relation,
    event_frames_by_type,
    human_event_frames_generic,
    infer_default_part,
    normalize_contact_label,
    resolve_human_state_key,
)

# audio event taxonomy (scripts/shared/events/audio_semantics.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "events"))
import audio_semantics  # noqa: E402


def read_ball_pose(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame": int(row["frame"]), "time": float(row["time"]), "tx": float(row["tx"]), "ty": float(row["ty"]), "tz": float(row["tz"]),
                "qw": float(row["qw"]), "qx": float(row["qx"]), "qy": float(row["qy"]), "qz": float(row["qz"]), "radius_m": float(row["radius_m"]),
                "coord_frame": row["coord_frame"], "u_obs": float(row["u_obs"]), "v_obs": float(row["v_obs"]), "radius_obs_px": float(row["radius_obs_px"]),
                "u_proj": float(row["u_proj"]), "v_proj": float(row["v_proj"]), "radius_proj_px": float(row["radius_proj_px"]), "bottom_proj_v": float(row["bottom_proj_v"]),
                "floor_v": float(row["floor_v"]), "residual_px": float(row["residual_px"]), "contact_frame": int(row.get("contact_frame", 0) or 0),
                "audio_contact_frame": int(row.get("audio_contact_frame", 0) or 0),
            })
    if not rows:
        raise RuntimeError(f"No ball rows found in {path}")
    return rows


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def read_object_observations(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cx = row.get("center_x", "")
            cy = row.get("center_y", "")
            radius = row.get("enclosing_radius_px", "")
            if not cx or not cy or not radius:
                continue
            rows[int(row["frame"])] = {"u_obs": float(cx), "v_obs": float(cy), "radius_obs_px": float(radius)}
    return rows


def read_support_geometry(path: Path) -> dict[str, float | str]:
    with path.open() as f:
        payload = json.load(f)
    return {
        "support_type": str(payload.get("support_type", "floor")),
        "floor_v": float(payload["floor_v"]),
        "source": str(payload.get("source", "unknown")),
        "confidence": float(payload.get("confidence", 0.0)),
    }


def read_human_result(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    params = data["smpl_params_incam"]
    return {
        "body_pose": np.asarray(params["body_pose"], dtype=np.float32),
        "betas": np.asarray(params["betas"], dtype=np.float32),
        "global_orient": np.asarray(params["global_orient"], dtype=np.float32),
        "transl": np.asarray(params["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(data["K_fullimg"], dtype=np.float32),
    }


def build_body_joints(body_models_root: Path, human_params: dict[str, np.ndarray]) -> np.ndarray:
    model = smplx.create(str(body_models_root), model_type="smplx", gender="neutral", ext="npz", use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=human_params["transl"].shape[0])
    with torch.inference_mode():
        output = model(body_pose=torch.from_numpy(human_params["body_pose"]), betas=torch.from_numpy(human_params["betas"]), global_orient=torch.from_numpy(human_params["global_orient"]), transl=torch.from_numpy(human_params["transl"]), return_verts=False)
    return output.joints.detach().cpu().numpy().astype(np.float64)


def reconstruct_xyz_from_uvz(u_obs: np.ndarray, v_obs: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    x = (u_obs - cx) * z / fx
    y = (v_obs - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def project_ball(ball_xyz: np.ndarray, K: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(ball_xyz[:, 2], 1e-6, None)
    r = K[:, 0, 0] * (radius_m / z)
    bottom_v = K[:, 1, 1] * (ball_xyz[:, 1] / z) + K[:, 1, 2] + r
    return r, bottom_v


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_audio_events(path: Path) -> list[dict[str, float | int]]:
    """Read audio onset events (frame, time, score).

    audio_events.csv comes in two flavours: some samples (football, the radius-free
    branch) carry a normalized ``audio_score`` column; the basketball detector only
    writes ``peak``/``prominence``. When ``audio_score`` is missing we derive it from
    ``prominence`` normalized to its max so both paths yield a confidence in [0, 1].
    """
    rows: list[dict[str, float | int]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        records = list(csv.DictReader(f))
    if not records:
        return rows
    has_score = "audio_score" in records[0]
    proms = [float(r.get("prominence", 0.0) or 0.0) for r in records]
    prom_max = max(proms) if proms else 0.0
    for r, prom in zip(records, proms):
        frame = r.get("audio_frame", "")
        if not frame:
            continue
        if has_score and (r.get("audio_score", "") not in ("", None)):
            score = float(r["audio_score"])
        else:
            score = (prom / prom_max) if prom_max > 0.0 else 0.0
        rows.append({
            "audio_frame": int(float(frame)),
            "audio_time": float(r.get("audio_time", 0.0) or 0.0),
            "audio_score": float(np.clip(score, 0.0, 1.0)),
        })
    return rows


def build_audio_support(frames: np.ndarray, audio_rows: list[dict[str, float | int]], radius: int = 2) -> np.ndarray:
    """Per-frame audio confidence in [0, 1].

    Each onset is spread over +-``radius`` frames with a triangular falloff and the
    events are combined by max, so ``audio_support[t]`` is the strength of the nearest
    impact. Frames with no nearby onset stay 0 (audio terms then become no-ops).
    """
    support = np.zeros(len(frames), dtype=np.float64)
    frame_to_idx = {int(fr): i for i, fr in enumerate(np.asarray(frames).tolist())}
    for row in audio_rows:
        center = int(row["audio_frame"])
        score = float(row["audio_score"])
        for fr in range(center - radius, center + radius + 1):
            idx = frame_to_idx.get(fr)
            if idx is None:
                continue
            weight = max(0.0, 1.0 - 0.25 * abs(fr - center))
            support[idx] = max(support[idx], score * weight)
    return support


def solve_anchor_interpolation(
    z_ref: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_values: np.ndarray,
    w_ref: float,
    w_temp: float,
    audio_support: np.ndarray | None = None,
    audio_anchor_target: np.ndarray | None = None,
    w_audio: float = 0.0,
    audio_accel_relax: float | np.ndarray = 0.0,
    w_audio_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Refine ball depth between contact anchors.

    Data-driven only: contact frames are pinned to the contacting part's depth
    (anchors), free frames stay near their prior depth, and the single prior is an
    acceleration-smoothness regularizer so nothing jumps around frame to frame.
    No gravity / ballistic assumption - that's object-specific and we want this to
    hold for anything (swinging, sliding, placing, ...), not just thrown balls.

    Audio (when ``audio_support`` is provided) adds two effects gated by the per-frame
    onset confidence in [0, 1]:
      * a soft contact pull (``w_audio``) toward ``audio_anchor_target`` on free
        frames near an impact - the audio moment says the object touched, so depth is
        nudged to the contacting part even where visual detection was weak;
      * a local relaxation of the acceleration regularizer (``audio_accel_relax``) so a
        real bounce/placement velocity kink is not smoothed away.
    With all-zero ``audio_support`` both terms vanish and the solve is unchanged.
    """
    free_idx = np.flatnonzero(~anchor_mask)
    anchor_idx = np.flatnonzero(anchor_mask)
    if len(anchor_idx) == 0:
        raise RuntimeError("No anchors available for anchor interpolation")

    n = len(z_ref)
    if audio_support is None:
        audio_support = np.zeros(n, dtype=np.float64)
    # per-frame relaxation strength gamma_t (scalar broadcasts); set by the audio
    # taxonomy (e.g. full kink for an impact, almost none for a sustained hold)
    relax = np.broadcast_to(np.asarray(audio_accel_relax, dtype=np.float64), (n,))
    pull_scale = np.ones(n) if w_audio_scale is None else np.asarray(w_audio_scale, dtype=np.float64)

    def unpack_z(free_values: np.ndarray) -> np.ndarray:
        z = np.asarray(z_ref, dtype=np.float64).copy()
        z[anchor_idx] = anchor_values[anchor_idx]
        z[free_idx] = free_values
        return np.maximum(z, 0.20)

    def residuals(free_values: np.ndarray) -> np.ndarray:
        z = unpack_z(free_values)
        residual_list = [w_ref * (z[free_idx] - z_ref[free_idx])]

        # audio-weighted soft contact anchor: at impact moments pull free-frame depth
        # toward the contacting part's depth, with confidence ~ audio onset strength
        if w_audio > 0.0 and audio_anchor_target is not None:
            a_free = audio_support[free_idx]
            tgt = np.asarray(audio_anchor_target, dtype=np.float64)[free_idx]
            valid = (a_free > 0.0) & np.isfinite(tgt)
            if np.any(valid):
                coef = w_audio * pull_scale[free_idx][valid] * a_free[valid]
                residual_list.append(coef * (z[free_idx][valid] - tgt[valid]))

        # penalize depth acceleration (second difference) on free frames - this is
        # the "no sudden unexpected moves" regularizer, not a physics prior. It is
        # locally relaxed at audio impacts (by the per-frame, per-event-type gamma) so
        # a real bounce/placement kink survives.
        if n >= 3:
            second_z = z[2:] - 2.0 * z[1:-1] + z[:-2]
            smooth_mask = ~anchor_mask[1:-1]
            if np.any(smooth_mask):
                accel_w = np.clip(w_temp * (1.0 - relax[1:-1] * audio_support[1:-1]), 0.0, None)
                residual_list.append((accel_w * second_z)[smooth_mask])

        return np.concatenate([np.ravel(r) for r in residual_list]).astype(np.float64)

    x0 = z_ref[free_idx].copy()
    result = least_squares(
        residuals,
        x0=x0,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=400,
    )
    return unpack_z(result.x)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generic anchor-only z refinement for human-ball contact.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    parser.add_argument("--out-subdir", type=str, default="pose6d_sharedcam_contactphase_anchorinterp_generic")
    parser.add_argument("--contact-state-csv", type=Path, default=None)
    parser.add_argument("--contact-event-csv", type=Path, default=None)
    parser.add_argument("--object-observation-csv", type=Path, default=None)
    parser.add_argument("--ball-trajectory-csv", type=Path, default=None,
                        help="Input ball trajectory (default sphere baseline; pass the "
                             "pose6d_sharedcam_depthv3 trajectory to refine DA3 depth).")
    parser.add_argument("--support-geometry-json", type=Path, default=None)
    parser.add_argument("--default-part", type=str, choices=["hand", "foot"], default=None)
    parser.add_argument("--outside-window-mode", type=str, choices=["global_ref", "boundary_constant"], default="global_ref")
    parser.add_argument("--delta-stat", type=str, choices=["median", "mean"], default="median")
    parser.add_argument("--w-ref", type=float, default=0.7, help="keep free frames near their prior depth")
    parser.add_argument("--w-temp", type=float, default=5.0, help="smoothness regularizer (depth acceleration)")
    parser.add_argument("--audio-events-csv", type=Path, default=None,
                        help="audio onset events (default results/events/audio_events.csv); drives "
                             "audio contact timing, soft contact pull, and acceleration relaxation")
    parser.add_argument("--w-audio", type=float, default=3.0,
                        help="weight of the soft audio-gated contact pull on free frames (0 disables)")
    parser.add_argument("--audio-accel-relax", type=float, default=0.8,
                        help="0..1: fraction by which to relax depth-acceleration smoothness at audio impacts")
    parser.add_argument("--audio-new-anchors", dest="audio_new_anchors", action="store_true", default=True,
                        help="promote strong audio onsets to hard contact anchors (default on)")
    parser.add_argument("--no-audio-new-anchors", dest="audio_new_anchors", action="store_false")
    parser.add_argument("--audio-anchor-thresh", type=float, default=0.5,
                        help="audio_score threshold for promoting an onset frame to a contact anchor")
    parser.add_argument("--audio-semantics", dest="audio_semantics", action="store_true", default=True,
                        help="classify each onset (impact/bounce/placement/scrape/sustained) and apply "
                             "per-event physics: per-frame gamma overrides --audio-accel-relax, and only "
                             "body-part contacts are promoted to anchors (floor bounces are not) (default on)")
    parser.add_argument("--no-audio-semantics", dest="audio_semantics", action="store_false")
    args = parser.parse_args(argv)

    sample_dir = args.sample_dir
    results_dir = sample_dir / "results"
    out_dir = results_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    state_csv = args.contact_state_csv or (results_dir / "contact_candidates" / "contact_state_frames.csv")
    event_csv = args.contact_event_csv or (results_dir / "contact_candidates" / "contact_candidates_labeled.csv")
    object_obs_csv = args.object_observation_csv or (results_dir / "object_observations" / "object_observations.csv")
    support_json = args.support_geometry_json or (results_dir / "pose6d_sharedcam" / "support_geometry.json")

    ball_csv = args.ball_trajectory_csv or (results_dir / "pose6d_sharedcam" / "ball_pose6d_sharedcam_trajectory.csv")
    ball_rows = read_ball_pose(ball_csv)
    if object_obs_csv.exists():
        object_obs = read_object_observations(object_obs_csv)
        for row in ball_rows:
            obs = object_obs.get(int(row["frame"]))
            if obs is None:
                continue
            row["u_obs"] = obs["u_obs"]
            row["v_obs"] = obs["v_obs"]
            row["radius_obs_px"] = obs["radius_obs_px"]

    support = None
    if support_json.exists():
        support = read_support_geometry(support_json)
        for row in ball_rows:
            row["floor_v"] = float(support["floor_v"])

    state_rows = read_rows(state_csv)
    event_rows = read_rows(event_csv)
    human = read_human_result(results_dir / "gvhmr" / "result.pkl")
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human["K_fullimg"], dtype=np.float64)
    if len(joints) < len(ball_rows):
        raise RuntimeError(f"GVHMR has fewer frames than sharedcam ball rows: {len(joints)} < {len(ball_rows)}")
    if len(joints) != len(ball_rows):
        joints = joints[:len(ball_rows)]
        K = K[:len(ball_rows)]

    state_by_frame = {int(r["frame"]): r for r in state_rows}
    ball_rows = [r for r in ball_rows if int(r["frame"]) in state_by_frame]
    joints = joints[:len(ball_rows)]
    K = K[:len(ball_rows)]
    state_rows = [state_by_frame[int(r["frame"])] for r in ball_rows]
    ball_frames = [int(r["frame"]) for r in ball_rows]

    default_part = args.default_part or infer_default_part([*state_rows, *event_rows], fallback="hand")
    human_event_frames = human_event_frames_generic(event_rows)
    floor_event_frames = event_frames_by_type(event_rows, {"floor_contact_event"})

    u_obs = np.asarray([r["u_obs"] for r in ball_rows], dtype=np.float64)
    v_obs = np.asarray([r["v_obs"] for r in ball_rows], dtype=np.float64)
    z_init = np.asarray([r["tz"] for r in ball_rows], dtype=np.float64)

    human_event_mask = np.asarray([f in human_event_frames for f in ball_frames], dtype=bool)
    floor_event_mask = np.asarray([f in floor_event_frames for f in ball_frames], dtype=bool)
    floor_state_mask = np.asarray([int(r["floor_contact_state"]) == 1 for r in state_rows], dtype=bool)
    state_key = resolve_human_state_key(state_rows[0])
    if state_key is None:
        raise RuntimeError("No generic human contact state field found")
    human_state_mask = np.asarray([int(r[state_key]) == 1 for r in state_rows], dtype=bool)

    contact_labels =[normalize_contact_label(r, default_part=default_part, fallback_side="right") for r in state_rows]
    part_y, part_z, part_name = choose_active_contact_relation(joints, contact_labels, fallback_label=f"right_{default_part}")
    if not np.any(human_event_mask):
        raise RuntimeError("No human contact events found; cannot compute global Delta-Z")

    deltas = part_z[human_event_mask] - z_init[human_event_mask]
    global_z_shift = float(np.median(deltas) if args.delta_stat == "median" else np.mean(deltas))
    z_ref = np.maximum(z_init + global_z_shift, 0.20)

    # Audio contact timing: per-frame onset confidence over the ball frames.
    audio_csv = args.audio_events_csv or (results_dir / "events" / "audio_events.csv")
    audio_rows = read_audio_events(audio_csv)
    ball_frames_arr = np.asarray(ball_frames, dtype=np.int64)
    audio_support = build_audio_support(ball_frames_arr, audio_rows)

    # Defaults: scalar relaxation, uniform pull, threshold-based promotion (Phase 1).
    relax_field: float | np.ndarray = args.audio_accel_relax
    w_audio_scale = None
    event_type_field = np.array(["" for _ in ball_frames], dtype=object)
    classified: list = []
    audio_promote_mask = None

    # Audio semantics (Phase 2): classify each onset and apply per-event physics.
    if args.audio_semantics and audio_rows:
        classified = audio_semantics.classify_audio_events(sample_dir / "audio.wav", audio_rows)
        if classified:
            sem = audio_semantics.build_semantic_fields(ball_frames_arr, classified)
            audio_support = sem["support"]
            relax_field = sem["gamma"]              # per-event gamma overrides the scalar flag
            w_audio_scale = sem["w_audio_scale"]
            event_type_field = sem["event_type"]
            audio_promote_mask = sem["promote"]     # already gated to body-part contacts
            audio_semantics.write_semantics_csv(results_dir / "events" / "audio_semantics.csv", classified)

    # Promote strong audio onsets (exact peak frame only) to hard contact anchors,
    # pinned to the contacting part's depth like the visual anchors. With semantics on,
    # only body-part contacts are promoted (floor bounces are left unpinned).
    audio_anchor_mask = np.zeros(len(ball_frames), dtype=bool)
    if args.audio_new_anchors and audio_rows:
        if audio_promote_mask is not None:
            promote = audio_promote_mask
        else:
            strong = {int(r["audio_frame"]) for r in audio_rows
                      if float(r["audio_score"]) >= args.audio_anchor_thresh}
            promote = np.asarray([f in strong for f in ball_frames], dtype=bool)
        audio_anchor_mask = promote & np.isfinite(part_z) & ~human_event_mask
    anchor_mask = human_event_mask | audio_anchor_mask

    z_final = solve_anchor_interpolation(
        z_ref=z_ref,
        anchor_mask=anchor_mask,
        anchor_values=part_z,
        w_ref=args.w_ref,
        w_temp=args.w_temp,
        audio_support=audio_support,
        audio_anchor_target=part_z,
        w_audio=args.w_audio,
        audio_accel_relax=relax_field,
        w_audio_scale=w_audio_scale,
    )
    if args.outside_window_mode == "boundary_constant":
        anchor_idx = np.flatnonzero(anchor_mask)
        interp_start = int(anchor_idx[0])
        interp_end = int(anchor_idx[-1])
        if interp_start > 0:
            z_final[:interp_start] = max(0.20, float(z_final[interp_start]))
        if interp_end + 1 < len(z_final):
            z_final[interp_end + 1:] = max(0.20, float(z_final[interp_end]))

    xyz_final = reconstruct_xyz_from_uvz(u_obs, v_obs, z_final, K)
    radius_m = float(ball_rows[0]["radius_m"])
    r_proj, bottom_proj = project_ball(xyz_final, K, radius_m)

    out_rows = []
    reproj_rows = []
    for idx, row in enumerate(ball_rows):
        contact_part, contact_side, contact_label = build_contact_identity(active_label=str(part_name[idx]), event_on=bool(human_event_mask[idx]), floor_event_on=bool(floor_event_mask[idx]), default_part=default_part)
        out_rows.append({
            "frame": row["frame"], "time": f"{row['time']:.6f}", "tx": f"{xyz_final[idx,0]:.6f}", "ty": f"{xyz_final[idx,1]:.6f}", "tz": f"{xyz_final[idx,2]:.6f}",
            "qw": f"{row['qw']:.6f}", "qx": f"{row['qx']:.6f}", "qy": f"{row['qy']:.6f}", "qz": f"{row['qz']:.6f}", "radius_m": f"{row['radius_m']:.6f}", "coord_frame": row["coord_frame"],
            "u_obs": f"{row['u_obs']:.3f}", "v_obs": f"{row['v_obs']:.3f}", "radius_obs_px": f"{row['radius_obs_px']:.3f}", "u_proj": f"{row['u_obs']:.3f}", "v_proj": f"{row['v_obs']:.3f}",
            "radius_proj_px": f"{r_proj[idx]:.3f}", "bottom_proj_v": f"{bottom_proj[idx]:.3f}", "floor_v": f"{row['floor_v']:.3f}",
            "support_type": support["support_type"] if support is not None else "floor", "support_source": support["source"] if support is not None else "sharedcam_csv",
            "support_confidence": f"{float(support['confidence']):.6f}" if support is not None else "", "residual_px": "0.000000", "contact_frame": int(human_event_mask[idx]),
            "audio_contact_frame": row["audio_contact_frame"],
            "human_contact_event": int(human_event_mask[idx]), "floor_contact_event": int(floor_event_mask[idx]),
            "human_contact_state": int(human_state_mask[idx]), "floor_contact_state": int(floor_state_mask[idx]),
            "contact_part": contact_part, "contact_side": contact_side, "contact_label": contact_label,
            "active_part": str(part_name[idx]), "active_part_y": f"{part_y[idx]:.6f}", "active_part_z": f"{part_z[idx]:.6f}",
            "global_z_ref": f"{z_ref[idx]:.6f}", "contact_depth_gap": f"{(xyz_final[idx,2] - part_z[idx]):.6f}",
            "audio_support": f"{audio_support[idx]:.6f}", "audio_anchor": int(audio_anchor_mask[idx]),
            "audio_event_type": str(event_type_field[idx]) if event_type_field[idx] not in ("none", "") else "",
        })
        reproj_rows.append({
            "frame": row["frame"], "u_obs": f"{row['u_obs']:.3f}", "v_obs": f"{row['v_obs']:.3f}", "u_reproj": f"{row['u_obs']:.3f}", "v_reproj": f"{row['v_obs']:.3f}",
            "error_u": "0.000000", "error_v": "0.000000", "error_px": "0.000000",
        })

    out_csv = out_dir / "ball_pose6d_sharedcam_contactphase_trajectory.csv"
    reproj_csv = out_dir / "ball_pose6d_sharedcam_contactphase_reprojection_comparison.csv"
    summary_txt = out_dir / "ball_pose6d_sharedcam_contactphase_summary.txt"

    write_csv(out_csv, out_rows, [
        "frame","time","tx","ty","tz","qw","qx","qy","qz","radius_m","coord_frame",
        "u_obs","v_obs","radius_obs_px","u_proj","v_proj","radius_proj_px","bottom_proj_v",
        "floor_v","support_type","support_source","support_confidence","residual_px","contact_frame","audio_contact_frame",
        "human_contact_event","floor_contact_event","human_contact_state","floor_contact_state",
        "contact_part","contact_side","contact_label","active_part","active_part_y","active_part_z",
        "global_z_ref","contact_depth_gap","audio_support","audio_anchor","audio_event_type",
    ])
    write_csv(reproj_csv, reproj_rows, ["frame","u_obs","v_obs","u_reproj","v_reproj","error_u","error_v","error_px"])

    with summary_txt.open("w") as f:
        f.write("Generic anchor interpolation z refinement for human-ball contact.\n")
        f.write(f"default_part: {default_part}\n")
        f.write(f"outside_window_mode: {args.outside_window_mode}\n")
        f.write(f"w_ref: {args.w_ref:.6f}\n")
        f.write(f"w_temp: {args.w_temp:.6f}\n")
        f.write(f"w_audio: {args.w_audio:.6f}\n")
        f.write(f"audio_accel_relax: {args.audio_accel_relax:.6f}\n")
        f.write(f"audio_events_csv: {audio_csv}\n")
        f.write(f"global_z_shift_from_human_events_m: {global_z_shift:.6f}\n")
        f.write(f"num_frames: {len(ball_rows)}\n")
        f.write(f"num_human_event_frames: {int(np.count_nonzero(human_event_mask))}\n")
        f.write(f"num_floor_event_frames: {int(np.count_nonzero(floor_event_mask))}\n")
        f.write(f"num_audio_events: {len(audio_rows)}\n")
        f.write(f"num_audio_anchor_frames: {int(np.count_nonzero(audio_anchor_mask))}\n")
        f.write(f"audio_semantics: {bool(args.audio_semantics and classified)}\n")
        if classified:
            from collections import Counter
            counts = dict(Counter(e.event_type for e in classified))
            f.write(f"audio_event_type_counts: {counts}\n")

    print(f"contactphase_csv: {out_csv}")
    print(f"contactphase_reproj_csv: {reproj_csv}")
    print(f"contactphase_summary: {summary_txt}")


if __name__ == "__main__":
    main()
