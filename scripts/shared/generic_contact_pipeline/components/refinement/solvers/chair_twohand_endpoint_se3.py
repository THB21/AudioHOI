#!/usr/bin/env python3
"""Small SE(3) refinement for two-hand chair endpoint contact.

This keeps the accepted 2D overlay as a strong projection target, while allowing
small 6D pose changes so left/right palms can attach to opposite top-rail
endpoints.  It is intentionally an experiment and does not replace the default
chair pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[6]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.shared.generic_contact_pipeline.components.pose.solvers import fit_chair_metric_6d_pose as fit  # noqa: E402
from scripts.shared.generic_contact_pipeline.components.render.scenes.render_chair_articraft_3d_scene import (  # noqa: E402
    palm_centers,
    project_cam,
)
from scripts.shared.generic_contact_pipeline.components.render.scenes.render_chair_articraft_visual_model import (  # noqa: E402
    build_body_joints,
    read_human_result,
)
from scripts.shared.generic_contact_pipeline.components.refinement.solvers.contact_chord_initializer import (  # noqa: E402
    align_contact_chord,
)
from scripts.shared.generic_contact_pipeline.core.state import ArticulatedKinematicProvider, SegmentJointRule  # noqa: E402

POSE_KEYS = ["rx_delta", "ry_delta", "rz_delta", "tx", "ty", "tz", "rear_joint_angle", "seat_joint_angle"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def ff(row: dict[str, str] | None, key: str, default: float = math.nan) -> float:
    if row is None:
        return default
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def pose_params(row: dict[str, str]) -> np.ndarray:
    return np.asarray([ff(row, k, 0.0) for k in POSE_KEYS], dtype=float)


def update_pose_quat(row: dict[str, str], params: np.ndarray, source: str) -> dict[str, str]:
    out = dict(row)
    for key, val in zip(POSE_KEYS, params):
        out[key] = f"{float(val):.9f}"
    r_total = Rotation.from_matrix(Rotation.from_rotvec(params[:3]).as_matrix() @ fit.BASE_LOCAL_TO_CAM)
    qx, qy, qz, qw = r_total.as_quat()
    out.update({"qx": f"{qx:.9f}", "qy": f"{qy:.9f}", "qz": f"{qz:.9f}", "qw": f"{qw:.9f}", "source": source})
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def local_rear_origin(segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]]) -> np.ndarray:
    pts = [segs[sid][2] for sid in ("rear_leg_left", "rear_leg_right") if sid in segs]
    return np.mean(np.vstack(pts), axis=0) if pts else np.array([0.0, 0.040, 0.548], dtype=float)


def local_seat_origin(segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]]) -> np.ndarray:
    if "seat_hinge_edge" in segs:
        _part, _role, a, b = segs["seat_hinge_edge"]
        return (a + b) * 0.5
    return np.array([0.0, 0.090, 0.440], dtype=float)


def chair_kinematic_provider(segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]]) -> ArticulatedKinematicProvider:
    rear_origin = local_rear_origin(segs)
    seat_origin = local_seat_origin(segs)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    return ArticulatedKinematicProvider(
        rules=(
            SegmentJointRule(
                rule_id="chair_rear_link",
                joint_id="joint.front_to_rear",
                parts=("rear_leg", "rear_stretcher"),
                segment_ids=("rear_feet_line", "rear_lower_stretcher"),
                origin=rear_origin,
                axis=x_axis,
            ),
            SegmentJointRule(
                rule_id="chair_side_stretcher_rear_endpoint",
                joint_id="joint.front_to_rear",
                parts=("side_stretcher",),
                segment_ids=("side_lower_stretcher",),
                origin=rear_origin,
                axis=x_axis,
                endpoint_selector="max_y",
            ),
            SegmentJointRule(
                rule_id="chair_seat_link",
                joint_id="joint.front_to_seat",
                parts=("seat", "seat_slat"),
                segment_ids=("seat_hinge_edge",),
                origin=seat_origin,
                axis=x_axis,
                angle_sign=-1.0,
            ),
        )
    )


def articulate_segment(
    sid: str,
    part: str,
    pts: np.ndarray,
    params: np.ndarray,
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
) -> np.ndarray:
    joint_values = {
        "joint.front_to_rear": float(params[6]) if len(params) > 6 else 0.0,
        "joint.front_to_seat": float(params[7]) if len(params) > 7 else 0.0,
    }
    return chair_kinematic_provider(segs).articulate_segment(sid, part, pts, joint_values)


def segment_cam(
    params: np.ndarray,
    sid: str,
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
) -> np.ndarray:
    part, _role, a, b = segs[sid]
    local = articulate_segment(sid, part, np.vstack([a, b]), params, segs)
    return fit.transform(params, local)


def point_cam(params: np.ndarray, local: np.ndarray) -> np.ndarray:
    return fit.transform(params, local[None, :])[0]


def contact_chord_seed(
    init: np.ndarray,
    contacts: list[dict[str, str]],
    palm: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Align the two local contact endpoints to the observed palm chord.

    Two point correspondences determine translation and two rotational degrees
    of freedom.  The remaining twist is fixed by choosing the shortest rotation
    from the Stage-3 orientation, so the initializer adds no historical pose or
    case-specific solved state.
    """
    by_hand = {row.get("hand_side", ""): row for row in contacts}
    required = ("left_hand", "right_hand")
    if any(hand not in by_hand or hand not in palm for hand in required):
        return init.copy(), {"used": False}

    local = np.asarray(
        [
            [ff(by_hand[hand], "chair_local_x"), ff(by_hand[hand], "chair_local_y"), ff(by_hand[hand], "chair_local_z")]
            for hand in required
        ],
        dtype=float,
    )
    target = np.asarray([palm[hand] for hand in required], dtype=float)
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(target)):
        return init.copy(), {"used": False}

    return align_contact_chord(init, local, target, fit.BASE_LOCAL_TO_CAM)


def contact_by_frame(contact_rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    for row in contact_rows:
        if row.get("contact_active") == "1":
            out.setdefault(int(row["frame"]), []).append(row)
    return out


def build_2d_targets(
    rows: list[dict[str, str]],
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    segment_ids: list[str],
) -> dict[int, dict[str, np.ndarray]]:
    targets: dict[int, dict[str, np.ndarray]] = {}
    for row in rows:
        params = pose_params(row)
        k = np.array([[ff(row, "fx"), 0.0, ff(row, "cx")], [0.0, ff(row, "fy"), ff(row, "cy")], [0.0, 0.0, 1.0]], dtype=float)
        frame_targets: dict[str, np.ndarray] = {}
        for sid in segment_ids:
            cam = segment_cam(params, sid, segs)
            uv, valid = project_cam(cam, k)
            if np.all(valid):
                frame_targets[sid] = uv
        targets[int(row["frame"])] = frame_targets
    return targets


def refine_contact_chord_gauge(
    chord_seed: np.ndarray,
    contacts: list[dict[str, str]],
    palm: dict[str, np.ndarray],
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    target_2d: dict[str, np.ndarray],
    segment_ids: list[str],
    k: np.ndarray,
    sigma_px: float,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Resolve the free twist around a two-point contact chord from 2D cues."""
    by_hand = {row.get("hand_side", ""): row for row in contacts}
    hands = ("left_hand", "right_hand")
    if any(hand not in by_hand or hand not in palm for hand in hands):
        return chord_seed, {"used": False}
    local = np.asarray(
        [[ff(by_hand[hand], "chair_local_x"), ff(by_hand[hand], "chair_local_y"), ff(by_hand[hand], "chair_local_z")] for hand in hands],
        dtype=float,
    )
    target = np.asarray([palm[hand] for hand in hands], dtype=float)
    axis = target[1] - target[0]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8 or not np.all(np.isfinite(local)):
        return chord_seed, {"used": False}
    axis /= axis_norm
    target_mid = np.mean(target, axis=0)
    local_mid = np.mean(local, axis=0)
    r_chord = Rotation.from_rotvec(chord_seed[:3]).as_matrix() @ fit.BASE_LOCAL_TO_CAM

    def params_from_state(state: np.ndarray) -> np.ndarray:
        twist = Rotation.from_rotvec(axis * state[0]).as_matrix()
        rotation = twist @ r_chord
        out = chord_seed.copy()
        out[:3] = Rotation.from_matrix(rotation @ fit.BASE_LOCAL_TO_CAM.T).as_rotvec()
        out[3:6] = target_mid - rotation @ local_mid
        out[6:8] = state[1:3]
        return out

    def residual(state: np.ndarray) -> np.ndarray:
        params = params_from_state(state)
        vals: list[float] = []
        for sid in segment_ids:
            if sid not in target_2d:
                continue
            cam = segment_cam(params, sid, segs)
            uv, valid = project_cam(cam, k)
            if np.all(valid):
                vals.extend(((uv - target_2d[sid]).reshape(-1) / sigma_px).tolist())
        return np.asarray(vals, dtype=float)

    state0 = np.asarray([0.0, chord_seed[6], chord_seed[7]], dtype=float)
    initial_residual = residual(state0)
    initial_cost = float(np.sum(np.sqrt(1.0 + initial_residual * initial_residual) - 1.0))
    result = least_squares(
        residual,
        state0,
        bounds=(np.asarray([-math.pi, -np.inf, -np.inf]), np.asarray([math.pi, np.inf, np.inf])),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
    )
    return params_from_state(result.x), {
        "used": True,
        "success": bool(result.success),
        "twist_rad": float(result.x[0]),
        "initial_cost": initial_cost,
        "cost": float(result.cost),
        "cost_nonincreasing": bool(float(result.cost) <= initial_cost + 1e-9),
        "rear_joint_angle": float(result.x[1]),
        "seat_joint_angle": float(result.x[2]),
    }


def optimize_contact_frame(
    frame: int,
    ref_row: dict[str, str],
    init_row: dict[str, str],
    contacts: list[dict[str, str]],
    joints_frame: np.ndarray,
    segs: dict[str, tuple[str, str, np.ndarray, np.ndarray]],
    target_2d: dict[str, np.ndarray],
    segment_ids: list[str],
    prev: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    ref = pose_params(ref_row)
    init = pose_params(init_row)
    left_palm, right_palm = palm_centers(joints_frame)
    palm = {"left_hand": left_palm, "right_hand": right_palm}
    chord_metrics: dict[str, float | bool] = {"used": False}
    if args.contact_chord_init:
        init, chord_metrics = contact_chord_seed(init, contacts, palm)
    k = np.array([[ff(ref_row, "fx"), 0.0, ff(ref_row, "cx")], [0.0, ff(ref_row, "fy"), ff(ref_row, "cy")], [0.0, 0.0, 1.0]], dtype=float)
    gauge_metrics: dict[str, float | bool] = {"used": False}
    if args.contact_chord_2d_gauge and bool(chord_metrics.get("used")):
        init, gauge_metrics = refine_contact_chord_gauge(
            init, contacts, palm, segs, target_2d, segment_ids, k, args.sigma_px, args.max_nfev
        )
    if args.preserve_contact_chord_constraint and bool(gauge_metrics.get("used")):
        gaps = []
        for contact in contacts:
            hand = palm.get(contact.get("hand_side", ""))
            local = np.asarray([ff(contact, "chair_local_x"), ff(contact, "chair_local_y"), ff(contact, "chair_local_z")], dtype=float)
            if hand is not None and np.all(np.isfinite(local)):
                gaps.append(float(np.linalg.norm(point_cam(init, local) - hand)))
        return init, {
            "frame": frame,
            "cost": float(gauge_metrics.get("cost", math.nan)),
            "median_contact_gap_m": float(np.median(gaps)) if gaps else math.nan,
            "contact_chord_initializer": chord_metrics,
            "contact_chord_2d_gauge": gauge_metrics,
            "contact_chord_constraint_preserved": True,
        }
    joint_vals = ref[6:8].copy()
    x0 = init.copy() if args.optimize_articulation else init[:6].copy()
    # The anchor-interp input already carries the physically plausible contact
    # depth. Center the local SE(3) search around it; use the 2D ref pose as a
    # projection residual/prior, not as a hard box center, otherwise tz gets
    # clipped back to the 2D-only depth and contact is lost.
    root_lower = init[:6] + np.array([-args.rot_bound, -args.rot_bound, -args.rot_bound, -args.xy_bound, -args.xy_bound, -args.z_bound])
    root_upper = init[:6] + np.array([args.rot_bound, args.rot_bound, args.rot_bound, args.xy_bound, args.xy_bound, args.z_bound])
    if args.optimize_articulation:
        bounds = (np.concatenate([root_lower, [-np.inf, -np.inf]]), np.concatenate([root_upper, [np.inf, np.inf]]))
    else:
        bounds = (root_lower, root_upper)

    def full_params(x: np.ndarray) -> np.ndarray:
        return x if args.optimize_articulation else np.concatenate([x, joint_vals])

    def residual(x: np.ndarray) -> np.ndarray:
        params = full_params(x)
        vals: list[float] = []
        for sid in segment_ids:
            if sid not in target_2d:
                continue
            cam = segment_cam(params, sid, segs)
            uv, valid = project_cam(cam, k)
            if np.all(valid):
                vals.extend((args.w_2d * (uv - target_2d[sid]).reshape(-1) / args.sigma_px).tolist())
        for contact in contacts:
            hand = palm.get(contact.get("hand_side", ""))
            if hand is None:
                continue
            local = np.asarray([ff(contact, "chair_local_x"), ff(contact, "chair_local_y"), ff(contact, "chair_local_z")], dtype=float)
            if not np.all(np.isfinite(local)):
                continue
            anchor = point_cam(params, local)
            vals.extend((args.w_contact * (anchor - hand) / args.sigma_contact_m).tolist())
        vals.extend((args.w_prior_rot * (x[:3] - ref[:3]) / args.rot_bound).tolist())
        vals.extend((args.w_prior_xy * (x[3:5] - ref[3:5]) / args.xy_bound).tolist())
        vals.extend((args.w_prior_z * (x[5:6] - init[5:6]) / args.z_bound).tolist())
        if prev is not None:
            vals.extend((args.w_temporal * (x[:6] - prev[:6]) / np.array([0.06, 0.06, 0.06, 0.08, 0.08, 0.12])).tolist())
        return np.asarray(vals, dtype=float)

    res = least_squares(residual, np.clip(x0, bounds[0], bounds[1]), bounds=bounds, loss="soft_l1", f_scale=1.0, max_nfev=args.max_nfev)
    out = full_params(res.x)
    gaps = []
    for contact in contacts:
        hand = palm.get(contact.get("hand_side", ""))
        if hand is None:
            continue
        local = np.asarray([ff(contact, "chair_local_x"), ff(contact, "chair_local_y"), ff(contact, "chair_local_z")], dtype=float)
        if np.all(np.isfinite(local)):
            gaps.append(float(np.linalg.norm(point_cam(out, local) - hand)))
    metrics = {
        "frame": frame,
        "cost": float(res.cost),
        "median_contact_gap_m": float(np.median(gaps)) if gaps else math.nan,
        "contact_chord_initializer": chord_metrics,
        "contact_chord_2d_gauge": gauge_metrics,
    }
    return out, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-pose-csv", type=Path, required=True, help="2D-authoritative pose. Projections from this pose are strong targets.")
    ap.add_argument("--init-pose-csv", type=Path, required=True, help="Initial pose, usually endpoint anchorinterp.")
    ap.add_argument("--contacts-csv", type=Path, required=True)
    ap.add_argument("--segments-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--metrics-json", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path, default=Path("third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--human-result", type=Path, default=Path("samples_known_object/05_chair/results/gvhmr/result.pkl"))
    ap.add_argument("--sigma-px", type=float, default=5.0)
    ap.add_argument("--sigma-contact-m", type=float, default=0.045)
    ap.add_argument("--w-2d", type=float, default=1.0)
    ap.add_argument("--w-contact", type=float, default=0.85)
    ap.add_argument("--w-prior-rot", type=float, default=0.35)
    ap.add_argument("--w-prior-xy", type=float, default=0.25)
    ap.add_argument("--w-prior-z", type=float, default=0.20)
    ap.add_argument("--w-temporal", type=float, default=0.50)
    ap.add_argument("--rot-bound", type=float, default=0.16)
    ap.add_argument("--xy-bound", type=float, default=0.20)
    ap.add_argument("--z-bound", type=float, default=0.45)
    ap.add_argument("--max-nfev", type=int, default=90)
    ap.add_argument(
        "--contact-chord-init",
        action="store_true",
        help="Initialize each active frame from current local endpoints and observed palms before SE(3) refinement.",
    )
    ap.add_argument(
        "--optimize-articulation",
        action="store_true",
        help="Keep the current Stage-3 joint values in the state so 2D residuals can re-fit them after root contact alignment.",
    )
    ap.add_argument(
        "--contact-chord-2d-gauge",
        action="store_true",
        help="Resolve the chord's free twist and joint initialization from current Stage-3 2D projections.",
    )
    ap.add_argument(
        "--preserve-contact-chord-constraint",
        action="store_true",
        help="Emit the 2D-gauged chord solution without relaxing its exact two-point contact constraint.",
    )
    args = ap.parse_args()

    ref_rows = read_rows(args.ref_pose_csv)
    init_rows = read_rows(args.init_pose_csv)
    ref_by_frame = {int(r["frame"]): r for r in ref_rows}
    init_by_frame = {int(r["frame"]): r for r in init_rows}
    contacts = contact_by_frame(read_rows(args.contacts_csv))
    segs = fit.load_segments(args.segments_csv)
    segment_ids = [sid for sid in [
        "front_leg_left", "front_leg_right", "rear_leg_left", "rear_leg_right",
        "backrest_top_edge", "backrest_bottom_edge", "seat_front_edge", "seat_hinge_edge",
        "left_side_lower_stretcher", "right_side_lower_stretcher", "front_lower_stretcher", "rear_lower_stretcher",
    ] if sid in segs]
    targets_2d = build_2d_targets(ref_rows, segs, segment_ids)
    human = read_human_result(args.human_result)
    joints = build_body_joints(args.body_model_root, human)

    out_by_frame: dict[int, dict[str, str]] = {}
    metrics = []
    prev: np.ndarray | None = None
    active_frames = sorted(contacts)
    first_active = active_frames[0] if active_frames else None
    last_active = active_frames[-1] if active_frames else None
    optimized_params: dict[int, np.ndarray] = {}
    for frame in active_frames:
        ref_row = ref_by_frame[frame]
        init_row = init_by_frame[frame]
        params, m = optimize_contact_frame(
            frame,
            ref_row,
            init_row,
            contacts[frame],
            joints[frame - 1],
            segs,
            targets_2d.get(frame, {}),
            segment_ids,
            prev,
            args,
        )
        optimized_params[frame] = params
        metrics.append(m)
        prev = params

    for row in ref_rows:
        frame = int(row["frame"])
        if frame in optimized_params:
            out_by_frame[frame] = update_pose_quat(row, optimized_params[frame], "chair_twohand_endpoint_se3_contact_refined")
        elif first_active is not None and frame < first_active:
            out_by_frame[frame] = update_pose_quat(row, optimized_params[first_active], "chair_twohand_endpoint_se3_boundary_freeze_prefix")
        elif last_active is not None and frame > last_active:
            out_by_frame[frame] = update_pose_quat(row, optimized_params[last_active], "chair_twohand_endpoint_se3_boundary_freeze_suffix")
        else:
            out_by_frame[frame] = dict(row)

    out_rows = [out_by_frame[int(row["frame"])] for row in ref_rows]
    output_fields = list(ref_rows[0].keys())
    if args.preserve_contact_chord_constraint:
        for row in out_rows:
            row["pose_lock_reason"] = "current_run_contact_chord_constraint_gate"
        if "pose_lock_reason" not in output_fields:
            output_fields.append("pose_lock_reason")
    write_csv(args.out_csv, out_rows, output_fields)
    constraint_checks: dict[str, bool] = {}
    if args.preserve_contact_chord_constraint:
        geometry_errors = []
        for metric in metrics:
            chord = metric.get("contact_chord_initializer", {})
            gauge = metric.get("contact_chord_2d_gauge", {})
            if chord.get("used") and math.isfinite(float(metric.get("median_contact_gap_m", math.nan))):
                geometry_errors.append(
                    abs(float(metric["median_contact_gap_m"]) - float(chord.get("theoretical_min_gap_m", math.inf)))
                )
        constraint_checks = {
            "all_active_frames_initialized": len(metrics) == len(active_frames) and all(
                bool(metric.get("contact_chord_initializer", {}).get("used")) for metric in metrics
            ),
            "all_2d_gauge_solves_succeeded": len(metrics) == len(active_frames) and all(
                bool(metric.get("contact_chord_2d_gauge", {}).get("success")) for metric in metrics
            ),
            "all_2d_gauge_costs_nonincreasing": len(metrics) == len(active_frames) and all(
                bool(metric.get("contact_chord_2d_gauge", {}).get("cost_nonincreasing")) for metric in metrics
            ),
            "contact_reaches_geometric_lower_bound": len(geometry_errors) == len(active_frames)
            and max(geometry_errors, default=math.inf) <= 1e-6,
        }
    summary = {
        "ref_pose_csv": str(args.ref_pose_csv),
        "init_pose_csv": str(args.init_pose_csv),
        "contacts_csv": str(args.contacts_csv),
        "segments_csv": str(args.segments_csv),
        "out_csv": str(args.out_csv),
        "active_frames": len(active_frames),
        "first_active": first_active,
        "last_active": last_active,
        "contact_chord_initializer_enabled": bool(args.contact_chord_init),
        "articulation_optimized": bool(args.optimize_articulation),
        "contact_chord_2d_gauge_enabled": bool(args.contact_chord_2d_gauge),
        "contact_chord_constraint_preserved": bool(args.preserve_contact_chord_constraint),
        "median_optimized_contact_gap_m": float(np.nanmedian([m["median_contact_gap_m"] for m in metrics])) if metrics else math.nan,
        "p90_optimized_contact_gap_m": float(np.nanpercentile([m["median_contact_gap_m"] for m in metrics], 90)) if metrics else math.nan,
        "contact_chord_constraint_gate": {
            "pass": bool(constraint_checks) and all(constraint_checks.values()),
            "checks": constraint_checks,
        },
        "metrics": metrics,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
