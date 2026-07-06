from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...core.base.config import CaseProfile
from ...core.base.camera import backproject_uvz
from ...core.base.io import copy_file, read_csv, write_csv, write_json
from ...core.base.schema import stage_paths
from ..refinement.sequence_se3_optimizer import smooth_quaternion_pose_sequence
from ..refinement.policies import (
    anchor_depth,
    anchor_propagate_freeze,
    backproject_xy,
    line_contact_lock,
    small_se3,
    stable_grasp_anchor,
    table_freeze,
)
from ..refinement.policies import generic_line_physical_smooth


COMPATIBILITY_SEED_BUILDERS = {
    "anchor_depth": anchor_depth,
    "anchor_propagate_freeze": anchor_propagate_freeze,
    "backproject_xy": backproject_xy,
    "line_contact_lock": line_contact_lock,
    "small_se3": small_se3,
    "stable_grasp_anchor": stable_grasp_anchor,
    "table_freeze": table_freeze,
}


def _has_se3(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and {"tx", "ty", "tz", "qw", "qx", "qy", "qz"}.issubset(rows[0])


def _read_pose(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def _copy_contacts(profile: CaseProfile) -> str:
    paths = stage_paths(profile)
    if paths["contact_candidates"].exists():
        return str(copy_file(paths["contact_candidates"], paths["object_contact_points"]))
    write_csv(paths["object_contact_points"], [])
    return str(paths["object_contact_points"])


def _run_compatibility_seed_builders(profile: CaseProfile) -> list[dict[str, object]]:
    """Run legacy refinement policies as seed/residual builders before final SE3 smoothing."""
    reports: list[dict[str, object]] = []
    for name in profile.refinement_policies():
        if name in {"sequence_se3_optimizer", "generic_line_physical_smooth"}:
            continue
        module = COMPATIBILITY_SEED_BUILDERS.get(name)
        if module is None:
            reports.append({"component": name, "enabled": False, "reason": "unknown_compatibility_seed_builder"})
            continue
        try:
            report = module.apply(profile)
            reports.append({"component": name, "enabled": True, "report": report})
        except Exception as exc:
            reports.append({"component": name, "enabled": False, "reason": str(exc)})
            raise
    return reports


def _trust_weights(profile: CaseProfile, rows: list[dict[str, str]]) -> list[float]:
    paths = stage_paths(profile)
    frames = [int(float(row.get("frame", "0") or 0)) for row in rows]
    trust = [1.0] * len(rows)
    if paths["anchor_state"].exists():
        by_frame: dict[int, float] = {}
        for row in read_csv(paths["anchor_state"]):
            try:
                fr = int(float(row.get("frame", "0") or 0))
                pose_anchor = int(float(row.get("pose_anchor_allowed", "0") or 0)) == 1
                observed = int(float(row.get("contact_observed", row.get("contact_active", "0")) or 0)) == 1
                conf = float(row.get("contact_confidence", row.get("contact_conf", "1.0")) or 1.0)
            except Exception:
                continue
            if pose_anchor and observed:
                by_frame[fr] = max(by_frame.get(fr, 0.0), min(1.0, max(0.25, conf)))
        if by_frame:
            trust = [max(0.35, by_frame.get(fr, 0.70)) for fr in frames]
    return trust


def _apply_vlm_trust_suppression(profile: CaseProfile, rows: list[dict[str, str]], trust: list[float]) -> list[float]:
    paths = stage_paths(profile)
    frames = [int(float(row.get("frame", "0") or 0)) for row in rows]
    out = list(trust)
    gate_root = paths["vlm_dir"] / "stage4" / "vlm_gates.csv"
    if gate_root.exists():
        disabled: set[int] = set()
        for row in read_csv(gate_root):
            if row.get("is_effective") != "1":
                continue
            if row.get("pass_gate") not in {"reject", "unclear"}:
                continue
            if row.get("allow_pose_refine") == "0":
                try:
                    disabled.add(int(float(row.get("frame", "0") or 0)))
                except Exception:
                    pass
        out = [min(t, 0.15) if fr in disabled else t for fr, t in zip(frames, out)]
    return out


def _quat(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row.get("qw", "1") or 1),
        float(row.get("qx", "0") or 0),
        float(row.get("qy", "0") or 0),
        float(row.get("qz", "0") or 0),
    )


def _quat_step(a: dict[str, str], b: dict[str, str]) -> float:
    aqw, aqx, aqy, aqz = _quat(a)
    bqw, bqx, bqy, bqz = _quat(b)
    an = math.sqrt(aqw * aqw + aqx * aqx + aqy * aqy + aqz * aqz) or 1.0
    bn = math.sqrt(bqw * bqw + bqx * bqx + bqy * bqy + bqz * bqz) or 1.0
    dot = abs((aqw * bqw + aqx * bqx + aqy * bqy + aqz * bqz) / (an * bn))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _contact_flags(profile: CaseProfile) -> dict[int, dict[str, bool]]:
    paths = stage_paths(profile)
    out: dict[int, dict[str, bool]] = {}
    if not paths["anchor_state"].exists():
        return out
    for row in read_csv(paths["anchor_state"]):
        try:
            fr = int(float(row.get("frame", "0") or 0))
            observed = int(float(row.get("contact_observed", row.get("contact_active", "0")) or 0)) == 1
            persistent = int(float(row.get("contact_persistent", "0") or 0)) == 1
            pose_anchor = int(float(row.get("pose_anchor_allowed", "0") or 0)) == 1
        except Exception:
            continue
        cur = out.setdefault(fr, {"observed": False, "persistent": False, "pose_anchor": False})
        cur["observed"] = cur["observed"] or observed
        cur["persistent"] = cur["persistent"] or persistent
        cur["pose_anchor"] = cur["pose_anchor"] or pose_anchor
    return out


def _local_anchor_from_row(profile: CaseProfile, row: dict[str, str]) -> tuple[float, float, float] | None:
    try:
        lx = row.get("stable_local_x", "")
        ly = row.get("stable_local_y", "")
        lz = row.get("stable_local_z", "")
        if lx not in {"", None} and ly not in {"", None} and lz not in {"", None}:
            return float(lx), float(ly), float(lz)
        s_raw = row.get("stable_object_local_s", row.get("stable_local_s", row.get("observed_object_local_s", row.get("observed_local_s", ""))))
        if s_raw not in {"", None} and profile.data.get("line_object"):
            s = float(s_raw)
            length_m = float(profile.data.get("line_object", {}).get("length_m", 1.0))
            return (s - 0.5) * length_m, 0.0, 0.0
    except Exception:
        return None
    return None


def _target_anchor_from_row(profile: CaseProfile, row: dict[str, str]) -> tuple[float, float, float] | None:
    try:
        x = row.get("anchor_x_m", row.get("palm_x", ""))
        y = row.get("anchor_y_m", row.get("palm_y", ""))
        z = row.get("anchor_z_m", row.get("palm_z", ""))
        if x not in {"", None} and y not in {"", None} and z not in {"", None}:
            return float(x), float(y), float(z)
        u = row.get("anchor_u", row.get("palm_u", ""))
        v = row.get("anchor_v", row.get("palm_v", ""))
        if u not in {"", None} and v not in {"", None} and z not in {"", None}:
            return backproject_uvz(float(u), float(v), float(z), profile.camera)
    except Exception:
        return None
    return None


def _anchor_constraints(profile: CaseProfile, rows: list[dict[str, str]]) -> tuple[list[list[tuple[tuple[float, float, float], tuple[float, float, float], float]]], dict[str, object]]:
    paths = stage_paths(profile)
    by_frame = {int(float(row.get("frame", "0") or 0)): i for i, row in enumerate(rows)}
    constraints: list[list[tuple[tuple[float, float, float], tuple[float, float, float], float]]] = [[] for _ in rows]
    if not paths["anchor_state"].exists():
        return constraints, {"enabled": False, "reason": "missing anchor_state"}
    used = 0
    skipped = 0
    for row in read_csv(paths["anchor_state"]):
        try:
            if row.get("pose_anchor_allowed") != "1" or row.get("contact_observed", "1") == "0":
                continue
            fr = int(float(row.get("frame", "0") or 0))
            idx = by_frame.get(fr)
            if idx is None:
                continue
            local = _local_anchor_from_row(profile, row)
            target = _target_anchor_from_row(profile, row)
            if local is None or target is None:
                skipped += 1
                continue
            conf = float(row.get("contact_confidence", row.get("contact_conf", "1.0")) or 1.0)
            constraints[idx].append((local, target, max(0.2, min(1.0, conf))))
            used += 1
        except Exception:
            skipped += 1
    return constraints, {
        "enabled": used > 0,
        "constraints": used,
        "skipped_rows": skipped,
        "source": str(paths["anchor_state"]),
        "policy": "generic SE3 anchor residual from anchor_state local coordinate to observed 3D/UVZ anchor",
    }


def _row_rotation(row: dict[str, str]) -> Rotation:
    q = np.asarray(
        [
            float(row.get("qx", "0") or 0),
            float(row.get("qy", "0") or 0),
            float(row.get("qz", "0") or 0),
            float(row.get("qw", "1") or 1),
        ],
        dtype=float,
    )
    if np.linalg.norm(q) < 1e-8:
        q = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    return Rotation.from_quat(q / np.linalg.norm(q))


def _quat_fields(rot: Rotation) -> dict[str, str]:
    qx, qy, qz, qw = rot.as_quat()
    return {"qw": f"{qw:.6f}", "qx": f"{qx:.6f}", "qy": f"{qy:.6f}", "qz": f"{qz:.6f}"}


def _anchor_error(row: dict[str, str], constraints: list[tuple[tuple[float, float, float], tuple[float, float, float], float]]) -> float:
    if not constraints:
        return 0.0
    rot = _row_rotation(row).as_matrix()
    t = np.asarray([float(row.get("tx", "0") or 0), float(row.get("ty", "0") or 0), float(row.get("tz", "0") or 0)], dtype=float)
    errs = []
    for local, target, _weight in constraints:
        errs.append(float(np.linalg.norm(t + rot @ np.asarray(local, dtype=float) - np.asarray(target, dtype=float))))
    return float(np.median(errs)) if errs else 0.0


def _apply_anchor_pose_prior(
    rows: list[dict[str, str]],
    constraints: list[list[tuple[tuple[float, float, float], tuple[float, float, float], float]]],
    *,
    max_translation_step: float = 3.00,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    out: list[dict[str, str]] = []
    adjusted = 0
    before_err: list[float] = []
    after_err: list[float] = []
    for row, frame_constraints in zip(rows, constraints):
        rec = dict(row)
        if rec.get("line_contact_pose_prior") == "1":
            out.append(rec)
            continue
        if not frame_constraints:
            out.append(rec)
            continue
        before_err.append(_anchor_error(rec, frame_constraints))
        locals_arr = np.asarray([c[0] for c in frame_constraints], dtype=float)
        targets_arr = np.asarray([c[1] for c in frame_constraints], dtype=float)
        weights = np.asarray([max(0.05, c[2]) for c in frame_constraints], dtype=float)
        cur_rot = _row_rotation(rec)
        new_rot = cur_rot
        if len(frame_constraints) >= 2:
            local_center = np.average(locals_arr, axis=0, weights=weights)
            target_center = np.average(targets_arr, axis=0, weights=weights)
            local_vec = locals_arr - local_center
            target_vec = targets_arr - target_center
            if np.linalg.norm(local_vec) > 1e-6 and np.linalg.norm(target_vec) > 1e-6:
                try:
                    new_rot, _rmsd = Rotation.align_vectors(target_vec, local_vec, weights=weights)
                except Exception:
                    new_rot = cur_rot
        trans_targets = targets_arr - (new_rot.as_matrix() @ locals_arr.T).T
        target_t = np.average(trans_targets, axis=0, weights=weights)
        cur_t = np.asarray([float(rec.get("tx", "0") or 0), float(rec.get("ty", "0") or 0), float(rec.get("tz", "0") or 0)], dtype=float)
        delta = target_t - cur_t
        norm = float(np.linalg.norm(delta))
        if norm > max_translation_step:
            target_t = cur_t + delta / max(norm, 1e-8) * max_translation_step
        rec.update(
            {
                "tx": f"{target_t[0]:.6f}",
                "ty": f"{target_t[1]:.6f}",
                "tz": f"{max(0.5, target_t[2]):.6f}",
                "anchor_pose_prior": "1",
                "anchor_pose_prior_count": str(len(frame_constraints)),
                "source": (rec.get("source", "") + "+generic_anchor_pose_prior").strip("+"),
            }
        )
        rec.update(_quat_fields(new_rot))
        after_err.append(_anchor_error(rec, frame_constraints))
        adjusted += 1
        out.append(rec)
    return out, {
        "enabled": adjusted > 0,
        "adjusted_rows": adjusted,
        "median_before_m": f"{float(np.median(before_err)):.6f}" if before_err else "0.000000",
        "median_after_m": f"{float(np.median(after_err)):.6f}" if after_err else "0.000000",
        "policy": "generic per-frame rigid anchor pose prior before sequence smoothing",
    }


def _annotate_anchor_residuals(
    rows: list[dict[str, str]],
    residuals: list[dict[str, str]],
    constraints: list[list[tuple[tuple[float, float, float], tuple[float, float, float], float]]],
) -> None:
    for row, residual, frame_constraints in zip(rows, residuals, constraints):
        residual["anchor_residual_enabled"] = "1" if frame_constraints else "0"
        residual["anchor_residual_count"] = str(len(frame_constraints))
        residual["anchor_residual_m"] = f"{_anchor_error(row, frame_constraints):.6f}" if frame_constraints else "0.000000"


def _motion_regime_rows(profile: CaseProfile, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    flags = _contact_flags(profile)
    speeds = [0.0] * len(rows)
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur = rows[i]
        dt = math.sqrt(
            (float(cur.get("tx", "0") or 0) - float(prev.get("tx", "0") or 0)) ** 2
            + (float(cur.get("ty", "0") or 0) - float(prev.get("ty", "0") or 0)) ** 2
            + (float(cur.get("tz", "0") or 0) - float(prev.get("tz", "0") or 0)) ** 2
        )
        speeds[i] = dt + 0.10 * _quat_step(prev, cur)
    sorted_speeds = sorted(speeds)
    median = sorted_speeds[len(sorted_speeds) // 2] if sorted_speeds else 0.0
    hi = max(0.09, sorted_speeds[int(0.80 * (len(sorted_speeds) - 1))] if len(sorted_speeds) > 1 else 0.09)
    static_thr = 0.012
    out: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        fr = int(float(row.get("frame", "0") or 0))
        c = flags.get(fr, {})
        speed = speeds[i]
        if c.get("persistent") and c.get("pose_anchor") and speed >= hi * 1.25:
            regime = "sustained_grasp_high_speed"
        elif c.get("persistent") and c.get("pose_anchor"):
            regime = "sustained_grasp"
        elif c.get("observed") and not c.get("persistent"):
            regime = "short_impact"
        elif speed <= static_thr:
            regime = "static_hold"
        elif speed >= hi * 1.25:
            regime = "high_speed"
        elif median > static_thr and abs(speed - median) <= max(0.015, 0.35 * median):
            regime = "uniform_motion"
        elif speed < max(0.035, median * 0.75):
            regime = "low_speed"
        else:
            regime = "free_motion"
        out.append(
            {
                "frame": row.get("frame", ""),
                "time": row.get("time", ""),
                "motion_regime": regime,
                "motion_speed_score": f"{speed:.6f}",
                "contact_observed": "1" if c.get("observed") else "0",
                "contact_persistent": "1" if c.get("persistent") else "0",
                "pose_anchor_allowed": "1" if c.get("pose_anchor") else "0",
                "source": "generic_motion_regime_from_se3_and_anchor_state",
            }
        )
    return out


def _regime_weights(regime_rows: list[dict[str, object]], base_trust: list[float]) -> tuple[list[float], list[float], list[float]]:
    trust: list[float] = []
    point_smooth: list[float] = []
    point_accel: list[float] = []
    for row, t in zip(regime_rows, base_trust):
        regime = row.get("motion_regime")
        if regime == "sustained_grasp_high_speed":
            trust.append(max(t, 0.86))
            point_smooth.append(0.38)
            point_accel.append(0.34)
        elif regime == "sustained_grasp":
            trust.append(max(t, 0.88))
            point_smooth.append(1.05)
            point_accel.append(1.20)
        elif regime == "short_impact":
            trust.append(max(t, 0.72))
            point_smooth.append(0.42)
            point_accel.append(0.38)
        elif regime == "high_speed":
            trust.append(max(t, 0.78))
            point_smooth.append(0.28)
            point_accel.append(0.24)
        elif regime == "uniform_motion":
            trust.append(t)
            point_smooth.append(0.82)
            point_accel.append(0.70)
        elif regime == "static_hold":
            trust.append(max(t, 0.80))
            point_smooth.append(1.45)
            point_accel.append(1.80)
        elif regime == "low_speed":
            trust.append(t)
            point_smooth.append(1.15)
            point_accel.append(1.25)
        else:
            trust.append(t)
            point_smooth.append(0.72)
            point_accel.append(0.62)
    smooth = [
        max(0.05, min(2.5, 0.5 * (point_smooth[i - 1] + point_smooth[i])))
        for i in range(1, len(point_smooth))
    ]
    accel = [
        max(0.05, min(2.5, (point_accel[i] + point_accel[i + 1] + point_accel[i + 2]) / 3.0))
        for i in range(0, max(0, len(point_accel) - 2))
    ]
    return trust, smooth, accel


def _stage_audit_reweight_sources(profile: CaseProfile) -> set[str]:
    path = stage_paths(profile)["stage_audit_dir"] / "stage4" / "stage_audit_gates.csv"
    if not path.exists():
        return set()
    sources: set[str] = set()
    for row in read_csv(path):
        if row.get("stage") != "stage4":
            continue
        if row.get("residual_reweight") == "1" or row.get("repair_action") == "reweight_and_rerun_optimizer":
            check_type = str(row.get("check_type", ""))
            if check_type.startswith("vlm_gate"):
                sources.add("vlm")
            elif check_type.startswith("llm_csv"):
                sources.add("llm")
            else:
                sources.add("machine")
    return sources


def _apply_feedback_reweight(
    trust: list[float],
    smooth: list[float],
    accel: list[float],
    *,
    sources: set[str],
) -> tuple[list[float], list[float], list[float], str]:
    if not sources:
        return trust, smooth, accel, ""
    trust_factor = 1.0
    smooth_factor = 1.0
    accel_factor = 1.0
    if "machine" in sources:
        trust_factor *= 0.82
        smooth_factor *= 1.45
        accel_factor *= 1.65
    if "vlm" in sources:
        trust_factor *= 0.86
        smooth_factor *= 1.20
        accel_factor *= 1.25
    if "llm" in sources:
        trust_factor *= 0.90
        smooth_factor *= 1.10
        accel_factor *= 1.15
    return (
        [max(0.20, min(8.0, t * trust_factor)) for t in trust],
        [max(0.05, min(3.5, w * smooth_factor)) for w in smooth],
        [max(0.05, min(3.8, w * accel_factor)) for w in accel],
        "stage_audit_residual_reweight:" + "+".join(sorted(sources)),
    )


def _candidate_score(audit_rows: list[dict[str, object]]) -> tuple[int, int, float, float]:
    spike_count = 0
    smooth_count = 0
    max_jump = 0.0
    max_delta_rot = 0.0
    for row in audit_rows:
        if row.get("visual_spike") == "1" or row.get("contact_spike") == "1" or row.get("smoothness_spike") == "1":
            spike_count += 1
        if row.get("smoothness_spike") == "1":
            smooth_count += 1
        try:
            max_jump = max(max_jump, float(row.get("final_translation_jump_m", "0") or 0))
        except Exception:
            pass
        try:
            max_delta_rot = max(max_delta_rot, float(row.get("delta_rotation_rad", "0") or 0))
        except Exception:
            pass
    return spike_count, smooth_count, max_jump, max_delta_rot


def _candidate_source_sets(sources: set[str]) -> list[set[str]]:
    ordered: list[set[str]] = []

    def add(item: set[str]) -> None:
        if item not in ordered:
            ordered.append(set(item))

    if not sources:
        add(set())
        return ordered
    add(set())
    add({"machine"})
    if "llm" in sources:
        add({"machine", "llm"})
    if "vlm" in sources:
        add({"machine", "vlm"})
    add(set(sources))
    return ordered


def _line_seed_score(path: Path) -> tuple[int, int, float]:
    rows = read_csv(path) if path.exists() else []
    spike_count = 0
    smooth_count = 0
    max_jump = 0.0
    for row in rows:
        if row.get("visual_spike") == "1" or row.get("contact_spike") == "1" or row.get("smoothness_spike") == "1":
            spike_count += 1
        if row.get("smoothness_spike") == "1":
            smooth_count += 1
        try:
            max_jump = max(max_jump, float(row.get("final_translation_jump_m", row.get("translation_jump_m", "0")) or 0))
        except Exception:
            pass
    return spike_count, smooth_count, max_jump


def _has_effective_vlm_gates(profile: CaseProfile) -> bool:
    vlm_dir = stage_paths(profile)["vlm_dir"]
    for stage in ("stage1", "stage2", "stage3", "stage4"):
        path = vlm_dir / stage / "vlm_gates.csv"
        if not path.exists():
            continue
        if any(row.get("is_effective") == "1" for row in read_csv(path)):
            return True
    return False


def _run_line_seed_builder(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    if not _has_effective_vlm_gates(profile):
        return generic_line_physical_smooth.apply(profile)

    candidates: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    base_pose_rows = read_csv(paths["object_pose"]) if paths["object_pose"].exists() else []
    for ignore_vlm, suffix in ((True, "_candidate_ignore_vlm"), (False, "_candidate_with_vlm")):
        write_csv(paths["object_pose"], base_pose_rows)
        report = generic_line_physical_smooth.apply(profile, ignore_vlm_gates=ignore_vlm, metrics_suffix=suffix)
        pose_rows = read_csv(paths["object_pose"]) if paths["object_pose"].exists() else []
        residual_rows = read_csv(paths["physical_smooth_residuals"]) if paths["physical_smooth_residuals"].exists() else []
        audit_rows = read_csv(paths["pose_jump_audit"]) if paths["pose_jump_audit"].exists() else []
        score = _line_seed_score(paths["pose_jump_audit"])
        candidate = {
            "ignore_vlm_gates": ignore_vlm,
            "metrics_suffix": suffix,
            "score": {
                "spike_count": score[0],
                "smoothness_spike_count": score[1],
                "max_translation_jump_m": f"{score[2]:.6f}",
            },
            "report": report,
            "_pose_rows": pose_rows,
            "_residual_rows": residual_rows,
            "_audit_rows": audit_rows,
            "_score_tuple": score,
        }
        candidates.append(candidate)
        if selected is None or score < selected["_score_tuple"]:  # type: ignore[index]
            selected = candidate
    assert selected is not None
    write_csv(paths["object_pose"], selected["_pose_rows"])  # type: ignore[arg-type]
    write_csv(paths["physical_smooth_residuals"], selected["_residual_rows"])  # type: ignore[arg-type]
    write_csv(paths["pose_jump_audit"], selected["_audit_rows"])  # type: ignore[arg-type]
    public_candidates = []
    for item in candidates:
        public_candidates.append(
            {
                "ignore_vlm_gates": item["ignore_vlm_gates"],
                "metrics_suffix": item["metrics_suffix"],
                "score": item["score"],
            }
        )
    chosen = {
        "component": "generic_line_physical_smooth_seed_candidate_selector",
        "enabled": True,
        "selected_ignore_vlm_gates": selected["ignore_vlm_gates"],
        "selected_score": selected["score"],
        "candidates": public_candidates,
        "policy": "line seed builder runs with and without VLM-disabled frames; hard pose-jump audit selects the non-degrading seed",
    }
    write_json(profile.result_dir / "generic_line_physical_smooth_metrics.json", chosen)
    return chosen


def _audit_rows(
    before: list[dict[str, str]],
    after: list[dict[str, str]],
    se3_audit: list[dict[str, str]],
    regime_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    audit_by_frame = {int(float(row.get("frame", "0") or 0)): row for row in se3_audit}
    regime_by_frame = {
        int(float(row.get("frame", "0") or 0)): str(row.get("motion_regime", ""))
        for row in regime_rows
    }
    prev_raw_t: tuple[float, float, float] | None = None
    prev_final_t: tuple[float, float, float] | None = None
    after_by_frame = {int(float(row.get("frame", "0") or 0)): row for row in after}
    for row in before:
        fr = int(float(row.get("frame", "0") or 0))
        raw_t = (
            float(row.get("tx", "0") or 0),
            float(row.get("ty", "0") or 0),
            float(row.get("tz", "0") or 0),
        )
        final_row = after_by_frame.get(fr, row)
        final_t = (
            float(final_row.get("tx", "0") or 0),
            float(final_row.get("ty", "0") or 0),
            float(final_row.get("tz", "0") or 0),
        )
        raw_jump = 0.0 if prev_raw_t is None else math.sqrt(sum((a - b) ** 2 for a, b in zip(raw_t, prev_raw_t)))
        final_jump = 0.0 if prev_final_t is None else math.sqrt(sum((a - b) ** 2 for a, b in zip(final_t, prev_final_t)))
        prev_raw_t = raw_t
        prev_final_t = final_t
        audit = audit_by_frame.get(fr, {})
        regime = regime_by_frame.get(fr, "")
        if regime in {"high_speed", "sustained_grasp_high_speed", "short_impact"}:
            visual_thr = 0.30
            velocity_thr = 0.42
            accel_thr = 0.34
        elif regime in {"static_hold", "low_speed"}:
            visual_thr = 0.10
            velocity_thr = 0.12
            accel_thr = 0.12
        else:
            visual_thr = 0.18
            velocity_thr = 0.22
            accel_thr = 0.22
        try:
            final_velocity = float(audit.get("final_velocity_score", "0") or 0)
            final_accel = float(audit.get("final_accel_score", "0") or 0)
        except Exception:
            final_velocity = velocity_thr if audit.get("se3_velocity_spike", "0") == "1" else 0.0
            final_accel = accel_thr if audit.get("se3_accel_spike", "0") == "1" else 0.0
        smooth_spike = final_velocity > velocity_thr or final_accel > accel_thr
        out.append(
            {
                "frame": str(fr),
                "visual_spike": "1" if final_jump > visual_thr else "0",
                "raw_visual_spike": "1" if raw_jump > 0.18 else "0",
                "contact_spike": "0",
                "smoothness_spike": "1" if smooth_spike else "0",
                "freeze_or_propagate": "optimize",
                "motion_regime": regime,
                "pre_smooth_translation_jump_m": f"{raw_jump:.6f}",
                "final_translation_jump_m": f"{final_jump:.6f}",
                "visual_spike_threshold_m": f"{visual_thr:.6f}",
                "velocity_spike_threshold": f"{velocity_thr:.6f}",
                "accel_spike_threshold": f"{accel_thr:.6f}",
                "se3_velocity_spike": audit.get("se3_velocity_spike", "0"),
                "se3_accel_spike": audit.get("se3_accel_spike", "0"),
                "raw_se3_velocity_spike": audit.get("raw_se3_velocity_spike", "0"),
                "raw_se3_accel_spike": audit.get("raw_se3_accel_spike", "0"),
                "delta_translation_m": audit.get("delta_translation_m", "0.000000"),
                "delta_rotation_rad": audit.get("delta_rotation_rad", "0.000000"),
            }
        )
    return out


def _decision_rows(
    rows: list[dict[str, str]],
    regime_rows: list[dict[str, object]],
    trust: list[float],
    smooth: list[float],
    accel: list[float],
    *,
    pose_lock_reason: str = "",
    feedback_reweight_reason: str = "",
    visual_prior_reason: str = "",
    overlay_prior_reason: str = "",
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for i, (row, regime, weight) in enumerate(zip(rows, regime_rows, trust)):
        out.append(
            {
                "frame": row.get("frame", ""),
                "motion_regime": regime.get("motion_regime", ""),
                "sequence_se3_optimizer_enabled": "1",
                "visual_residual_enabled": "1",
                "contact_anchor_residual_enabled": "1" if weight >= 0.35 else "0",
                "depth_residual_enabled": "1",
                "geometry_residual_enabled": "1",
                "velocity_residual_enabled": "1",
                "acceleration_residual_enabled": "1",
                "static_freeze_residual_enabled": "1",
                "boundary_freeze_interpolation_enabled": "1",
                "static_tail_freeze_enabled": "1",
                "trust_weight": f"{weight:.6f}",
                "smooth_weight": f"{(smooth[i - 1] if i > 0 and smooth else (smooth[0] if smooth else 1.0)):.6f}",
                "accel_weight": f"{(accel[min(max(i - 1, 0), len(accel) - 1)] if accel else 1.0):.6f}",
                "feedback_reoptimized": "1",
                "feedback_reweight_reason": feedback_reweight_reason,
                "visual_prior_reason": visual_prior_reason,
                "overlay_prior_reason": overlay_prior_reason,
                "pose_lock_reason": pose_lock_reason,
                "decision_source": "generic_sequence_se3_mainline",
            }
        )
    return out


def _apply_overlay_pose_prior_weights(
    rows: list[dict[str, str]],
    trust: list[float],
    smooth: list[float],
    accel: list[float],
) -> tuple[list[float], list[float], list[float], str]:
    flags = [row.get("line_contact_pose_prior") == "1" or row.get("generic_geometry_pose_prior") == "1" for row in rows]
    if not any(flags):
        return trust, smooth, accel, ""
    out_trust = [max(t, 8.0) if ok else t for t, ok in zip(trust, flags)]
    out_smooth = list(smooth)
    for i in range(len(out_smooth)):
        if flags[i] and flags[i + 1]:
            out_smooth[i] = min(out_smooth[i], 0.015)
    out_accel = list(accel)
    for i in range(len(out_accel)):
        if flags[i] and flags[i + 1] and flags[i + 2]:
            out_accel[i] = min(out_accel[i], 0.015)
    return out_trust, out_smooth, out_accel, "line_contact_pose_prior"


def _float_or_none(value: object) -> float | None:
    try:
        if value in {"", None}:
            return None
        out = float(value)  # type: ignore[arg-type]
        if not math.isfinite(out):
            return None
        return out
    except Exception:
        return None


def _projection_uv(row: dict[str, str], camera: dict[str, float]) -> tuple[float | None, float | None]:
    u = _float_or_none(row.get("u_proj"))
    v = _float_or_none(row.get("v_proj"))
    if u is not None and v is not None:
        return u, v
    tx = _float_or_none(row.get("tx"))
    ty = _float_or_none(row.get("ty"))
    tz = _float_or_none(row.get("tz"))
    if tx is None or ty is None or tz is None or abs(tz) < 1e-8:
        return None, None
    return camera["fx"] * tx / tz + camera["cx"], camera["fy"] * ty / tz + camera["cy"]


def _set_depth_preserving_overlay(row: dict[str, str], z: float, camera: dict[str, float]) -> None:
    u, v = _projection_uv(row, camera)
    if u is None or v is None:
        row["tz"] = f"{z:.6f}"
        return
    tx, ty, tz = backproject_uvz(u, v, z, camera)
    row.update({"tx": f"{tx:.9f}", "ty": f"{ty:.9f}", "tz": f"{tz:.9f}"})
    row.setdefault("u_proj", f"{u:.6f}")
    row.setdefault("v_proj", f"{v:.6f}")


def _angle_span(values: list[float]) -> float:
    if not values:
        return 0.0
    unit = np.asarray([[math.cos(v), math.sin(v)] for v in values], dtype=float)
    mean = unit.mean(axis=0)
    if np.linalg.norm(mean) < 1e-8:
        return math.pi
    center = math.atan2(float(mean[1]), float(mean[0]))
    return max(abs(math.atan2(math.sin(v - center), math.cos(v - center))) for v in values)


def _line_overlay_static_tail_indices(
    rows: list[dict[str, str]],
    *,
    min_frames: int,
    uv_thr_px: float,
    angle_thr_rad: float,
    length_rel_thr: float,
    max_depth_range_m: float,
) -> list[int]:
    flags = [row.get("line_contact_pose_prior") == "1" for row in rows]
    if len(rows) < min_frames or not all(flags[-min_frames:]):
        return []
    start = len(rows) - min_frames
    while start > 0 and flags[start - 1]:
        cand = list(range(start - 1, len(rows)))
        us = [_float_or_none(rows[i].get("u_proj")) for i in cand]
        vs = [_float_or_none(rows[i].get("v_proj")) for i in cand]
        angles = [_float_or_none(rows[i].get("contact_lock_angle_rad")) for i in cand]
        lens = [_float_or_none(rows[i].get("contact_lock_pixel_len")) for i in cand]
        depths = [_float_or_none(rows[i].get("tz")) for i in cand]
        if any(v is None for v in us + vs + angles + lens + depths):
            break
        u_vals = [float(v) for v in us if v is not None]
        v_vals = [float(v) for v in vs if v is not None]
        a_vals = [float(v) for v in angles if v is not None]
        l_vals = [float(v) for v in lens if v is not None]
        d_vals = [float(v) for v in depths if v is not None]
        l_med = float(np.median(l_vals)) if l_vals else 0.0
        overlay_stable = (
            max(u_vals) - min(u_vals) <= uv_thr_px
            and max(v_vals) - min(v_vals) <= uv_thr_px
            and _angle_span(a_vals) <= angle_thr_rad
            and (max(l_vals) - min(l_vals)) / max(1.0, l_med) <= length_rel_thr
        )
        depth_unstable = max(d_vals) - min(d_vals) >= max_depth_range_m
        if not overlay_stable:
            break
        if not depth_unstable and len(cand) > min_frames:
            break
        start -= 1
    idxs = list(range(start, len(rows)))
    if len(idxs) < min_frames:
        return []
    us = [_float_or_none(rows[i].get("u_proj")) for i in idxs]
    vs = [_float_or_none(rows[i].get("v_proj")) for i in idxs]
    angles = [_float_or_none(rows[i].get("contact_lock_angle_rad")) for i in idxs]
    lens = [_float_or_none(rows[i].get("contact_lock_pixel_len")) for i in idxs]
    if any(v is None for v in us + vs + angles + lens):
        return []
    u_vals = [float(v) for v in us if v is not None]
    v_vals = [float(v) for v in vs if v is not None]
    a_vals = [float(v) for v in angles if v is not None]
    l_vals = [float(v) for v in lens if v is not None]
    l_med = float(np.median(l_vals)) if l_vals else 0.0
    if (
        max(u_vals) - min(u_vals) <= uv_thr_px
        and max(v_vals) - min(v_vals) <= uv_thr_px
        and _angle_span(a_vals) <= angle_thr_rad
        and (max(l_vals) - min(l_vals)) / max(1.0, l_med) <= length_rel_thr
    ):
        return idxs
    return []


def _apply_line_overlay_depth_stabilizer(
    profile: CaseProfile,
    rows: list[dict[str, str]],
    config: dict[str, object],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    if not any(row.get("line_contact_pose_prior") == "1" for row in rows):
        return rows, {"enabled": False, "reason": "no line_contact_pose_prior rows"}
    out = [dict(row) for row in rows]
    camera = profile.camera
    max_step = float(config.get("line_overlay_depth_max_step_m", 0.35))
    outlier_range = float(config.get("line_overlay_depth_local_outlier_m", 0.85))
    adjusted = 0
    clamped = 0
    for i, row in enumerate(out):
        if row.get("line_contact_pose_prior") != "1":
            continue
        z = _float_or_none(row.get("tz"))
        if z is None:
            continue
        local: list[float] = []
        for j in range(max(0, i - 2), min(len(out), i + 3)):
            if out[j].get("line_contact_pose_prior") != "1":
                continue
            v = _float_or_none(out[j].get("tz"))
            if v is not None and 0.5 <= v <= 12.0:
                local.append(v)
        if len(local) >= 3:
            med = float(np.median(local))
            if abs(z - med) > outlier_range:
                _set_depth_preserving_overlay(row, med, camera)
                row["line_overlay_depth_stabilized"] = "1"
                row["line_overlay_depth_stabilizer_reason"] = "local_depth_outlier_clamp"
                adjusted += 1
                clamped += 1
    tail = _line_overlay_static_tail_indices(
        out,
        min_frames=int(config.get("line_overlay_static_tail_min_frames", 18)),
        uv_thr_px=float(config.get("line_overlay_static_tail_uv_px", 8.0)),
        angle_thr_rad=float(config.get("line_overlay_static_tail_angle_rad", 0.08)),
        length_rel_thr=float(config.get("line_overlay_static_tail_length_rel", 0.04)),
        max_depth_range_m=float(config.get("line_overlay_static_tail_depth_range_m", 0.30)),
    )
    tail_freeze = 0
    tail_anchor_depth_used = False
    if tail:
        tail_frames = {int(float(out[i].get("frame", "0") or 0)) for i in tail}
        anchor_depth_values: list[float] = []
        anchor_path = stage_paths(profile)["anchor_state"]
        if anchor_path.exists():
            for anchor in read_csv(anchor_path):
                try:
                    fr = int(float(anchor.get("frame", "0") or 0))
                except Exception:
                    continue
                if fr not in tail_frames:
                    continue
                if anchor.get("contact_observed") != "1" or anchor.get("pose_anchor_allowed") != "1":
                    continue
                if anchor.get("contact_persistent", "1") != "1":
                    continue
                z_anchor = _float_or_none(anchor.get("anchor_z_m"))
                if z_anchor is not None and 0.5 <= z_anchor <= 12.0:
                    anchor_depth_values.append(z_anchor)
        if len(anchor_depth_values) >= max(2, len(tail) // 4):
            depth_candidates = anchor_depth_values
            tail_anchor_depth_used = True
        else:
            depth_candidates = [
                _float_or_none(out[i].get("z_from_pixel_len_m"))
                for i in tail
            ]
        depth_values = [v for v in depth_candidates if v is not None and 0.5 <= v <= 12.0]
        if len(depth_values) < max(3, len(tail) // 3):
            depth_values = [
                v for i in tail
                if (v := _float_or_none(out[i].get("tz"))) is not None and 0.5 <= v <= 12.0
            ]
        if depth_values:
            z_tail = float(np.median(depth_values))
            for i in tail:
                _set_depth_preserving_overlay(out[i], z_tail, camera)
                out[i]["line_overlay_depth_stabilized"] = "1"
                out[i]["line_overlay_static_tail_freeze"] = "1"
                out[i]["line_overlay_depth_stabilizer_reason"] = (
                    "static_tail_anchor_depth_freeze"
                    if tail_anchor_depth_used
                    else "static_tail_overlay_preserving_depth_freeze"
                )
                tail_freeze += 1
            adjusted += tail_freeze
    prev_idx: int | None = None
    for i, row in enumerate(out):
        if row.get("line_contact_pose_prior") != "1":
            prev_idx = None
            continue
        if prev_idx is None:
            prev_idx = i
            continue
        z_prev = _float_or_none(out[prev_idx].get("tz"))
        z = _float_or_none(row.get("tz"))
        if z_prev is None or z is None:
            prev_idx = i
            continue
        dz = z - z_prev
        if abs(dz) > max_step:
            z_new = z_prev + math.copysign(max_step, dz)
            _set_depth_preserving_overlay(row, z_new, camera)
            row["line_overlay_depth_stabilized"] = "1"
            row["line_overlay_depth_stabilizer_reason"] = row.get("line_overlay_depth_stabilizer_reason") or "depth_velocity_clamp"
            adjusted += 1
        prev_idx = i
    for row in out:
        row.setdefault("line_overlay_depth_stabilized", "0")
        row.setdefault("line_overlay_static_tail_freeze", "0")
        row.setdefault("line_overlay_depth_stabilizer_reason", "")
    return out, {
        "enabled": adjusted > 0,
        "adjusted_rows": adjusted,
        "local_outlier_clamped_rows": clamped,
        "static_tail_freeze_rows": tail_freeze,
        "static_tail_anchor_depth_used": tail_anchor_depth_used,
        "policy": "preserve line overlay center while smoothing/freezing ambiguous monocular depth",
    }


def apply(profile: CaseProfile) -> dict[str, object]:
    paths = stage_paths(profile)
    requested_legacy_policies = [
        name for name in profile.refinement_policies() if name != "sequence_se3_optimizer"
    ]

    if not paths["object_pose_init"].exists():
        raise FileNotFoundError(f"Missing Stage3 SE3 pose init: {paths['object_pose_init']}")
    copy_file(paths["object_pose_init"], paths["object_pose"])
    copy_file(paths["object_pose"], paths["object_pose_pre_smooth"])

    compatibility_reports = _run_compatibility_seed_builders(profile)
    line_seed_report: dict[str, object] = {"enabled": False, "reason": "not_line_object"}
    if profile.data.get("line_object"):
        line_seed_report = _run_line_seed_builder(profile)
    if paths["object_pose"].exists():
        copy_file(paths["object_pose"], paths["object_pose_pre_smooth"])

    rows = _read_pose(paths["object_pose"])
    if not _has_se3(rows):
        raise ValueError(f"Stage4 requires SE3 pose rows in {paths['object_pose']}")
    config = profile.data.get("sequence_se3_optimizer", {}) if isinstance(profile.data.get("sequence_se3_optimizer", {}), dict) else {}
    base_trust = _trust_weights(profile, rows)
    regime_rows = _motion_regime_rows(profile, rows)
    write_csv(paths["motion_regime"], regime_rows)
    trust, smooth_weight, accel_weight = _regime_weights(regime_rows, base_trust)
    anchor_constraints, anchor_metrics = _anchor_constraints(profile, rows)
    rows, anchor_prior_metrics = _apply_anchor_pose_prior(
        rows,
        anchor_constraints,
        max_translation_step=float(config.get("anchor_prior_max_translation_step", 3.00)),
    )
    anchor_prior_trust = float(config.get("anchor_prior_trust", 8.0))
    trust = [
        max(t, anchor_prior_trust) if frame_constraints else t
        for t, frame_constraints in zip(trust, anchor_constraints)
    ]
    pose_lock_reason = ""
    feedback_reweight_reason = ""
    visual_prior_reason = ""
    overlay_prior_reason = ""
    trust, smooth_weight, accel_weight, overlay_prior_reason = _apply_overlay_pose_prior_weights(
        rows,
        trust,
        smooth_weight,
        accel_weight,
    )
    reweight_sources = _stage_audit_reweight_sources(profile)
    candidate_reports: list[dict[str, object]] = []
    best: tuple[
        tuple[int, int, float, float],
        set[str],
        list[dict[str, str]],
        list[dict[str, str]],
        list[float],
        list[float],
        list[float],
        str,
        list[dict[str, object]],
    ] | None = None
    for candidate_sources in _candidate_source_sets(reweight_sources):
        source_trust = _apply_vlm_trust_suppression(profile, rows, trust) if "vlm" in candidate_sources else list(trust)
        cand_trust, cand_smooth, cand_accel, cand_reason = _apply_feedback_reweight(
            source_trust,
            list(smooth_weight),
            list(accel_weight),
            sources=candidate_sources,
        )
        cand_rows, cand_audit = smooth_quaternion_pose_sequence(
            rows,
            trust_weight=cand_trust,
            smooth_weight=cand_smooth,
            accel_weight=cand_accel,
            translation_scales=tuple(config.get("translation_scales", (0.065, 0.065, 0.095))),
            rotation_scale=float(config.get("rotation_scale", 0.070)),
            prior_w=float(config.get("prior_w", 0.34)),
            smooth_w=float(config.get("smooth_w", 0.95)),
            accel_w=float(config.get("accel_w", 4.2)),
            anchor_w=float(config.get("anchor_w", 1.0)),
        )
        _annotate_anchor_residuals(cand_rows, cand_audit, anchor_constraints)
        cand_jump_audit = _audit_rows(rows, cand_rows, cand_audit, regime_rows)
        score = _candidate_score(cand_jump_audit)
        candidate_reports.append(
            {
                "sources": sorted(candidate_sources),
                "feedback_reweight_reason": cand_reason,
                "score": {
                    "spike_count": score[0],
                    "smoothness_spike_count": score[1],
                    "max_translation_jump_m": f"{score[2]:.6f}",
                    "max_delta_rotation_rad": f"{score[3]:.6f}",
                },
                "trust_first": f"{cand_trust[0]:.6f}" if cand_trust else "",
                "smooth_first": f"{cand_smooth[0]:.6f}" if cand_smooth else "",
                "accel_first": f"{cand_accel[0]:.6f}" if cand_accel else "",
            }
        )
        cand = (score, candidate_sources, cand_rows, cand_audit, cand_trust, cand_smooth, cand_accel, cand_reason, cand_jump_audit)
        if best is None or score < best[0] or (score == best[0] and len(candidate_sources) > len(best[1])):
            best = cand
    if best is None:
        out_rows, se3_audit = smooth_quaternion_pose_sequence(rows)
        selected_sources: set[str] = set()
        feedback_reweight_reason = ""
        selected_jump_audit = _audit_rows(rows, out_rows, se3_audit, regime_rows)
    else:
        _score, selected_sources, out_rows, se3_audit, trust, smooth_weight, accel_weight, feedback_reweight_reason, selected_jump_audit = best
    rejected_sources = sorted(reweight_sources - selected_sources)
    write_csv(paths["object_pose"], out_rows)
    write_csv(paths["physical_smooth_residuals"], se3_audit)
    write_csv(paths["pose_jump_audit"], selected_jump_audit)
    write_csv(
        paths["optimizer_decisions"],
        _decision_rows(
            rows,
            regime_rows,
            trust,
            smooth_weight,
            accel_weight,
            pose_lock_reason=pose_lock_reason,
            feedback_reweight_reason=feedback_reweight_reason,
            visual_prior_reason=visual_prior_reason,
            overlay_prior_reason=overlay_prior_reason,
        ),
    )
    contacts_path = _copy_contacts(profile)

    metrics = {
        "component": "generic_sequence_se3_mainline",
        "case_name": profile.case_name,
        "compatibility_adapters": compatibility_reports,
        "legacy_refinement_policies_ignored": [],
        "legacy_refinement_policies_as_seed_builders": requested_legacy_policies,
        "line_object_special_refinement": bool(profile.data.get("line_object")),
        "line_object_seed_builder": line_seed_report,
        "object_pose_pre_smooth": str(paths["object_pose_pre_smooth"]),
        "object_pose": str(paths["object_pose"]),
        "object_contact_points": contacts_path,
        "motion_regime": str(paths["motion_regime"]),
        "physical_smooth_residuals": str(paths["physical_smooth_residuals"]),
        "pose_jump_audit": str(paths["pose_jump_audit"]),
        "optimizer_decisions": str(paths["optimizer_decisions"]),
        "rows": len(out_rows),
        "smoothed_rows": sum(1 for row in se3_audit if row.get("se3_smoothed") == "1"),
        "pose_lock_reason": pose_lock_reason,
        "residuals": {
            "visual": True,
            "contact_anchor": True,
            "depth": True,
            "geometry": True,
            "velocity": True,
            "acceleration": True,
            "static_freeze": True,
            "boundary_freeze_interpolation": True,
            "static_tail_freeze": True,
            "motion_regime_reweighting": True,
            "feedback_reoptimization": True,
        },
        "anchor_residual": anchor_metrics,
        "anchor_pose_prior": anchor_prior_metrics,
        "line_overlay_depth_stabilizer": {"enabled": False, "reason": "replaced_by_generic_line_physical_smooth_seed_builder"},
        "feedback_reweight_sources": sorted(selected_sources),
        "requested_feedback_reweight_sources": sorted(reweight_sources),
        "rejected_feedback_sources": rejected_sources,
        "feedback_candidate_selection": {
            "policy": "choose the candidate with the fewest hard pose_jump_audit spikes; ties prefer more accepted audit sources",
            "candidates": candidate_reports,
        },
        "feedback_reweight_reason": feedback_reweight_reason,
        "visual_prior_reason": visual_prior_reason,
        "overlay_prior_reason": overlay_prior_reason,
        "policy": "compatibility refinement policies run as seed/residual builders; line objects run generic_line_physical_smooth as seed/visual prior; final output is unified through smooth_quaternion_pose_sequence",
    }
    write_json(paths["stage4_metrics"], metrics)
    return metrics
