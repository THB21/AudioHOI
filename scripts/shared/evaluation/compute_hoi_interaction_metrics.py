#!/usr/bin/env python3
"""HOI-level interaction metrics — the human-object layer missing from the
object-centric final evaluator (docs/final_result_evaluation_method_en.md).

Where Yixin's final evaluator scores the OBJECT (SE3 schema, anchor drift, jumps),
this scores the INTERACTION between the reconstructed human (SMPL-X body + stitched
HaMeR fingers, contact-refined when available) and the object trajectory:

  A. Penetration family (GT-free plausibility, HOI-PAGE/PROX-style):
     pen_frame_ratio      fraction of frames with any part point inside the surface
     pen_depth_mean_mm    mean penetration depth over penetrating (frame, point)s
     pen_depth_max_mm     worst single-point penetration
     non_collision_ratio  1 - pen_frame_ratio

  B. Contact family:
     contact_frame_ratio  fraction of frames with the closest part point inside the
                          touch band [surface - eps_in, surface + band] (is the object
                          actually being touched when it should be held/hit?)
     contact_gap_mm       mean |closest-point distance - radius| over expected-contact
                          frames (expected = contact_frame/audio flags in the trajectory
                          CSV, else audio contact_records with relevant target parts)
     part_correct_ratio   at expected-contact frames with a named target part
                          (VLM/audio records target_entity), whether the geometrically
                          closest part matches the named part
  C. Temporal family:
     object jerk          mean ||Δ³T||² (smoothness; co-report with timing)
     grasp_stability_mm   during contact intervals: std of the closest-point distance
                          (a held object should keep a constant hand-surface distance)

Object geometry: sphere (balls, radius_m column) or capsule (stick: SE3 pose columns +
--object-length-m/--object-radius-m) via object_geometry.py.

Human params precedence (same as renderers): contact_refine > stitched hands > GVHMR.

Usage:
  conda run -n gvhmr python scripts/shared/evaluation/compute_hoi_interaction_metrics.py \
      --sample-dir samples_known_object/11_stick \
      --trajectory-csv .../object_pose.csv --object-type capsule --object-length-m 1.86 \
      [--label stick] [--params auto|raw|stitched|refined] \
      [--out-csv samples_known_object/hoi_interaction_evaluation/hoi_summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "human_ball" / "contact"))
from object_geometry import geometry_from_rows, quat_to_rotmat  # noqa: E402
from refine_body_pose_contact import _PART_POINTS  # noqa: E402
from hoi_paper_metrics import box_corners, evaluate_hoi_paper_metrics  # noqa: E402

_RECORD_PARTS = {"left_hand", "right_hand", "left_foot", "right_foot"}


def _first_existing(*cands: Path) -> Path | None:
    for c in cands:
        if c.exists():
            return c
    return None


def load_human_params(results_dir: Path, mode: str = "auto") -> tuple[dict, str]:
    # Case dirs use either `contact_refine`/`hands`/`gvhmr` (mug, stick, samples/*) or the
    # `human_`-prefixed variants (basketball, football, chair under samples_known_object).
    refined = _first_existing(
        results_dir / "contact_refine" / "contact_refined_smplx_params.pkl",
        results_dir / "human_contact_refine" / "contact_refined_smplx_params.pkl")
    stitched = _first_existing(
        results_dir / "hands" / "stitched_smplx_params.pkl",
        results_dir / "human_hands" / "stitched_smplx_params.pkl")
    raw_pkl = _first_existing(results_dir / "gvhmr" / "result.pkl",
                              results_dir / "human_gvhmr" / "result.pkl")
    raw = pickle.load(raw_pkl.open("rb"))
    p = raw["smpl_params_incam"]
    params = {
        "body_pose": np.asarray(p["body_pose"], dtype=np.float32),
        "betas": np.asarray(p["betas"], dtype=np.float32),
        "global_orient": np.asarray(p["global_orient"], dtype=np.float32),
        "transl": np.asarray(p["transl"], dtype=np.float32),
    }
    src = "gvhmr"
    if mode in ("auto", "stitched", "refined") and stitched is not None:
        st = pickle.load(stitched.open("rb"))
        params.update({k: np.asarray(v) for k, v in st.items() if isinstance(v, np.ndarray)})
        src = "stitched"
    if mode in ("auto", "refined") and refined is not None:
        rf = pickle.load(refined.open("rb"))
        params.update({k: np.asarray(v) for k, v in rf.items() if isinstance(v, np.ndarray)})
        src = "contact_refine"
    return params, src


def build_part_points(params: dict, body_model_root: Path, n: int, device: str,
                      use_verts: bool = True, parts: list[str] | None = None,
                      return_joints: bool = False, return_vertices: bool = False):
    """Per-part point clouds. use_verts=True (default) uses dense SMPL-X mesh vertices
    (hand/foot flesh) so penetration/contact are measured on the actual surface, not
    21 skeleton points; parts may add 'pelvis'/'back' for body-object contact.

    return_joints=True additionally returns the full SMPL-X joint array [N, J, 3]
    (metres, camera frame) for the human temporal metrics (HOI-PAGE smoothness,
    CARI4D acceleration)."""
    import smplx
    betas = params["betas"]
    if betas.shape[0] == 1:
        betas = np.repeat(betas, n, axis=0)
    model = smplx.create(str(body_model_root), model_type="smplx", gender="neutral", ext="npz",
                         use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=n).to(device)
    kwargs = dict(
        body_pose=torch.from_numpy(params["body_pose"][:n]).to(device),
        betas=torch.from_numpy(betas[:n].astype(np.float32)).to(device),
        global_orient=torch.from_numpy(params["global_orient"][:n]).to(device),
        transl=torch.from_numpy(params["transl"][:n]).to(device),
        return_verts=use_verts,
    )
    for side in ("left", "right"):
        key = f"{side}_hand_pose"
        if key in params:
            kwargs[key] = torch.from_numpy(params[key][:n].astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(**kwargs)
    plist = parts or list(_PART_POINTS)
    joints = out.joints.detach().cpu().numpy() if return_joints else None
    if use_verts:
        from part_vertices import part_vertex_indices
        pv = part_vertex_indices(body_model_root, plist)
        V = out.vertices
        pts = {name: V[:, torch.as_tensor(pv[name], device=device), :] for name in plist}
    else:
        pts = {name: out.joints[:, _PART_POINTS[name], :] for name in plist if name in _PART_POINTS}
    values = [pts]
    if return_joints:
        values.append(joints)
    if return_vertices:
        values.append(out.vertices.detach())
    return tuple(values) if len(values) > 1 else pts


def expected_contacts(rows: list[dict], sample_dir: Path, n: int) -> tuple[np.ndarray, list[str], list[int]]:
    """Per-frame expected-contact flag + named target part ('' when unnamed) +
    audio event frames (0-based, from contact_records) for the audio-windowed metrics."""
    flag = np.zeros(n, dtype=bool)
    part = ["" for _ in range(n)]
    events: list[int] = []
    for r in rows:
        i = int(r["frame"]) - 1
        if not (0 <= i < n):
            continue
        if (int(r.get("contact_frame", 0) or 0) == 1 or int(r.get("audio_anchor", 0) or 0) == 1
                or int(r.get("audio_contact_frame", 0) or 0) == 1
                or int(r.get("human_contact_state", 0) or 0) == 1):
            flag[i] = True
        ap = str(r.get("active_part", "") or "")
        if ap:
            part[i] = ap
    records = _first_existing(
        sample_dir / "results" / "audio_semantics" / "contact_records.csv",
        sample_dir / "results" / "human_audio_semantics" / "contact_records.csv")
    if records is not None:
        with records.open() as f:
            for r in csv.DictReader(f):
                i = int(round(float(r.get("frame", 0)))) - 1
                tgt = str(r.get("target_entity", "") or "")
                if tgt not in _RECORD_PARTS:
                    # sustained grasps (mug/stick) name the holder in stable_entity,
                    # target_entity stays 'none' for non-impact taps
                    tgt = str(r.get("stable_entity", "") or "")
                relevant = str(r.get("relevant", "1") or "1") not in ("0", "false", "")
                state = str(r.get("contact_state", "") or "")
                grasp = str(r.get("interval_id", "-1") or "-1") not in ("-1", "")
                if 0 <= i < n and tgt in _RECORD_PARTS and (relevant or state in ("contact", "keep_grasp") or grasp):
                    flag[i] = True
                    events.append(i)
                    if not part[i]:
                        part[i] = tgt
    return flag, part, sorted(set(events))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--trajectory-csv", type=Path, required=True)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--params", choices=["auto", "raw", "stitched", "refined"], default="auto")
    ap.add_argument("--skeleton-only", action="store_true",
                    help="measure on 21 skeleton points per hand (legacy) instead of dense mesh verts")
    ap.add_argument("--body-parts", type=str, default=None,
                    help="extra parts for penetration/contact: comma of pelvis,back")
    ap.add_argument("--object-type", choices=["sphere", "capsule", "mesh", "mesh_sdf"], default="sphere")
    ap.add_argument("--object-length-m", type=float, default=None)
    ap.add_argument("--object-radius-m", type=float, default=None)
    ap.add_argument("--object-axis-local", choices=["x", "y", "z"], default="x")
    ap.add_argument("--object-mesh-path", type=str, default=None,
                    help="URDF-frame GLB for --object-type mesh (bake_urdf_to_glb --keep-origin)")
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--pen-eps-mm", type=float, default=3.0,
                    help="penetration counts only past this depth (mesh/skeleton slack)")
    ap.add_argument("--touch-band-mm", type=float, default=15.0,
                    help="closest point within surface±band counts as touching")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    results_dir = args.sample_dir / "results"
    with args.trajectory_csv.open() as f:
        rows = list(csv.DictReader(f))
    params, src = load_human_params(results_dir, args.params)
    n = min(params["body_pose"].shape[0], max(int(r["frame"]) for r in rows))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    geom = geometry_from_rows(rows, n, object_type=args.object_type,
                              length_m=args.object_length_m, radius_m=args.object_radius_m,
                              axis_local=args.object_axis_local,
                              mesh_path=args.object_mesh_path, device=device)
    plist = list(_PART_POINTS) + [p for p in (args.body_parts.split(",") if args.body_parts else []) if p]
    parts, joints, all_human_vertices = build_part_points(
        params, args.body_model_root, n, device,
        use_verts=not args.skeleton_only, parts=plist,
        return_joints=True, return_vertices=True)
    valid = geom.valid.cpu().numpy()

    # distances [part][N] min over part points; and all-point matrix for penetration depth
    dmin = {}
    pen_depth = []  # penetration depths (mm) over all (frame, part, point) past eps
    pen_pts_per_frame = np.zeros(n)   # # penetrating query points per frame (HOI-PAGE vertex ratio)
    total_query_pts = 0               # total query points across parts
    radius = geom.radius
    valid_t = torch.from_numpy(valid).to(device)
    for name, pts in parts.items():
        d = geom.point_distances(pts)                      # [N, P]
        dmin[name] = d.min(dim=1).values.cpu().numpy()
        depth = (radius[:, None] - d).clamp_min(0.0) * 1000.0
        pen_mask = depth > args.pen_eps_mm                 # [N, P] point inside past slack
        pen_pts_per_frame += (pen_mask & valid_t[:, None]).sum(dim=1).cpu().numpy()
        total_query_pts += pts.shape[1]
        depth = depth[valid_t]
        pen_depth.append(depth[depth > args.pen_eps_mm].cpu().numpy())
    pen_depth = np.concatenate(pen_depth) if pen_depth else np.array([])

    # Paper-compatible physical plausibility uses every SMPL-X vertex, rather than
    # only the semantic hand/foot subsets used by the diagnostic metrics above.
    # Mesh proximity can otherwise materialize N_frames * N_vertices auxiliary
    # arrays inside trimesh and exceed RAM. Chunking vertices is mathematically
    # identical and also works for analytic sphere/capsule geometry.
    paper_sdf_chunks = []
    paper_vertex_chunk = 512 if args.object_type == "mesh" else all_human_vertices.shape[1]
    for vertex_start in range(0, all_human_vertices.shape[1], paper_vertex_chunk):
        vertex_end = min(all_human_vertices.shape[1], vertex_start + paper_vertex_chunk)
        paper_sdf_chunks.append(
            (geom.point_distances(all_human_vertices[:, vertex_start:vertex_end])
             - geom.radius[:, None]).detach().cpu()
        )
    paper_signed_distances = torch.cat(paper_sdf_chunks, dim=1).numpy()
    # HOI-PAGE non-collision ratio: fraction of human (part) vertices NOT inside the
    # object surface, averaged over valid frames (their metric is per-vertex, not per-frame).
    if valid.any() and total_query_pts:
        frac_non_col = 1.0 - pen_pts_per_frame[valid] / float(total_query_pts)
        non_collision_vertex_ratio = round(float(frac_non_col.mean()), 5)
    else:
        non_collision_vertex_ratio = None

    rad = radius.cpu().numpy()
    closest = np.min(np.stack(list(dmin.values())), axis=0)   # [N] nearest human point
    closest_part = np.array(list(dmin.keys()))[np.argmin(np.stack(list(dmin.values())), axis=0)]

    pen_frame = np.zeros(n, dtype=bool)
    for name in dmin:
        pen_frame |= (dmin[name] < rad - args.pen_eps_mm / 1000.0)
    pen_frame &= valid

    band = args.touch_band_mm / 1000.0
    touching = (closest < rad + band) & valid

    exp_flag, exp_part, event_frames = expected_contacts(rows, args.sample_dir, n)
    exp_flag &= valid
    gaps = np.abs(closest[exp_flag] - rad[exp_flag]) * 1000.0 if exp_flag.any() else np.array([])
    named = [(i, exp_part[i]) for i in range(n) if exp_flag[i] and exp_part[i] in _RECORD_PARTS]
    correct = [closest_part[i] == p for i, p in named]

    # grasp stability: over contiguous expected-contact runs, std of closest distance
    stab = []
    i = 0
    while i < n:
        if exp_flag[i]:
            j = i
            while j + 1 < n and exp_flag[j + 1]:
                j += 1
            if j - i >= 2:
                stab.append(float(np.std(closest[i:j + 1]) * 1000.0))
            i = j + 1
        else:
            i += 1

    T = np.array([[float(r["tx"]), float(r["ty"]), float(r["tz"])] for r in rows[:n]])
    jerk = float((np.linalg.norm(T[3:] - 3 * T[2:-1] + 3 * T[1:-2] - T[:-3], axis=1) ** 2).mean()) \
        if len(T) >= 4 else None

    # Paper-definition temporal metrics (HOI-PAGE smoothness + CARI4D acceleration).
    # HOI-PAGE "Smoothness" = per-frame acceleration magnitude of the motion (lower=smoother),
    #   reported separately for Human and Object; we report the same quantity GT-free.
    # CARI4D "Acc-h"/"Acc-o" are ACCELERATION ERROR vs GT (cm); we can only give the GT-free
    #   acceleration MAGNITUDE (cm/frame^2) — a different quantity, flagged in the comparison.
    def _accel_mag(x):  # x [N, ..., 3] -> mean ||second-difference|| over points & frames
        if x is None or len(x) < 3:
            return None
        a = x[2:] - 2 * x[1:-1] + x[:-2]
        return float(np.linalg.norm(a, axis=-1).mean())

    def _jerk_mag(x):   # mean ||third-difference||^2
        if x is None or len(x) < 4:
            return None
        j = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]
        return float((np.linalg.norm(j, axis=-1) ** 2).mean())

    Jv = joints[:n] if joints is not None else None            # [N, J, 3] metres
    human_accel_m = _accel_mag(Jv)                             # mean per-joint accel (m/frame^2)
    human_smoothness = round(human_accel_m, 6) if human_accel_m is not None else None
    human_accel_cm = round(human_accel_m * 100.0, 4) if human_accel_m is not None else None
    hj = _jerk_mag(Jv)
    human_jerk = round(hj, 6) if hj is not None else None
    oa = _accel_mag(T)                                         # object translation accel (m/frame^2)
    object_smoothness = round(oa, 6) if oa is not None else None
    object_accel_cm = round(oa * 100.0, 4) if oa is not None else None

    # HOI-paper object smoothness transforms all eight local bounding-box corners,
    # so rotation and scale changes contribute instead of translation alone.
    quats = np.asarray([
        [float(r.get("qw", 1.0) or 1.0), float(r.get("qx", 0.0) or 0.0),
         float(r.get("qy", 0.0) or 0.0), float(r.get("qz", 0.0) or 0.0)]
        for r in rows[:n]
    ], dtype=np.float64)
    rotations = quat_to_rotmat(quats)
    scales = np.asarray([float(r.get("scale", 1.0) or 1.0) for r in rows[:n]], dtype=np.float64)
    if args.object_type == "sphere":
        unit_corners = box_corners(np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
        object_corners_world = unit_corners[None] * rad[:n, None, None] + T[:, None, :]
    elif args.object_type == "capsule":
        radius_m = float(args.object_radius_m if args.object_radius_m is not None else 0.018)
        half = np.full(3, radius_m, dtype=np.float64)
        half[{"x": 0, "y": 1, "z": 2}[args.object_axis_local]] += float(args.object_length_m) / 2.0
        local_corners = box_corners(np.stack([-half, half]))
        object_corners_world = np.einsum(
            "fij,fpj->fpi", rotations, local_corners[None] * scales[:, None, None]
        ) + T[:, None, :]
    else:
        import trimesh
        mesh = trimesh.load(args.object_mesh_path, force="mesh")
        local_corners = box_corners(np.asarray(mesh.bounds, dtype=np.float64))
        object_corners_world = np.einsum(
            "fij,fpj->fpi", rotations, local_corners[None] * scales[:, None, None]
        ) + T[:, None, :]

    paper_metrics = evaluate_hoi_paper_metrics(
        human_joints=Jv,
        object_corners_world=object_corners_world,
        human_object_signed_distances=paper_signed_distances,
        valid_frames=valid,
    )

    # Tier-3 novel metrics (docs/research_hoi_eval_metrics_sota.md)
    delta = 2  # ± frames around an audio event
    # audio-windowed contact ratio: fraction of audio events with touching nearby
    c_audio = None
    if event_frames:
        hits = [any(touching[max(0, e - delta):min(n, e + delta + 1)]) for e in event_frames]
        c_audio = round(float(np.mean(hits)), 3)
    # event-vs-flight acceleration split: a spike AT an event is physics, elsewhere artifact
    accel_event = accel_flight = None
    if len(T) >= 3:
        acc = np.linalg.norm(T[2:] - 2 * T[1:-1] + T[:-2], axis=1)  # index i ~ frame i+1
        near_event = np.zeros(len(acc), dtype=bool)
        for e in event_frames:
            near_event[max(0, e - 1 - delta):min(len(acc), e - 1 + delta + 1)] = True
        if event_frames and near_event.any():
            accel_event = round(float(acc[near_event].mean()), 5)
        if (~near_event).any():
            accel_flight = round(float(acc[~near_event].mean()), 5)
    # MDev* (GT-free ARCTIC MDev adaptation): during expected-contact runs, does the
    # closest part move WITH the object? part proxy = mean of the run's dominant part points
    mdev_vals = []
    i = 0
    while i < n:
        if exp_flag[i]:
            j = i
            while j + 1 < n and exp_flag[j + 1]:
                j += 1
            if j - i >= 2:
                names, counts = np.unique(closest_part[i:j + 1], return_counts=True)
                dom = str(names[np.argmax(counts)])
                p = parts[dom][i:j + 1].mean(dim=1).cpu().numpy()  # [run,3] part centroid
                dv = np.diff(p, axis=0) - np.diff(T[i:j + 1], axis=0)
                mdev_vals.append(float(np.linalg.norm(dv, axis=1).mean() * 1000.0))
            i = j + 1
        else:
            i += 1
    mdev_star = round(float(np.mean(mdev_vals)), 2) if mdev_vals else None

    nv = int(valid.sum())
    out = {
        "case": args.label or args.sample_dir.name,
        "trajectory": str(args.trajectory_csv),
        "human_params": src,
        "object_type": args.object_type,
        "n_frames": nv,
        # A. penetration
        "pen_frame_ratio": round(float(pen_frame.sum()) / max(nv, 1), 4),
        "non_collision_ratio": round(1.0 - float(pen_frame.sum()) / max(nv, 1), 4),
        "non_collision_vertex_ratio": non_collision_vertex_ratio,   # HOI-PAGE definition (per-vertex)
        "pen_depth_mean_mm": round(float(pen_depth.mean()), 2) if pen_depth.size else 0.0,
        "pen_depth_max_mm": round(float(pen_depth.max()), 2) if pen_depth.size else 0.0,
        # B. contact
        "contact_frame_ratio": round(float(touching.sum()) / max(nv, 1), 4),
        "expected_contact_frames": int(exp_flag.sum()),
        "contact_gap_mm": round(float(gaps.mean()), 2) if gaps.size else None,
        "part_named_frames": len(named),
        "part_correct_ratio": round(float(np.mean(correct)), 3) if correct else None,
        # C. temporal
        "object_jerk": round(jerk, 6) if jerk is not None else None,
        "grasp_stability_mm": round(float(np.mean(stab)), 2) if stab else None,
        # Paper-definition temporal (HOI-PAGE smoothness / CARI4D acceleration)
        "human_smoothness": human_smoothness,        # mean per-joint accel magnitude (m/frame^2)
        "object_smoothness": object_smoothness,      # object translation accel magnitude (m/frame^2)
        "human_jerk": human_jerk,                    # mean ||third-difference||^2 of joints
        "human_accel_cm": human_accel_cm,            # CARI4D Acc-h form (cm/frame^2), GT-free magnitude
        "object_accel_cm": object_accel_cm,          # CARI4D Acc-o form (cm/frame^2), GT-free magnitude
        # Generalized HOI-paper protocol (automatic, GT-free subset). Smoothness is
        # neighbour-average residual in metres; physical scores are in [0,1].
        "hoi_paper_human_temporal_smoothness_m": round(paper_metrics["human_temporal_smoothness"], 6)
        if paper_metrics["human_temporal_smoothness"] is not None else None,
        "hoi_paper_object_temporal_smoothness_m": round(paper_metrics["object_temporal_smoothness"], 6)
        if paper_metrics["object_temporal_smoothness"] is not None else None,
        "hoi_paper_non_collision_score": round(paper_metrics["non_collision_score"], 5)
        if paper_metrics["non_collision_score"] is not None else None,
        "hoi_paper_contact_score": round(paper_metrics["contact_score"], 5)
        if paper_metrics["contact_score"] is not None else None,
        "hoi_paper_query_vertex_scope": "all_smplx_vertices",
        "hoi_paper_contact_definition": "fraction_of_valid_frames_with_any_smplx_vertex_on_or_inside_object_surface",
        # Tier-3 novel (audio-aware) metrics
        "audio_events": len(event_frames),
        "contact_ratio_audio_windows": c_audio,
        "accel_at_events": accel_event,
        "accel_in_flight": accel_flight,
        "mdev_star_mm": mdev_star,
    }

    print(json.dumps(out, indent=2))
    out_json = args.out_json or (results_dir / "hoi_eval" / "hoi_interaction_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = args.out_csv.exists()
        with args.out_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out.keys()), extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(out)


if __name__ == "__main__":
    main()
