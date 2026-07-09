#!/usr/bin/env python3
"""Body-side contact refinement — the object solver's counterpart.

The depth solver keeps the OBJECT out of the human (and pins it to the part at
contact), but at anchor frames the object is fixed, so a hand/foot that GVHMR
placed inside the object stays inside it. This stage nudges the BODY: for every
frame where a part center sits inside the object, and for contact frames where
the active part should visibly rest on the surface (hand on top of a dribbled
ball), it optimizes only that part's kinematic chain (collar/shoulder/elbow/
wrist for a hand, hip/knee/ankle for a foot) so the part lands on the object
surface. A stay-close pose prior and temporal smoothness keep it a slight,
pop-free correction; everything is derived from the trajectory CSV's per-frame
active part, so it works for any object/part combination.

Output: results/contact_refine/contact_refined_smplx_params.pkl (same schema as
the stitched hand params — renderers prefer it when present) + a report CSV.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_part_utils import LEFT_FOOT_ID, RIGHT_FOOT_ID  # noqa: E402
from object_geometry import geometry_from_rows  # noqa: E402

# Point cloud per part: EVERY point must clear the object (fingers must not dip
# inside even when the palm is out); the CLOSEST point carries the touch. Extra
# SMPL-X landmark joints: 60-65 = L/R big toe, small toe, heel; 66-75 = L/R
# thumb/index/middle/ring/pinky fingertips.
_PART_POINTS = {
    "left_hand": [20, *range(25, 40), 66, 67, 68, 69, 70],   # wrist, finger joints, tips
    "right_hand": [21, *range(40, 55), 71, 72, 73, 74, 75],
    "left_foot": [7, LEFT_FOOT_ID, 60, 61, 62],              # ankle, foot, toes, heel
    "right_foot": [8, RIGHT_FOOT_ID, 63, 64, 65],
}
# body_pose slot = SMPL-X joint id - 1 (pelvis is global_orient)
_CHAIN_SLOTS = {
    "left_hand": [12, 15, 17, 19],   # L_collar, L_shoulder, L_elbow, L_wrist
    "right_hand": [13, 16, 18, 20],  # R_collar, R_shoulder, R_elbow, R_wrist
    "left_foot": [0, 3, 6],          # L_hip, L_knee, L_ankle
    "right_foot": [1, 4, 7],         # R_hip, R_knee, R_ankle
}
_PARTS = list(_CHAIN_SLOTS)


def part_point_dists(points_source: torch.Tensor, geom,
                     part_ids: dict = None) -> dict[str, torch.Tensor]:
    """Per-part distances [N, P] from each part point to the object surface
    (sphere center / capsule axis / mesh SDF — see object_geometry).

    points_source is either SMPL-X `.joints` (with skeleton `part_ids`, default) or
    `.vertices` (with vertex `part_ids` from part_vertices.part_vertex_indices)."""
    ids_map = part_ids or _PART_POINTS
    return {
        name: geom.point_distances(points_source[:, ids, :])
        for name, ids in ids_map.items()
    }


def load_params(results_dir: Path) -> tuple[dict, bool]:
    stitched_pkl = results_dir / "hands" / "stitched_smplx_params.pkl"
    raw = pickle.load((results_dir / "gvhmr" / "result.pkl").open("rb"))
    p = raw["smpl_params_incam"]
    params = {
        "body_pose": np.asarray(p["body_pose"], dtype=np.float32),
        "betas": np.asarray(p["betas"], dtype=np.float32),
        "global_orient": np.asarray(p["global_orient"], dtype=np.float32),
        "transl": np.asarray(p["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(raw["K_fullimg"], dtype=np.float32),
    }
    has_hands = False
    if stitched_pkl.exists():
        stitched = pickle.load(stitched_pkl.open("rb"))
        params.update({k: np.asarray(v) for k, v in stitched.items() if isinstance(v, np.ndarray)})
        has_hands = "left_hand_pose" in params
    return params, has_hands


def read_trajectory(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No trajectory rows in {path}")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optimize arm/leg pose so contact parts rest on the object surface.")
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    parser.add_argument("--trajectory-csv", type=Path, default=None,
                        help="final object trajectory (default: pose6d_sharedcam_contactphase_depthv3)")
    parser.add_argument("--object-type", choices=["sphere", "capsule", "mesh_sdf"], default="sphere",
                        help="capsule = line object (stick); mesh_sdf = differentiable voxel SDF "
                             "of a baked URDF mesh (chair/mug) — EasyHOI-style surface hinge")
    parser.add_argument("--object-length-m", type=float, default=None,
                        help="capsule axis length (stick: 1.86 from case config)")
    parser.add_argument("--object-radius-m", type=float, default=None,
                        help="surface radius override (capsule default 0.018 = stick URDF)")
    parser.add_argument("--object-axis-local", choices=["x", "y", "z"], default="x",
                        help="object-local long axis (stick: x, verified by reprojection)")
    parser.add_argument("--object-mesh-path", type=str, default=None,
                        help="URDF-frame GLB for --object-type mesh_sdf (bake_urdf_to_glb --keep-origin)")
    parser.add_argument("--use-part-verts", action="store_true",
                        help="clear penetration on dense SMPL-X part VERTICES (hand/foot flesh, "
                             "not just 21 skeleton points) — needed for true zero-penetration")
    parser.add_argument("--body-parts", type=str, default=None,
                        help="comma list to also refine (pelvis,back) for sit/lean contact; "
                             "implies --use-part-verts")
    parser.add_argument("--pen-margin-m", type=float, default=0.0)
    parser.add_argument("--touch-offset-m", type=float, default=0.002,
                        help="how far outside the surface the part's closest point should rest at contact")
    parser.add_argument("--point-margin-m", type=float, default=0.0,
                        help="finger/skin thickness: part points are skeleton joints, the flesh needs this much extra clearance")
    parser.add_argument("--pull-range-m", type=float, default=0.04,
                        help="only pull a part to the surface when it is already this close (a large gap "
                             "means tracking and body pose genuinely disagree — forcing contact fakes it)")
    parser.add_argument("--halo-frames", type=int, default=2, help="neighbor frames co-optimized for smooth blending")
    parser.add_argument("--w-pen", type=float, default=500.0)
    parser.add_argument("--w-touch", type=float, default=50.0)
    parser.add_argument("--w-prior", type=float, default=1.0)
    parser.add_argument("--w-temp", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args(argv)

    results_dir = args.sample_dir / "results"
    traj_csv = args.trajectory_csv or (
        results_dir / "pose6d_sharedcam_contactphase_depthv3" / "ball_pose6d_sharedcam_contactphase_trajectory.csv")
    out_dir = results_dir / "contact_refine"
    out_dir.mkdir(parents=True, exist_ok=True)

    params, has_hands = load_params(results_dir)
    n = params["body_pose"].shape[0]
    rows = read_trajectory(traj_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    geom = geometry_from_rows(rows, n, object_type=args.object_type,
                              length_m=args.object_length_m, radius_m=args.object_radius_m,
                              axis_local=args.object_axis_local,
                              mesh_path=args.object_mesh_path, device=device)
    radius = geom.radius
    contact_flag = np.zeros(n, dtype=bool)
    active_part = np.array(["" for _ in range(n)], dtype=object)
    for r in rows:
        i = int(r["frame"]) - 1
        if not (0 <= i < n):
            continue
        # touch holds through whole contact-state intervals (dribble guidance),
        # not just the event frames — else the hand hovers between events
        contact_flag[i] = (int(r.get("contact_frame", 0) or 0) == 1
                           or int(r.get("audio_anchor", 0) or 0) == 1
                           or int(r.get("audio_contact_frame", 0) or 0) == 1
                           or int(r.get("human_contact_state", 0) or 0) == 1)
        part = str(r.get("active_part", "") or "")
        if not part and r.get("depth_anchor_sides"):
            # generic-mainline line objects: 'left+right' palm anchors → hand parts
            part = "+".join(f"{s}_hand" for s in str(r["depth_anchor_sides"]).split("+")
                            if s in ("left", "right"))
        active_part[i] = part
    valid = geom.valid

    # part set: skeleton points (default) or dense mesh vertices (--use-part-verts)
    body_parts = [p for p in (args.body_parts.split(",") if args.body_parts else []) if p]
    use_verts = args.use_part_verts or bool(body_parts)
    parts_list = list(_PARTS)          # hands + feet always
    chain_slots = dict(_CHAIN_SLOTS)
    # body_pose slot = SMPL-X joint id - 1; free spine+hips for sit/lean body contact
    _BODY_CHAINS = {"pelvis": [0, 1, 2, 5], "back": [2, 5, 8, 11]}
    for bp in body_parts:
        if bp in _BODY_CHAINS:
            parts_list.append(bp)
            chain_slots[bp] = _BODY_CHAINS[bp]
    if use_verts:
        from part_vertices import part_vertex_indices
        pv = part_vertex_indices(args.body_model_root, parts_list)
        part_ids = {k: torch.as_tensor(v, device=device) for k, v in pv.items()}
    else:
        part_ids = {k: torch.as_tensor(_PART_POINTS[k], device=device) for k in parts_list}

    import smplx
    betas = params["betas"]
    if betas.shape[0] == 1:
        betas = np.repeat(betas, n, axis=0)
    model = smplx.create(str(args.body_model_root), model_type="smplx", gender="neutral", ext="npz",
                         use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=n).to(device)
    body0 = torch.from_numpy(params["body_pose"]).to(device)
    fixed = dict(
        betas=torch.from_numpy(betas).to(device),
        global_orient=torch.from_numpy(params["global_orient"]).to(device),
        transl=torch.from_numpy(params["transl"]).to(device),
        return_verts=use_verts,
    )
    if has_hands:
        fixed["left_hand_pose"] = torch.from_numpy(params["left_hand_pose"].astype(np.float32)).to(device)
        fixed["right_hand_pose"] = torch.from_numpy(params["right_hand_pose"].astype(np.float32)).to(device)

    def forward(body_pose: torch.Tensor) -> dict[str, torch.Tensor]:
        out = model(body_pose=body_pose, **fixed)
        src = out.vertices if use_verts else out.joints
        return part_point_dists(src, geom, part_ids)

    with torch.no_grad():
        dists0 = forward(body0)

    clearance = radius + args.pen_margin_m + args.point_margin_m
    touch_target = radius + args.touch_offset_m

    # collect (part, frame) work items: interpenetrations anywhere + surface-touch at contacts
    pen_pairs, touch_pairs = [], []
    for name in parts_list:
        dmin0 = dists0[name].min(dim=1).values
        for i in range(n):
            if not bool(valid[i]):
                continue
            if float(dmin0[i]) < float(clearance[i]):
                pen_pairs.append((name, i))
            if (contact_flag[i] and name in str(active_part[i]).split("+")
                    and float(dmin0[i]) < float(radius[i] + args.pull_range_m)):
                touch_pairs.append((name, i))
    if not pen_pairs and not touch_pairs:
        print("No interpenetration and no reachable contacts — nothing to refine.")
        return

    # optimization mask: each work item frees its part's chain slots on frame±halo
    mask = torch.zeros(n, 63, device=device)
    frame_mask = torch.zeros(n, dtype=torch.bool, device=device)
    for name, i in pen_pairs + touch_pairs:
        for j in range(max(0, i - args.halo_frames), min(n, i + args.halo_frames + 1)):
            frame_mask[j] = True
            for s in chain_slots[name]:
                mask[j, 3 * s: 3 * s + 3] = 1.0

    delta = torch.zeros(n, 63, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=args.lr)
    touch_idx = {name: torch.tensor([i for p, i in touch_pairs if p == name], dtype=torch.long, device=device)
                 for name in parts_list}
    # the hinge must be LIVE on every valid frame (not just initially-penetrating
    # ones): the touch pull and temporal coupling otherwise drag clean frames into
    # the object with nothing to stop them
    hinge_idx = torch.nonzero(valid, as_tuple=False).flatten()

    for _ in range(args.steps):
        opt.zero_grad()
        d_eff = delta * mask
        dists = forward(body0 + d_eff)
        loss = args.w_prior * (d_eff ** 2).sum() + args.w_temp * ((d_eff[1:] - d_eff[:-1]) ** 2).sum()
        for name in parts_list:
            # every point of the part must clear; 1 mm overshoot because the
            # prior/hinge equilibrium otherwise parks a hair inside
            loss = loss + args.w_pen * (torch.relu(clearance[hinge_idx, None] + 1e-3 - dists[name][hinge_idx]) ** 2).sum()
            if len(touch_idx[name]):
                idx = touch_idx[name]
                dmin = dists[name][idx].min(dim=1).values
                loss = loss + args.w_touch * ((dmin - touch_target[idx]) ** 2).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        d_eff = (delta * mask).detach()
        dists1 = forward(body0 + d_eff)
        body1 = (body0 + d_eff).cpu().numpy().astype(np.float32)

    report = []
    for name, i in sorted(set(pen_pairs + touch_pairs), key=lambda t: (t[1], t[0])):
        d0 = float(dists0[name][i].min())
        d1 = float(dists1[name][i].min())
        report.append({
            "frame": i + 1, "part": name,
            "kind": "touch" if (name, i) in set(touch_pairs) else "penetration",
            "dist_before_m": f"{d0:.4f}", "dist_after_m": f"{d1:.4f}",
            "radius_m": f"{float(radius[i]):.4f}",
            "pose_delta_deg": f"{float(torch.rad2deg(d_eff[i].abs().max())):.2f}",
        })

    out_params = {k: v for k, v in params.items()}
    out_params["body_pose"] = body1
    out_params["source_trajectory"] = str(traj_csv)
    out_pkl = out_dir / "contact_refined_smplx_params.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(out_params, f)
    report_csv = out_dir / "contact_refine_report.csv"
    with report_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    n_pen_fixed = sum(1 for r in report if r["kind"] == "penetration" and float(r["dist_after_m"]) >= float(r["radius_m"]))
    n_pen = sum(1 for r in report if r["kind"] == "penetration")
    print(f"refined {len(set(i for _, i in pen_pairs + touch_pairs))} frames "
          f"({n_pen} penetrations, {n_pen_fixed} cleared; {len(touch_pairs)} touch targets)")
    print(f"refined_params: {out_pkl}")
    print(f"report: {report_csv}")


if __name__ == "__main__":
    main()
