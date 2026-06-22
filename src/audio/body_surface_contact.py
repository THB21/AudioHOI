"""Body-surface contact points (hand/finger for a ball, foot surface for a kick).

For a ball, "which body part" (a wrist/ankle joint) is too coarse — the real contact is a
point on the hand or foot *surface*, often a specific finger. This computes, per contact
frame, the nearest body-surface vertex to the object and names the contact region from the
nearest hand/foot joint. It is the body-side analog of the mug handle contact point: the
object-side gives a point on the object, this gives the point on the human.

Output ``body_surface_contacts.csv``: frame, time, object_xyz, surface_xyz, dist_cm,
contact_part (left_hand/right_hand/left_foot/right_foot), contact_region (e.g. right_index,
right_palm, left_foot_toe), object_local-style for the body.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

# SMPL-X joint groups used to name the contact region
_FINGER_JOINTS = {  # name -> joint indices (proximal..distal) per hand
    "left_index": [25, 26, 27], "left_middle": [28, 29, 30], "left_pinky": [31, 32, 33],
    "left_ring": [34, 35, 36], "left_thumb": [37, 38, 39],
    "right_index": [40, 41, 42], "right_middle": [43, 44, 45], "right_pinky": [46, 47, 48],
    "right_ring": [49, 50, 51], "right_thumb": [52, 53, 54],
}
_WRIST = {"left_hand": 20, "right_hand": 21}
_FOOT = {"left_foot": (7, 10), "right_foot": (8, 11)}   # (ankle, foot/toe) joints


def _build_smplx(sample_dir: Path, body_model_root: Path):
    import torch
    import smplx

    gv = pickle.load(open(sample_dir / "results" / "gvhmr" / "result.pkl", "rb"))["smpl_params_incam"]
    nf = np.asarray(gv["transl"]).shape[0]
    betas = np.asarray(gv["betas"], np.float32)
    betas = np.repeat(betas[:1], nf, 0) if betas.shape[0] != nf else betas
    bm = smplx.create(str(body_model_root), model_type="smplx", gender="neutral", ext="npz",
                      use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=nf)
    kw = dict(global_orient=torch.from_numpy(np.asarray(gv["global_orient"], np.float32)),
              body_pose=torch.from_numpy(np.asarray(gv["body_pose"], np.float32)),
              betas=torch.from_numpy(betas),
              transl=torch.from_numpy(np.asarray(gv["transl"], np.float32)))
    st_path = sample_dir / "results" / "hands" / "stitched_smplx_params.pkl"
    if st_path.exists():
        st = pickle.load(open(st_path, "rb"))
        for k in ("left_hand_pose", "right_hand_pose"):
            if k in st:
                kw[k] = torch.from_numpy(np.asarray(st[k], np.float32))
    with torch.inference_mode():
        out = bm(return_verts=True, **kw)
    return out.vertices.detach().numpy(), out.joints.detach().numpy()


def _read_ball(path: Path) -> dict[int, np.ndarray]:
    out = {}
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["frame"])] = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
            except (KeyError, ValueError):
                continue
    return out


def run(sample_dir: Path, ball_csv: Path, body_model_root: Path, parts: list[str],
        contact_thresh_cm: float = 18.0, out_csv: Path | None = None):
    sample_dir = Path(sample_dir)
    verts, joints = _build_smplx(sample_dir, body_model_root)
    ball = _read_ball(ball_csv)
    nf = verts.shape[0]

    # candidate body vertices: pick vertices near the relevant joints so we don't scan all ~10k
    # (nearest-vertex to those joints' neighbourhood). Cheap proxy: use all verts, masked per frame.
    rows = []
    for fr, bxyz in sorted(ball.items()):
        i = fr - 1
        if i < 0 or i >= nf:
            continue
        V = verts[i]                       # [Nv,3]
        d = np.linalg.norm(V - bxyz[None], axis=1)
        vi = int(np.argmin(d))
        sp = V[vi]
        dist_cm = float(d[vi] * 100)
        # which part: nearest of the requested part anchors (wrist/foot) to the surface point
        anchors = {}
        for pt in parts:
            if pt in _WRIST:
                anchors[pt] = joints[i, _WRIST[pt]]
            elif pt in _FOOT:
                anchors[pt] = joints[i, _FOOT[pt][1]]
        if not anchors:
            continue
        part = min(anchors, key=lambda p: np.linalg.norm(anchors[p] - sp))
        # region: nearest finger joint (hands) or toe/heel (feet)
        if part in _WRIST:
            side = part.split("_")[0]
            fj = {n: joints[i, idx].mean(0) if isinstance(idx, list) else joints[i, idx]
                  for n, idx in _FINGER_JOINTS.items() if n.startswith(side)}
            # include palm (the wrist) as a fallback region
            fj[f"{side}_palm"] = joints[i, _WRIST[part]]
            region = min(fj, key=lambda n: np.linalg.norm(fj[n] - sp))
        else:
            side = part.split("_")[0]
            ankle, toe = _FOOT[part]
            region = f"{side}_foot_toe" if np.linalg.norm(joints[i, toe] - sp) <= \
                np.linalg.norm(joints[i, ankle] - sp) else f"{side}_foot_heel"
        rows.append({
            "frame": fr, "time": f"{(fr-1)/24.0:.6f}",
            "object_x": f"{bxyz[0]:.6f}", "object_y": f"{bxyz[1]:.6f}", "object_z": f"{bxyz[2]:.6f}",
            "surface_x": f"{sp[0]:.6f}", "surface_y": f"{sp[1]:.6f}", "surface_z": f"{sp[2]:.6f}",
            "dist_cm": f"{dist_cm:.1f}", "contact": int(dist_cm <= contact_thresh_cm),
            "contact_part": part, "contact_region": region, "vertex_index": vi,
        })

    out_csv = out_csv or (sample_dir / "results" / "audio_semantics" / "body_surface_contacts.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["frame", "time", "object_x", "object_y", "object_z", "surface_x", "surface_y",
            "surface_z", "dist_cm", "contact", "contact_part", "contact_region", "vertex_index"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    ct = [r for r in rows if r["contact"]]
    print(f"[{sample_dir.name}] {len(rows)} frames, {len(ct)} surface contacts (<= {contact_thresh_cm}cm) "
          f"→ {out_csv}")
    print(f"  contact regions: {dict(Counter(r['contact_region'] for r in ct))}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--ball-csv", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--parts", nargs="+", default=["left_hand", "right_hand"],
                    help="left_hand right_hand left_foot right_foot")
    ap.add_argument("--contact-thresh-cm", type=float, default=18.0)
    args = ap.parse_args()
    run(args.sample_dir, args.ball_csv, args.body_model_root, args.parts, args.contact_thresh_cm)


if __name__ == "__main__":
    main()
