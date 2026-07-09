"""Grasp-interval hand-object attachment (sustained contact).

A short touch only constrains one instant. A *held* object must stay rigidly attached to the
hand for the whole grasp interval — the hand stays at the object-local contact point (the mug
handle, a guitar neck position, ...) frame after frame, not just at the onset.

This implements the teammate's M15 idea generically:
  1. grasp intervals come from the persistence state machine in the contact records
     (contact_state ∈ {direct_contact, keep_grasp}, grouped by interval_id / stable_entity),
  2. a single STABLE object-local contact point is fixed per interval (the confirmed grasp
     point, e.g. handle_loop), and
  3. over the whole interval the object 6DOF pose is *translation-corrected* so its contact
     point tracks the hand — human motion is trusted, the object follows the grasp.

Result: a corrected object pose trajectory in which the contact point stays glued to the hand
throughout the hold. Outside grasp intervals the original pose is kept.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_STAGE1 = Path(__file__).resolve().parents[2] / "scripts" / "shared" / "radius_free_proxy" / "stage1_observation"
sys.path.insert(0, str(_STAGE1))
import fit_mug_articraft_keyframe_pose as _base  # noqa: E402

# SMPL-X wrist joint indices
_WRIST = {"left_hand": 20, "right_hand": 21}


def read_records(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def read_pose6d(path: Path) -> dict[int, np.ndarray]:
    out = {}
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            out[int(r["frame"])] = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"]),
                                             float(r["yaw"]), float(r["pitch"]), float(r["roll"]),
                                             float(r["scale"])])
    return out


def hand_world_positions(sample_dir: Path, body_model_root: Path, side: str, n: int) -> np.ndarray:
    """Per-frame world position of the grasping wrist from the (stitched) GVHMR body."""
    import pickle
    import torch
    import smplx

    d = pickle.load(open(sample_dir / "results" / "gvhmr" / "result.pkl", "rb"))
    p = d["smpl_params_incam"]
    nf = np.asarray(p["transl"]).shape[0]
    betas = np.asarray(p["betas"], dtype=np.float32)
    if betas.shape[0] != nf:
        betas = np.repeat(betas[:1], nf, axis=0)
    bm = smplx.create(str(body_model_root), model_type="smplx", gender="neutral", ext="npz",
                      use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=nf)
    with torch.inference_mode():
        out = bm(global_orient=torch.from_numpy(np.asarray(p["global_orient"], np.float32)),
                 body_pose=torch.from_numpy(np.asarray(p["body_pose"], np.float32)),
                 betas=torch.from_numpy(betas),
                 transl=torch.from_numpy(np.asarray(p["transl"], np.float32)), return_verts=False)
    return out.joints.detach().numpy()[:, _WRIST.get(side, 20), :]


def grasp_intervals(records: list[dict]) -> list[dict]:
    """Group consecutive held frames into intervals with a stable object-local grasp point."""
    held = [r for r in records if r.get("contact_state") in ("direct_contact", "keep_grasp")]
    held.sort(key=lambda r: int(r["frame"]))
    intervals, cur = [], []
    for r in held:
        if cur and int(r["frame"]) - int(cur[-1]["frame"]) > 30:
            intervals.append(cur); cur = []
        cur.append(r)
    if cur:
        intervals.append(cur)
    out = []
    for iv in intervals:
        # stable grasp point = the confirmed part-contact point in the interval (last wins)
        confirmed = [r for r in iv if r.get("object_local_x") and r.get("contact_target") == "part"
                     or (r.get("semantic_contact_part") in ("handle_loop", "handle"))]
        pick = confirmed[-1] if confirmed else iv[-1]
        if not pick.get("object_local_x"):
            continue
        side = pick.get("target_entity") or pick.get("vlm_hand_side") or "left_hand"
        if side not in _WRIST:
            side = "left_hand"
        out.append({
            "start": int(iv[0]["frame"]), "end": int(iv[-1]["frame"]),
            "grasp_local": np.array([float(pick["object_local_x"]), float(pick["object_local_y"]),
                                     float(pick["object_local_z"])]),
            "part": pick.get("semantic_contact_part", ""), "side": side,
        })
    return out


def attach(sample_dir: Path, records_csv: Path, pose6d_csv: Path, body_model_root: Path,
           out_csv: Path | None = None, w_attach: float = 1.0):
    records = read_records(records_csv)
    pose = read_pose6d(pose6d_csv)
    n = max(pose) if pose else 0
    intervals = grasp_intervals(records)
    if not intervals:
        print("no grasp intervals found"); return

    side = intervals[0]["side"]
    hands = hand_world_positions(Path(sample_dir), Path(body_model_root), side, n + 1)

    corrected = {fr: p.copy() for fr, p in pose.items()}
    report = []
    for iv in intervals:
        gl = iv["grasp_local"]
        for fr in range(iv["start"], iv["end"] + 1):
            if fr not in pose:
                continue
            params = pose[fr]
            gw = _base.transform(params, gl[None])[0]       # current contact point in world
            hw = hands[min(fr - 1, len(hands) - 1)]          # hand in world
            before = np.linalg.norm(gw - hw)
            # translation correction so the contact point lands on the hand (object follows grasp)
            new = params.copy()
            new[:3] += w_attach * (hw - gw)
            corrected[fr] = new
            gw2 = _base.transform(new, gl[None])[0]
            report.append((fr, iv["part"], before, np.linalg.norm(gw2 - hw)))

    out_csv = out_csv or pose6d_csv.with_name(pose6d_csv.stem + "_grasp_attached.csv")
    with Path(out_csv).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time", "tx", "ty", "tz", "yaw", "pitch", "roll", "scale", "radius_m"])
        for fr in sorted(corrected):
            p = corrected[fr]
            w.writerow([fr, f"{(fr-1)/24.0:.6f}", *[f"{v:.6f}" for v in p], 0.05])

    bef = np.mean([r[2] for r in report]) * 100
    aft = np.mean([r[3] for r in report]) * 100
    print(f"grasp intervals: {[(iv['start'], iv['end'], iv['part'], iv['side']) for iv in intervals]}")
    print(f"hand<->contact-point dist over holds: {bef:.1f} cm  →  {aft:.1f} cm after attachment")
    print(f"corrected 6DOF pose (object stays glued to the hand): {out_csv}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--records-csv", type=Path, required=True)
    ap.add_argument("--pose6d-csv", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()
    attach(args.sample_dir, args.records_csv, args.pose6d_csv, args.body_model_root, args.out_csv)


if __name__ == "__main__":
    main()
