#!/usr/bin/env python3
"""Merge HaMeR hand pose into GVHMR's SMPL-X body params."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


SMPLX_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
    12, 13, 14, 16, 17, 18, 19,
]
LEFT_WRIST_JOINT = 20
RIGHT_WRIST_JOINT = 21
LEFT_WRIST_BP_IDX = LEFT_WRIST_JOINT - 1
RIGHT_WRIST_BP_IDX = RIGHT_WRIST_JOINT - 1

_M_FLIP_X = np.diag([-1.0, 1.0, 1.0])


def rotmats_to_aa(rotmats):
    N, J = rotmats.shape[:2]
    aa = np.zeros((N, J, 3), dtype=np.float32)
    for i in range(N):
        for j in range(J):
            rvec, _ = cv2.Rodrigues(rotmats[i, j].astype(np.float64))
            aa[i, j] = rvec.ravel().astype(np.float32)
    return aa.reshape(N, J * 3)


def aa_to_R(aa):
    flat = aa.reshape(-1, 3).astype(np.float64)
    R = np.empty((flat.shape[0], 3, 3), dtype=np.float64)
    for i in range(flat.shape[0]):
        R[i], _ = cv2.Rodrigues(flat[i])
    return R.reshape(*aa.shape[:-1], 3, 3)


def R_to_aa(R):
    flat = R.reshape(-1, 3, 3).astype(np.float64)
    aa = np.empty((flat.shape[0], 3), dtype=np.float64)
    for i in range(flat.shape[0]):
        rvec, _ = cv2.Rodrigues(flat[i])
        aa[i] = rvec.ravel()
    return aa.reshape(*R.shape[:-2], 3)


def mirror_left_hand_aa(pose_aa):
    # HaMeR's left-hand input is horizontally flipped; undo the mirror so the
    # pose can be used as a SMPL-X left_hand_pose.
    aa = pose_aa.reshape(-1, 15, 3).copy()
    aa[:, :, 1] *= -1
    aa[:, :, 2] *= -1
    return aa.reshape(-1, 45)


def compute_global_rotations(global_orient_aa, body_pose_aa):
    N = global_orient_aa.shape[0]
    R_local = np.zeros((N, 22, 3, 3), dtype=np.float64)
    R_local[:, 0] = aa_to_R(global_orient_aa)
    R_local[:, 1:] = aa_to_R(body_pose_aa.reshape(N, 21, 3)).reshape(N, 21, 3, 3)
    R_global = np.zeros_like(R_local)
    R_global[:, 0] = R_local[:, 0]
    for j in range(1, 22):
        R_global[:, j] = R_global[:, SMPLX_PARENTS[j]] @ R_local[:, j]
    return R_global


def override_wrist_orientations(
    body_pose_aa,
    global_orient_aa,
    hamer_left_R,
    hamer_right_R,
    left_detected,
    right_detected,
):
    body_pose_3 = body_pose_aa.copy().reshape(-1, 21, 3)
    R_global = compute_global_rotations(global_orient_aa, body_pose_3)
    R_elbow_left = R_global[:, SMPLX_PARENTS[LEFT_WRIST_JOINT]]
    R_elbow_right = R_global[:, SMPLX_PARENTS[RIGHT_WRIST_JOINT]]

    for i in range(body_pose_3.shape[0]):
        if right_detected[i]:
            R_local = R_elbow_right[i].T @ hamer_right_R[i]
            body_pose_3[i, RIGHT_WRIST_BP_IDX] = R_to_aa(R_local[None])[0]
        if left_detected[i]:
            R_target = _M_FLIP_X @ hamer_left_R[i] @ _M_FLIP_X
            R_local = R_elbow_left[i].T @ R_target
            body_pose_3[i, LEFT_WRIST_BP_IDX] = R_to_aa(R_local[None])[0]

    return body_pose_3.reshape(-1, 63).astype(np.float32)


def select_anchor(pose_aa, detected, mode):
    if not detected.any():
        return np.zeros(pose_aa.shape[1], dtype=np.float32)
    det = pose_aa[detected]
    return det[0].copy() if mode == "first" else np.median(det, axis=0).astype(np.float32)


def fill_gaps(pose_aa, detected, anchor, mode):
    if not detected.any():
        out = np.zeros_like(pose_aa)
        out[:] = anchor
        return out

    N = pose_aa.shape[0]
    det_idx = np.where(detected)[0]
    all_f = np.arange(N, dtype=np.float64)
    det_f = det_idx.astype(np.float64)

    result = np.zeros_like(pose_aa)
    for d in range(pose_aa.shape[1]):
        result[:, d] = np.interp(all_f, det_f, pose_aa[detected, d])

    if mode != "none":
        if det_idx[0] > 0:
            result[:det_idx[0]] = anchor
        if det_idx[-1] < N - 1:
            result[det_idx[-1] + 1:] = anchor
    return result


def smooth_pose(pose_aa, detected, sigma):
    if sigma <= 0 or detected.sum() < 3:
        return pose_aa
    smoothed = gaussian_filter1d(pose_aa, sigma=sigma, axis=0)
    out = pose_aa.copy()
    out[detected] = smoothed[detected]
    return out


def read_gvhmr(path):
    with path.open("rb") as f:
        data = pickle.load(f)
    p = data["smpl_params_incam"]
    return {
        "body_pose": np.asarray(p["body_pose"], dtype=np.float32),
        "betas": np.asarray(p["betas"], dtype=np.float32),
        "global_orient": np.asarray(p["global_orient"], dtype=np.float32),
        "transl": np.asarray(p["transl"], dtype=np.float32),
        "K_fullimg": np.asarray(data["K_fullimg"], dtype=np.float32),
    }


def read_hamer(path):
    with path.open("rb") as f:
        return pickle.load(f)


def write_pkl(path, data):
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=Path("samples/basketball_01"))
    parser.add_argument("--anchor-mode", choices=("first", "median", "none"), default="first")
    parser.add_argument("--smooth-pose-sigma", type=float, default=2.0)
    parser.add_argument("--use-hamer-wrist", action="store_true", default=True)
    parser.add_argument("--no-hamer-wrist", dest="use_hamer_wrist", action="store_false")
    parser.add_argument("--smooth-wrist-sigma", type=float, default=2.0)
    args = parser.parse_args()

    results_dir = args.sample_dir / "results"
    gvhmr_pkl = results_dir / "gvhmr" / "result.pkl"
    hamer_pkl = results_dir / "hands" / "hand_mano_params.pkl"
    out_pkl = results_dir / "hands" / "stitched_smplx_params.pkl"

    print(f"Loading GVHMR: {gvhmr_pkl}")
    gvhmr = read_gvhmr(gvhmr_pkl)
    n_gvhmr = gvhmr["transl"].shape[0]

    print(f"Loading HaMeR: {hamer_pkl}")
    hamer = read_hamer(hamer_pkl)
    frame_to_hamer = {int(f): i for i, f in enumerate(hamer["frames"])}

    def _aligned_hand(side):
        h = hamer[side]
        rotmats_hamer = h["hand_pose"]
        detected_hamer = h["detected"]

        rotmats_aligned = np.zeros((n_gvhmr, 15, 3, 3), dtype=np.float64)
        for j in range(3):
            rotmats_aligned[:, :, j, j] = 1.0
        detected_aligned = np.zeros(n_gvhmr, dtype=bool)

        for gi in range(n_gvhmr):
            hi = frame_to_hamer.get(gi + 1)
            if hi is not None and detected_hamer[hi]:
                rotmats_aligned[gi] = rotmats_hamer[hi]
                detected_aligned[gi] = True

        print(f"  {side}: {detected_aligned.sum()}/{n_gvhmr} frames detected "
              f"({detected_aligned.mean():.1%})")

        pose_aa = np.zeros((n_gvhmr, 45), dtype=np.float32)
        if detected_aligned.any():
            pose_aa[detected_aligned] = rotmats_to_aa(rotmats_aligned[detected_aligned])

        if side == "left":
            pose_aa = mirror_left_hand_aa(pose_aa)

        anchor = select_anchor(pose_aa, detected_aligned, args.anchor_mode)
        pose_aa = fill_gaps(pose_aa, detected_aligned, anchor, args.anchor_mode)
        pose_aa = smooth_pose(pose_aa, detected_aligned, args.smooth_pose_sigma)
        return pose_aa, detected_aligned

    print("Converting hand pose (rot mat -> axis-angle)...")
    left_pose_aa, left_detected = _aligned_hand("left")
    right_pose_aa, right_detected = _aligned_hand("right")

    betas = gvhmr["betas"]
    if betas.shape[0] == 1 and n_gvhmr > 1:
        betas = np.repeat(betas, n_gvhmr, axis=0)

    K = gvhmr["K_fullimg"]
    if K.ndim == 2:
        K = K[np.newaxis].repeat(n_gvhmr, axis=0)
    elif K.shape[0] == 1 and n_gvhmr > 1:
        K = np.repeat(K, n_gvhmr, axis=0)

    body_pose_out = gvhmr["body_pose"].copy()

    if args.use_hamer_wrist:
        def _aligned_global_orient(side):
            h = hamer[side]
            R_hamer = h["global_orient"]
            det_hamer = h["detected"]
            R_out = np.tile(np.eye(3), (n_gvhmr, 1, 1)).astype(np.float64)
            det_out = np.zeros(n_gvhmr, dtype=bool)
            for gi in range(n_gvhmr):
                hi = frame_to_hamer.get(gi + 1)
                if hi is not None and det_hamer[hi]:
                    R_out[gi] = R_hamer[hi]
                    det_out[gi] = True
            return R_out, det_out

        hamer_left_R, left_det_w = _aligned_global_orient("left")
        hamer_right_R, right_det_w = _aligned_global_orient("right")

        body_pose_out = override_wrist_orientations(
            body_pose_aa=body_pose_out,
            global_orient_aa=gvhmr["global_orient"],
            hamer_left_R=hamer_left_R,
            hamer_right_R=hamer_right_R,
            left_detected=left_det_w,
            right_detected=right_det_w,
        )

        if args.smooth_wrist_sigma > 0:
            bp3 = body_pose_out.reshape(n_gvhmr, 21, 3)
            for bp_idx, det in (
                (LEFT_WRIST_BP_IDX, left_det_w),
                (RIGHT_WRIST_BP_IDX, right_det_w),
            ):
                if det.sum() >= 3:
                    smoothed = gaussian_filter1d(
                        bp3[:, bp_idx], sigma=args.smooth_wrist_sigma, axis=0
                    )
                    bp3[det, bp_idx] = smoothed[det]
            body_pose_out = bp3.reshape(n_gvhmr, 63).astype(np.float32)

        print(f"  wrist override: left {int(left_det_w.sum())}/{n_gvhmr}, "
              f"right {int(right_det_w.sum())}/{n_gvhmr}")

    stitched = {
        "body_pose": body_pose_out,
        "betas": betas,
        "global_orient": gvhmr["global_orient"],
        "transl": gvhmr["transl"],
        "left_hand_pose": left_pose_aa,
        "right_hand_pose": right_pose_aa,
        "left_detected": left_detected,
        "right_detected": right_detected,
        "K_fullimg": K,
        "n_frames": n_gvhmr,
        "anchor_mode": args.anchor_mode,
        "smooth_pose_sigma": args.smooth_pose_sigma,
    }

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    write_pkl(out_pkl, stitched)

    print(f"stitched_pkl: {out_pkl}")
    print(f"n_frames: {n_gvhmr}")
    print(f"left/right detected: {int(left_detected.sum())} / {int(right_detected.sum())}")


if __name__ == "__main__":
    main()
