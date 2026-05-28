#!/usr/bin/env python3
"""Render contact-candidate overlays back onto the original video frames."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import smplx

LEFT_PALM_IDS = [20, 25, 28, 31, 34]
RIGHT_PALM_IDS = [21, 40, 43, 46, 49]
BALL_BGR = (32, 122, 219)
LEFT_PALM_BGR = (70, 180, 70)
RIGHT_PALM_BGR = (70, 200, 230)
HAND_ON_BGR = (72, 196, 92)
FLOOR_ON_BGR = (40, 190, 250)
CENTER_BGR = (50, 50, 240)
TEXT_BGR = (235, 235, 235)
SHADOW_BGR = (15, 15, 15)
FLOOR_LINE_BGR = (70, 220, 220)
BODY_LINE_BGR = (104, 78, 214)
BODY_POINT_BGR = (194, 228, 244)
BODY_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    (12, 15),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


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
    model = smplx.create(
        str(body_models_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=human_params["transl"].shape[0],
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(human_params["body_pose"]),
            betas=torch.from_numpy(human_params["betas"]),
            global_orient=torch.from_numpy(human_params["global_orient"]),
            transl=torch.from_numpy(human_params["transl"]),
            return_verts=False,
        )
    return output.joints.detach().cpu().numpy().astype(np.float64)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = K[:, 0, 0] * (points_cam[:, 0] / z) + K[:, 0, 2]
    v = K[:, 1, 1] * (points_cam[:, 1] / z) + K[:, 1, 2]
    valid = points_cam[:, 2] > 1e-6
    return np.stack([u, v], axis=1), valid


def build_palm_centers(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = joints[:, LEFT_PALM_IDS, :].mean(axis=1)
    right = joints[:, RIGHT_PALM_IDS, :].mean(axis=1)
    return left, right


def draw_text(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.7, thickness: int = 2) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, SHADOW_BGR, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_body_skeleton(frame: np.ndarray, joints_cam: np.ndarray, K: np.ndarray) -> None:
    uv, valid = project_points(joints_cam[:22], K[None])[0], None
    valid = joints_cam[:22, 2] > 1e-6
    for a, b in BODY_EDGES:
        if not (valid[a] and valid[b]):
            continue
        pa = tuple(np.round(uv[a]).astype(int))
        pb = tuple(np.round(uv[b]).astype(int))
        cv2.line(frame, pa, pb, BODY_LINE_BGR, 2, cv2.LINE_AA)
    for p in uv[valid]:
        cv2.circle(frame, tuple(np.round(p).astype(int)), 2, BODY_POINT_BGR, -1, cv2.LINE_AA)


def render_video(
    frames_dir: Path,
    out_path: Path,
    preview_path: Path,
    fps: float,
    ball_rows: list[dict[str, str]],
    state_rows: list[dict[str, str]],
    center_rows: dict[int, list[dict[str, str]]],
    joints: np.ndarray,
    K: np.ndarray,
    mode: str,
) -> None:
    first = cv2.imread(str(frames_dir / '00001.png'))
    if first is None:
        raise RuntimeError(f"Could not read frames from {frames_dir}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    preview_written = False
    for i, (ball, state) in enumerate(zip(ball_rows, state_rows)):
        frame_id = int(ball['frame'])
        frame = cv2.imread(str(frames_dir / f"{frame_id:05d}.png"))
        if frame is None:
            continue

        draw_body_skeleton(frame, joints[i], K[i])

        left_palm, right_palm = build_palm_centers(joints[i:i+1])
        palms = np.concatenate([left_palm, right_palm], axis=0)
        palms_uv, palms_valid = project_points(palms, np.repeat(K[i:i+1], 2, axis=0))
        palms_uv = palms_uv.reshape(2, 2)
        palms_valid = palms_valid.reshape(2)
        for palm_idx, (uv, ok, color) in enumerate(zip(palms_uv, palms_valid, [LEFT_PALM_BGR, RIGHT_PALM_BGR])):
            if ok:
                pt = tuple(np.round(uv).astype(int))
                cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 8, SHADOW_BGR, 1, cv2.LINE_AA)

        ball_pt = (int(round(float(ball['ball_center_x']))), int(round(float(ball['ball_center_y']))))
        ball_r = max(4, int(round(float(ball['radius']))))
        cv2.circle(frame, ball_pt, ball_r, BALL_BGR, 2, cv2.LINE_AA)
        cv2.circle(frame, ball_pt, 3, BALL_BGR, -1, cv2.LINE_AA)

        floor_y = int(round(float(ball['floor_v']))) if 'floor_v' in ball else None
        if floor_y is None:
            floor_y = int(round(float(ball.get('floor_v', state.get('floor_v', 0))))) if 'floor_v' in state else None
        if floor_y is None and 'floor_v' in ball_rows[0]:
            floor_y = int(round(float(ball['floor_v'])))
        if floor_y is not None and floor_y > 0:
            cv2.line(frame, (0, floor_y), (w - 1, floor_y), FLOOR_LINE_BGR, 2, cv2.LINE_AA)

        hand_on = int(state['hand_contact_state']) == 1
        floor_on = int(state['floor_contact_state']) == 1
        hand_score = float(state['hand_score'])
        floor_score = float(state['floor_score'])
        active_hand = state['active_hand']

        draw_text(frame, f"frame {frame_id}  t={float(ball['time']):.2f}s", (24, h - 26), TEXT_BGR, 0.7, 2)
        draw_text(frame, f"active={active_hand}", (24, 34), TEXT_BGR, 0.7, 2)
        draw_text(frame, f"hand_score={hand_score:.3f}", (24, 64), HAND_ON_BGR if hand_on else TEXT_BGR, 0.7, 2)
        draw_text(frame, f"floor_score={floor_score:.3f}", (24, 94), FLOOR_ON_BGR if floor_on else TEXT_BGR, 0.7, 2)

        if hand_on:
            cv2.rectangle(frame, (w - 230, 18), (w - 28, 54), HAND_ON_BGR, -1)
            draw_text(frame, "HAND CONTACT", (w - 218, 44), (20, 20, 20), 0.7, 2)
        if floor_on:
            cv2.rectangle(frame, (w - 230, 62), (w - 28, 98), FLOOR_ON_BGR, -1)
            draw_text(frame, "FLOOR CONTACT", (w - 221, 88), (20, 20, 20), 0.7, 2)

        if mode == 'centers':
            for item in center_rows.get(frame_id, []):
                label = item['contact_type']
                score = float(item['score'])
                if 'hand' in label:
                    cv2.circle(frame, ball_pt, ball_r + 10, CENTER_BGR, 2, cv2.LINE_AA)
                    draw_text(frame, f"HAND EVENT {score:.2f}", (24, 126), CENTER_BGR, 0.7, 2)
                if 'floor' in label:
                    cv2.circle(frame, ball_pt, ball_r + 16, FLOOR_ON_BGR, 2, cv2.LINE_AA)
                    draw_text(frame, f"FLOOR EVENT {score:.2f}", (24, 156), FLOOR_ON_BGR, 0.7, 2)

        if mode == 'states' and int(state['multi_contact_state']) == 1:
            cv2.rectangle(frame, (8, 8), (w - 8, h - 8), CENTER_BGR, 3)
            draw_text(frame, 'MULTI CONTACT', (w - 230, 132), CENTER_BGR, 0.7, 2)

        writer.write(frame)
        if not preview_written and ((mode == 'centers' and frame_id in center_rows) or (mode == 'states' and (hand_on or floor_on))):
            cv2.imwrite(str(preview_path), frame)
            preview_written = True

    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description='Render contact candidate overlays onto the video.')
    parser.add_argument('--sample-dir', type=Path, default=Path('samples/basketball_01'))
    parser.add_argument('--body-model-root', type=Path, default=Path('third-party/GVHMR/inputs/checkpoints/body_models'))
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / 'results'
    cc_dir = results_dir / 'contact_candidates'
    frames_dir = sample_dir / 'frames'
    out_dir = cc_dir / 'renders'
    out_dir.mkdir(parents=True, exist_ok=True)

    ball_rows = read_csv_rows(results_dir / 'tracking' / 'ball_trajectory.csv')
    shared_rows = read_csv_rows(results_dir / 'pose6d_sharedcam' / 'ball_pose6d_sharedcam_trajectory.csv')
    state_rows = read_csv_rows(cc_dir / 'contact_state_frames.csv')
    center_list = read_csv_rows(cc_dir / 'contact_candidates_labeled.csv')
    center_rows: dict[int, list[dict[str, str]]] = {}
    for row in center_list:
        center_rows.setdefault(int(row['frame']), []).append(row)

    if not (len(ball_rows) == len(shared_rows) == len(state_rows)):
        raise RuntimeError('Frame count mismatch among tracking/sharedcam/contact state rows')

    for ball, shared in zip(ball_rows, shared_rows):
        ball['floor_v'] = shared['floor_v']

    human = read_human_result(results_dir / 'gvhmr' / 'result.pkl')
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human['K_fullimg'], dtype=np.float64)
    if len(joints) != len(ball_rows):
        raise RuntimeError('Frame count mismatch between GVHMR joints and ball rows')

    times = np.asarray([float(r['time']) for r in ball_rows], dtype=np.float64)
    fps = 24.0
    if len(times) >= 2:
        dt = np.median(np.diff(times))
        if dt > 1e-6:
            fps = float(round(1.0 / dt))

    render_video(
        frames_dir,
        out_dir / 'contact_states_overlay.mp4',
        out_dir / 'contact_states_overlay_preview.png',
        fps,
        ball_rows,
        state_rows,
        center_rows,
        joints,
        K,
        mode='states',
    )
    render_video(
        frames_dir,
        out_dir / 'contact_centers_overlay.mp4',
        out_dir / 'contact_centers_overlay_preview.png',
        fps,
        ball_rows,
        state_rows,
        center_rows,
        joints,
        K,
        mode='centers',
    )

    print(f"Wrote {out_dir / 'contact_states_overlay.mp4'}")
    print(f"Wrote {out_dir / 'contact_centers_overlay.mp4'}")


if __name__ == '__main__':
    main()
