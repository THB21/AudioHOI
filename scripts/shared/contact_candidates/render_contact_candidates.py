#!/usr/bin/env python3
"""Render contact-candidate overlays back onto the original video frames."""

from __future__ import annotations

import argparse
import csv
import pickle
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import smplx

BALL_BGR = (32, 122, 219)
LEFT_ANCHOR_BGR = (70, 180, 70)
RIGHT_ANCHOR_BGR = (70, 200, 230)
ANCHOR_ON_BGR = (72, 196, 92)
FLOOR_ON_BGR = (40, 190, 250)
CENTER_BGR = (50, 50, 240)
TEXT_BGR = (235, 235, 235)
SHADOW_BGR = (15, 15, 15)
FLOOR_LINE_BGR = (70, 220, 220)
BODY_LINE_BGR = (104, 78, 214)
BODY_POINT_BGR = (194, 228, 244)
LEFT_FOOT_IDS = [10]
RIGHT_FOOT_IDS = [11]
BODY_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
    (12, 15),
]


def resolve_ffmpeg_bin() -> str:
    candidates = [
        Path('/home/yang/miniconda3/bin/ffmpeg'),
        Path('/usr/bin/ffmpeg'),
        Path('/usr/local/bin/ffmpeg'),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        probe = subprocess.run([str(candidate), '-hide_banner', '-encoders'], capture_output=True, text=True)
        if probe.returncode == 0 and 'libx264' in probe.stdout:
            return str(candidate)
    return 'ffmpeg'


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
    if K.ndim == 3:
        K = K[0]
    z = np.clip(points_cam[:, 2], 1e-6, None)
    u = K[0, 0] * (points_cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (points_cam[:, 1] / z) + K[1, 2]
    valid = points_cam[:, 2] > 1e-6
    return np.column_stack([u, v]), valid


def draw_text(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.7, thickness: int = 2) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, SHADOW_BGR, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_body_skeleton(frame: np.ndarray, joints_cam: np.ndarray, K: np.ndarray) -> None:
    uv, _ = project_points(joints_cam[:22], K[None])
    valid = joints_cam[:22, 2] > 1e-6
    for a, b in BODY_EDGES:
        if not (valid[a] and valid[b]):
            continue
        pa = tuple(np.round(uv[a]).astype(int))
        pb = tuple(np.round(uv[b]).astype(int))
        cv2.line(frame, pa, pb, BODY_LINE_BGR, 2, cv2.LINE_AA)
    for p in uv[valid]:
        cv2.circle(frame, tuple(np.round(p).astype(int)), 2, BODY_POINT_BGR, -1, cv2.LINE_AA)


def compute_stable_foot_points(joints: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_cam = joints[:, LEFT_FOOT_IDS, :].mean(axis=1)
    right_cam = joints[:, RIGHT_FOOT_IDS, :].mean(axis=1)
    left_uv, left_valid = project_points(left_cam, K)
    right_uv, right_valid = project_points(right_cam, K)
    return left_uv, left_valid, right_uv, right_valid


def build_interval_side_map(interval_rows: list[dict[str, str]]) -> dict[int, str]:
    side_by_frame: dict[int, str] = {}
    for row in interval_rows:
        if row.get('contact_type') != 'anchor_contact_state':
            continue
        target = row.get('target', '')
        if target not in {'left_foot', 'right_foot'}:
            continue
        side = 'left' if target.startswith('left') else 'right'
        start = int(row['start_frame'])
        end = int(row['end_frame'])
        for frame in range(start, end + 1):
            side_by_frame[frame] = side
    return side_by_frame


def draw_active_foot_marker(frame: np.ndarray, side: str | None, left_uv: np.ndarray, left_valid: bool, right_uv: np.ndarray, right_valid: bool) -> None:
    if left_valid:
        left_pt = tuple(np.round(left_uv).astype(int))
        cv2.circle(frame, left_pt, 4, LEFT_ANCHOR_BGR, -1, cv2.LINE_AA)
        cv2.circle(frame, left_pt, 6, SHADOW_BGR, 1, cv2.LINE_AA)
        draw_text(frame, 'L', (left_pt[0] + 8, left_pt[1] - 8), LEFT_ANCHOR_BGR, 0.55, 2)
    if right_valid:
        right_pt = tuple(np.round(right_uv).astype(int))
        cv2.circle(frame, right_pt, 4, RIGHT_ANCHOR_BGR, -1, cv2.LINE_AA)
        cv2.circle(frame, right_pt, 6, SHADOW_BGR, 1, cv2.LINE_AA)
        draw_text(frame, 'R', (right_pt[0] + 8, right_pt[1] - 8), RIGHT_ANCHOR_BGR, 0.55, 2)

    if side == 'left' and left_valid:
        pt = tuple(np.round(left_uv).astype(int))
        color = LEFT_ANCHOR_BGR
        label = 'JUDGED L'
    elif side == 'right' and right_valid:
        pt = tuple(np.round(right_uv).astype(int))
        color = RIGHT_ANCHOR_BGR
        label = 'JUDGED R'
    else:
        return
    cv2.circle(frame, pt, 8, color, -1, cv2.LINE_AA)
    cv2.circle(frame, pt, 12, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, pt, 14, SHADOW_BGR, 1, cv2.LINE_AA)
    draw_text(frame, label, (pt[0] + 12, pt[1] - 14), color, 0.65, 2)


def safe_int(s: str | None) -> int | None:
    if s is None or s == '':
        return None
    return int(round(float(s)))


def render_video(
    frames_dir: Path,
    out_path: Path,
    preview_path: Path,
    fps: float,
    ball_rows: list[dict[str, str]],
    state_rows: list[dict[str, str]],
    center_rows: dict[int, list[dict[str, str]]],
    interval_side_by_frame: dict[int, str],
    left_foot_uv: np.ndarray,
    left_foot_valid: np.ndarray,
    right_foot_uv: np.ndarray,
    right_foot_valid: np.ndarray,
    joints: np.ndarray,
    K: np.ndarray,
    mode: str,
) -> None:
    first = cv2.imread(str(frames_dir / '00001.png'))
    if first is None:
        raise RuntimeError(f"Could not read frames from {frames_dir}")
    h, w = first.shape[:2]

    with tempfile.TemporaryDirectory(prefix='contact_candidates_frames_') as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        preview_written = False
        rendered_count = 0
        for i, (ball, state) in enumerate(zip(ball_rows, state_rows)):
            frame_id = int(ball['frame'])
            frame = cv2.imread(str(frames_dir / f"{frame_id:05d}.png"))
            if frame is None:
                continue

            draw_body_skeleton(frame, joints[i], K[i])

            ball_pt = (int(round(float(ball['ball_center_x']))), int(round(float(ball['ball_center_y']))))
            ball_r = max(4, int(round(float(ball['radius']))))
            cv2.circle(frame, ball_pt, ball_r, BALL_BGR, 2, cv2.LINE_AA)
            cv2.circle(frame, ball_pt, 3, BALL_BGR, -1, cv2.LINE_AA)

            floor_y = safe_int(ball.get('floor_v'))
            if floor_y is not None and floor_y > 0:
                cv2.line(frame, (0, floor_y), (w - 1, floor_y), FLOOR_LINE_BGR, 2, cv2.LINE_AA)

            anchor_state_key = 'anchor_contact_state' if 'anchor_contact_state' in state else 'hand_contact_state' if 'hand_contact_state' in state else 'foot_contact_state'
            anchor_on = int(state[anchor_state_key]) == 1
            floor_on = int(state['floor_contact_state']) == 1
            anchor_score = float(state['anchor_score'])
            floor_score = float(state['floor_score'])
            anchor_type = state.get('anchor_type', '')
            active_probe = state.get('active_anchor_probe', '')
            fixed_side = interval_side_by_frame.get(frame_id, state.get('active_anchor_side', '')) if anchor_on else ''

            if mode == 'centers':
                for item in center_rows.get(frame_id, []):
                    target = item.get('target', '')
                    if target == 'left_foot':
                        fixed_side = 'left'
                        break
                    if target == 'right_foot':
                        fixed_side = 'right'
                        break

            draw_active_foot_marker(frame, fixed_side, left_foot_uv[i], bool(left_foot_valid[i]), right_foot_uv[i], bool(right_foot_valid[i]))

            cv2.rectangle(frame, (w - 108, 12), (w - 12, 52), (0, 0, 0), -1)
            draw_text(frame, f"{frame_id}", (w - 90, 40), (255, 255, 255), 1.0, 2)
            draw_text(frame, f"anchor={anchor_type}:{fixed_side}  probe={active_probe}", (24, 34), TEXT_BGR, 0.7, 2)
            draw_text(frame, f"anchor_score={anchor_score:.3f}", (24, 64), ANCHOR_ON_BGR if anchor_on else TEXT_BGR, 0.7, 2)
            draw_text(frame, f"floor_score={floor_score:.3f}", (24, 94), FLOOR_ON_BGR if floor_on else TEXT_BGR, 0.7, 2)

            if anchor_on:
                cv2.rectangle(frame, (w - 250, 18), (w - 28, 54), ANCHOR_ON_BGR, -1)
                draw_text(frame, "ANCHOR CONTACT", (w - 236, 44), (20, 20, 20), 0.7, 2)
            if floor_on:
                cv2.rectangle(frame, (w - 250, 62), (w - 28, 98), FLOOR_ON_BGR, -1)
                draw_text(frame, "FLOOR CONTACT", (w - 223, 88), (20, 20, 20), 0.7, 2)

            if mode == 'centers':
                for item in center_rows.get(frame_id, []):
                    label = item['contact_type']
                    score = float(item['score'])
                    if 'anchor' in label:
                        cv2.circle(frame, ball_pt, ball_r + 10, CENTER_BGR, 2, cv2.LINE_AA)
                        draw_text(frame, f"ANCHOR EVENT {score:.2f}", (24, 126), CENTER_BGR, 0.7, 2)
                    if 'floor' in label:
                        cv2.circle(frame, ball_pt, ball_r + 16, FLOOR_ON_BGR, 2, cv2.LINE_AA)
                        draw_text(frame, f"FLOOR EVENT {score:.2f}", (24, 156), FLOOR_ON_BGR, 0.7, 2)

            if mode == 'states' and int(state['multi_contact_state']) == 1:
                cv2.rectangle(frame, (8, 8), (w - 8, h - 8), CENTER_BGR, 3)
                draw_text(frame, 'MULTI CONTACT', (w - 230, 132), CENTER_BGR, 0.7, 2)

            frame_path = tmp_dir / f"frame_{rendered_count + 1:05d}.png"
            cv2.imwrite(str(frame_path), frame)
            rendered_count += 1
            if not preview_written and ((mode == 'centers' and frame_id in center_rows) or (mode == 'states' and (anchor_on or floor_on))):
                cv2.imwrite(str(preview_path), frame)
                preview_written = True

        ffmpeg_cmd = [
            resolve_ffmpeg_bin(),
            '-y',
            '-hide_banner',
            '-loglevel', 'error',
            '-framerate', f'{fps:.6f}',
            '-i', str(tmp_dir / 'frame_%05d.png'),
            '-an',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(out_path),
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {out_path} with exit code {result.returncode}: {result.stderr.strip()}")


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
    state_rows = read_csv_rows(cc_dir / 'contact_state_frames.csv')
    floor_rows = read_csv_rows(cc_dir / 'floor_contact_candidates.csv')
    center_list = read_csv_rows(cc_dir / 'contact_candidates_labeled.csv')
    interval_rows = read_csv_rows(cc_dir / 'contact_intervals.csv')
    center_rows: dict[int, list[dict[str, str]]] = {}
    for row in center_list:
        center_rows.setdefault(int(row['frame']), []).append(row)

    if not (len(ball_rows) == len(floor_rows) == len(state_rows)):
        raise RuntimeError('Frame count mismatch among tracking/floor-contact/contact state rows')

    for ball, floor in zip(ball_rows, floor_rows):
        ball['floor_v'] = floor['floor_v']

    human = read_human_result(results_dir / 'gvhmr' / 'result.pkl')
    joints = build_body_joints(args.body_model_root, human)
    K = np.asarray(human['K_fullimg'], dtype=np.float64)
    if len(joints) < len(ball_rows):
        raise RuntimeError('Frame count mismatch between GVHMR joints and ball rows')
    if len(joints) != len(ball_rows):
        joints = joints[:len(ball_rows)]
        K = K[:len(ball_rows)]

    left_foot_uv, left_foot_valid, right_foot_uv, right_foot_valid = compute_stable_foot_points(joints, K)
    interval_side_by_frame = build_interval_side_map(interval_rows)

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
        interval_side_by_frame,
        left_foot_uv,
        left_foot_valid,
        right_foot_uv,
        right_foot_valid,
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
        interval_side_by_frame,
        left_foot_uv,
        left_foot_valid,
        right_foot_uv,
        right_foot_valid,
        joints,
        K,
        mode='centers',
    )

    print(f"Wrote {out_dir / 'contact_states_overlay.mp4'}")
    print(f"Wrote {out_dir / 'contact_centers_overlay.mp4'}")


if __name__ == '__main__':
    main()
