#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

TEXT_BGR = (235, 235, 235)
SHADOW_BGR = (15, 15, 15)
REF_BGR = (32, 122, 219)
SUPPORT_BGR = (70, 220, 220)
CONTACT_BGR = (50, 50, 240)
ACTIVE_BGR = (210, 90, 240)
ANCHOR_ON_BGR = (72, 196, 92)
FLOOR_ON_BGR = (40, 190, 250)
EVENT_BGR = (255, 255, 255)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'No rows found in {path}')
    return rows


def resolve_ffmpeg_bin() -> str:
    candidates = [
        Path('/home/yang/miniconda3/bin/ffmpeg'),
        Path('/usr/bin/ffmpeg'),
        Path('/usr/local/bin/ffmpeg'),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return 'ffmpeg'


def draw_text(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.65, thickness: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, SHADOW_BGR, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, '')
    return float(value) if value not in {'', None} else float('nan')


def parse_int01(row: dict[str, str], key: str) -> bool:
    value = row.get(key, '')
    return value not in {'', None} and int(round(float(value))) == 1


def pt(u: float, v: float) -> tuple[int, int] | None:
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(u)), int(round(v))


def draw_point(img: np.ndarray, uv: tuple[int, int] | None, color: tuple[int, int, int], label: str, radius: int = 5) -> None:
    if uv is None:
        return
    cv2.circle(img, uv, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, uv, radius + 2, SHADOW_BGR, 1, cv2.LINE_AA)
    draw_text(img, label, (uv[0] + 8, uv[1] - 8), color, 0.55, 2)


def render_mode(
    frames_dir: Path,
    out_path: Path,
    preview_path: Path,
    fps: float,
    proxy_rows: list[dict[str, str]],
    state_rows: dict[int, dict[str, str]],
    event_rows: dict[int, list[dict[str, str]]],
    mode: str,
) -> None:
    first = cv2.imread(str(frames_dir / '00001.png'))
    if first is None:
        raise RuntimeError(f'Could not read frames from {frames_dir}')
    h, w = first.shape[:2]
    ffmpeg_bin = resolve_ffmpeg_bin()

    with tempfile.TemporaryDirectory(prefix='object_contact_render_') as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        preview_written = False
        rendered = 0
        for row in proxy_rows:
            frame_id = int(row['frame'])
            frame = cv2.imread(str(frames_dir / f'{frame_id:05d}.png'))
            if frame is None:
                continue

            state = state_rows.get(frame_id, {})
            events = event_rows.get(frame_id, [])
            ref_uv = pt(parse_float(row, 'ref_u'), parse_float(row, 'ref_v'))
            support_uv = pt(parse_float(row, 'support_u'), parse_float(row, 'support_v'))
            contact_uv = pt(parse_float(row, 'contact_u'), parse_float(row, 'contact_v'))
            active_uv = pt(parse_float(row, 'active_part_u'), parse_float(row, 'active_part_v'))

            draw_point(frame, ref_uv, REF_BGR, 'REF')
            draw_point(frame, support_uv, SUPPORT_BGR, 'SUP')
            draw_point(frame, contact_uv, CONTACT_BGR, 'CPX')
            draw_point(frame, active_uv, ACTIVE_BGR, 'HUM')

            if ref_uv is not None and support_uv is not None:
                cv2.line(frame, ref_uv, support_uv, SUPPORT_BGR, 1, cv2.LINE_AA)
            if active_uv is not None and contact_uv is not None:
                cv2.line(frame, active_uv, contact_uv, CONTACT_BGR, 1, cv2.LINE_AA)

            human_on = parse_int01(state, 'human_contact_state')
            floor_on = parse_int01(state, 'floor_contact_state')
            anchor_on = parse_int01(state, 'anchor_contact_state')
            if human_on and contact_uv is not None:
                cv2.circle(frame, contact_uv, 10, ANCHOR_ON_BGR, 2, cv2.LINE_AA)
            if floor_on and support_uv is not None:
                cv2.circle(frame, support_uv, 10, FLOOR_ON_BGR, 2, cv2.LINE_AA)
            if mode == 'events':
                for ev in events:
                    target = ev.get('target', '')
                    if target == 'floor' and support_uv is not None:
                        cv2.circle(frame, support_uv, 14, EVENT_BGR, 2, cv2.LINE_AA)
                    elif contact_uv is not None:
                        cv2.circle(frame, contact_uv, 14, EVENT_BGR, 2, cv2.LINE_AA)

            y = 28
            draw_text(frame, f'frame={frame_id}', (20, y), TEXT_BGR); y += 28
            draw_text(frame, f'label={row.get("active_label", "")}', (20, y), TEXT_BGR); y += 28
            if state:
                draw_text(frame, f'anchor={state.get("anchor_score", "")}', (20, y), ANCHOR_ON_BGR if anchor_on else TEXT_BGR); y += 28
                draw_text(frame, f'floor={state.get("floor_score", "")}', (20, y), FLOOR_ON_BGR if floor_on else TEXT_BGR); y += 28
                draw_text(frame, f'gap_px={state.get("min_object_boundary_gap_px", "")}', (20, y), TEXT_BGR); y += 28
            draw_text(frame, f'offset_z={row.get("contact_depth_offset_m", "")}', (20, y), TEXT_BGR); y += 28
            draw_text(frame, f'cproxy={row.get("contact_proxy_name", "")}', (20, y), TEXT_BGR); y += 28
            if events:
                ev_txt = ', '.join(f"{e.get('contact_type','')}:{e.get('target','')}" for e in events[:2])
                draw_text(frame, ev_txt, (20, y), EVENT_BGR); y += 28

            cv2.rectangle(frame, (w - 150, 12), (w - 20, 48), (0, 0, 0), -1)
            draw_text(frame, mode.upper(), (w - 130, 38), TEXT_BGR, 0.8, 2)

            out_frame = tmp_dir / f'{rendered + 1:05d}.png'
            cv2.imwrite(str(out_frame), frame)
            if not preview_written:
                cv2.imwrite(str(preview_path), frame)
                preview_written = True
            rendered += 1

        if rendered == 0:
            raise RuntimeError('No frames rendered')

        subprocess.run([
            ffmpeg_bin,
            '-y',
            '-framerate', f'{fps:.6f}',
            '-i', str(tmp_dir / '%05d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(out_path),
        ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Render experimental object-proxy contact candidate overlays.')
    parser.add_argument('--sample-dir', type=Path, required=True)
    parser.add_argument('--contact-subdir', type=str, default='contact_candidates_object_proxy_test')
    parser.add_argument('--proxy-subdir', type=str, default='object_proxy_observations_test')
    parser.add_argument('--fps', type=float, default=24.0)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    results_dir = sample_dir / 'results'
    frames_dir = sample_dir / 'frames'
    contact_dir = results_dir / args.contact_subdir
    proxy_csv = results_dir / args.proxy_subdir / 'object_proxy_observations.csv'
    states_csv = contact_dir / 'contact_state_frames.csv'
    events_csv = contact_dir / 'contact_candidates_labeled.csv'
    render_dir = contact_dir / 'renders'
    render_dir.mkdir(parents=True, exist_ok=True)

    proxy_rows = read_csv_rows(proxy_csv)
    state_rows = {int(r['frame']): r for r in read_csv_rows(states_csv)}
    event_rows: dict[int, list[dict[str, str]]] = {}
    for row in read_csv_rows(events_csv):
        event_rows.setdefault(int(row['frame']), []).append(row)

    render_mode(
        frames_dir,
        render_dir / 'object_contact_states_overlay.mp4',
        render_dir / 'object_contact_states_overlay_preview.png',
        args.fps,
        proxy_rows,
        state_rows,
        event_rows,
        mode='states',
    )
    render_mode(
        frames_dir,
        render_dir / 'object_contact_events_overlay.mp4',
        render_dir / 'object_contact_events_overlay_preview.png',
        args.fps,
        proxy_rows,
        state_rows,
        event_rows,
        mode='events',
    )
    print(f'render_dir: {render_dir}')


if __name__ == '__main__':
    main()
