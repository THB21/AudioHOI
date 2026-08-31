#!/usr/bin/env python3
"""Export the accepted final 4D-HOI render for all nine cases into one folder."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
FFMPEG = Path("/home/hebestreit/miniforge3/envs/gvhmr/bin/ffmpeg")
FFPROBE = Path("/home/hebestreit/miniforge3/envs/gvhmr/bin/ffprobe")
DEFAULT_OUT = REPO / "deliverables/all_9_final_4d_hoi_videos"

# One accepted full-system scene per case. This is intentionally not an ablation.
SCENES = {
    "01_basketball": REPO / "samples_known_object/01_basketball/results/renders/final_results/with_human/camera3d.mp4",
    "02_mug": REPO / "samples_known_object/02_mug/results/renders/perceptual_full_v2/with_human/camera3d.mp4",
    "03_chair": REPO / "samples_known_object/05_chair/results/renders/benchmark_vlm_qwen/with_human/camera3d.mp4",
    "04_football": REPO / "samples_known_object/10_football/results/renders/final_results/with_human/camera3d.mp4",
    "05_stick": REPO / "samples_known_object/11_stick/results/renders/benchmark_vlm_qwen/with_human/camera3d.mp4",
    "06_back_view_basketball": REPO / "samples_known_object/12_back_view_basketball/results/renders/final_nine_release/with_human/camera3d.mp4",
    "07_volleyball": REPO / "samples_known_object/13_volleyball/results/renders/final_full_4d_hoi/with_human/camera3d.mp4",
    "08_pingpong": REPO / "samples_known_object/14_pingpong_wall/results/renders/final_nine_release/with_human/camera3d.mp4",
    "09_suitcase": REPO / "samples_known_object/15_suitcase_drag/results/renders/final_nine_release/with_human/camera3d.mp4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    cmd = [
        str(FFPROBE), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()
    missing = [str(path) for path in SCENES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing accepted final renders:\n" + "\n".join(missing))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for scene, source in SCENES.items():
        target = args.out_dir / f"{scene}.mp4"
        cmd = [
            str(FFMPEG), "-y", "-i", str(source),
            "-vf", f"scale={args.width}:{args.height}:flags=lanczos",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(target),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = probe(target)
        stream = info["streams"][0]
        if int(stream["width"]) != args.width or int(stream["height"]) != args.height:
            raise RuntimeError(f"resolution check failed for {target}")
        manifest.append({
            "scene": scene,
            "source": str(source),
            "output": str(target),
            "sha256": sha256(target),
            "video": info,
        })
        print(target, flush=True)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "purpose": "one accepted final full-system 4D-HOI video for each of nine scenes",
        "comparison": False,
        "resolution": [args.width, args.height],
        "scenes": manifest,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
