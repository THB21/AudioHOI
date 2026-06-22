#!/usr/bin/env python3
"""Audio -> exact contact point on the object.

Audio onsets give the exact *moment* of contact (see ``audio_semantics.py``); this
script turns each audio-confirmed contact frame into the 2D *point* on the object
surface where the touch happens. That point is the bridge from audio timing to object
**pose** optimization: reprojecting a known surface point to an observed contact pixel
constrains object translation and, for non-spherical objects, rotation (the
``R_kp`` / ``R_contact`` terms in ``method_losses.md`` Section 5).

Two sources:

  geometric (default, no GPU)
      Pick the body part (hand/foot) nearest the object centre at the onset frame, and
      place the contact point on the object surface along the centre->part ray
      (``O + r * unit(part - O)`` for a sphere of radius ``r``). Self-contained: needs
      only the lifted trajectory + GVHMR joints. Exact for a sphere; a good initializer
      for any convex proxy.

  vlm (needs GPU + Qwen weights)
      Crop around the object+hand and ask a vision-language model for the precise
      contact pixel and which object part is touched - the zero-shot "where" to pair
      with audio's "when". Follows the Qwen3-VL pattern used by the mug pipeline
      (scripts/shared/radius_free_proxy/.../run_qwen_mug_contact_keyframes.py). Not run
      on the dev box (8 GB VRAM); structured so it drops onto a GPU machine.

Output: ``results/events/contact_points.csv`` with columns
    frame,time,source,event_type,object_part,confidence,
    contact_u,contact_v,object_u,object_v,part_u,part_v
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "human_ball" / "contact"))
from contact_part_utils import build_contact_part_centers  # noqa: E402


# ---------------------------------------------------------------------------
# IO helpers (kept local so this module is self-contained)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def read_ball_trajectory(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for r in _read_csv(path):
        out[int(float(r["frame"]))] = {
            "time": float(r.get("time", 0.0) or 0.0),
            "tx": float(r["tx"]), "ty": float(r["ty"]), "tz": float(r["tz"]),
            "radius_m": float(r.get("radius_m", 0.0) or 0.0),
        }
    return out


def read_audio_onsets(events_csv: Path, semantics_csv: Path) -> list[dict]:
    """Onset frames with score and (if available) classified event type."""
    if not events_csv.exists():
        return []
    recs = _read_csv(events_csv)
    proms = [float(r.get("prominence", 0.0) or 0.0) for r in recs]
    pmax = max(proms) if proms else 0.0
    types: dict[int, str] = {}
    if semantics_csv.exists():
        types = {int(float(r["frame"])): r["event_type"] for r in _read_csv(semantics_csv)}
    out = []
    for r, prom in zip(recs, proms):
        if not r.get("audio_frame"):
            continue
        fr = int(float(r["audio_frame"]))
        score = float(r["audio_score"]) if r.get("audio_score") else (prom / pmax if pmax > 0 else 0.0)
        out.append({"frame": fr, "time": float(r.get("audio_time", 0.0) or 0.0),
                    "score": float(np.clip(score, 0.0, 1.0)), "event_type": types.get(fr, "")})
    return out


def read_human_result(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    p = data["smpl_params_incam"]
    return {"body_pose": np.asarray(p["body_pose"], np.float32),
            "betas": np.asarray(p["betas"], np.float32),
            "global_orient": np.asarray(p["global_orient"], np.float32),
            "transl": np.asarray(p["transl"], np.float32),
            "K_fullimg": np.asarray(data["K_fullimg"], np.float32)}


def build_body_joints(body_models_root: Path, human: dict[str, np.ndarray]) -> np.ndarray:
    import smplx
    import torch
    model = smplx.create(str(body_models_root), model_type="smplx", gender="neutral", ext="npz",
                         use_pca=False, flat_hand_mean=True, num_betas=10,
                         batch_size=human["transl"].shape[0])
    with torch.inference_mode():
        out = model(body_pose=torch.from_numpy(human["body_pose"]), betas=torch.from_numpy(human["betas"]),
                    global_orient=torch.from_numpy(human["global_orient"]),
                    transl=torch.from_numpy(human["transl"]), return_verts=False)
    return out.joints.detach().cpu().numpy().astype(np.float64)


def _project(p_cam: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    z = max(float(p_cam[2]), 1e-6)
    return float(K[0, 0] * p_cam[0] / z + K[0, 2]), float(K[1, 1] * p_cam[1] / z + K[1, 2])


# ---------------------------------------------------------------------------
# Geometric contact point
# ---------------------------------------------------------------------------

def geometric_contact_point(O: np.ndarray, radius_m: float, part_centers: dict[str, np.ndarray],
                            idx: int) -> tuple[str, np.ndarray, np.ndarray]:
    """Nearest body part to the object centre, and the object-surface point on the
    centre->part ray. Returns (part_name, surface_point_cam, part_cam)."""
    best_name, best_d, best_p = None, np.inf, None
    for name, arr in part_centers.items():
        if idx >= len(arr):
            continue
        p = np.asarray(arr[idx], dtype=np.float64)
        if not np.all(np.isfinite(p)):
            continue
        d = float(np.linalg.norm(p - O))
        if d < best_d:
            best_name, best_d, best_p = name, d, p
    if best_name is None:
        return "unknown", O.copy(), O.copy()
    direction = best_p - O
    norm = float(np.linalg.norm(direction))
    surface = O + (radius_m * direction / norm) if norm > 1e-9 else O.copy()
    return best_name, surface, best_p


# ---------------------------------------------------------------------------
# VLM contact point (GPU; Qwen3-VL) - structured after the mug pipeline's pattern
# ---------------------------------------------------------------------------

_VLM_PROMPT = (
    "You are shown a cropped video frame of a person interacting with an object. "
    "An impact sound occurs at this exact frame, so the hand or foot is touching the "
    "object now. Reply with ONE line of JSON: "
    '{"contact_u": <pixel x in the crop>, "contact_v": <pixel y in the crop>, '
    '"object_part": "<which part of the object is touched>", '
    '"body_part": "left_hand|right_hand|left_foot|right_foot", "confidence": <0..1>}. '
    "The pixel must lie on the object surface at the contact, not on the hand."
)


def vlm_contact_point(frame_path: Path, crop_box: tuple[int, int, int, int], model) -> dict:
    """Query a loaded Qwen3-VL model for the contact pixel inside ``crop_box``.

    ``model`` is the tuple returned by :func:`load_qwen`. Returns crop-local pixel +
    labels; the caller maps the pixel back to full-image coordinates. Kept import-light
    so the geometric path never pulls in transformers.
    """
    import json
    from PIL import Image

    proc, net = model
    x0, y0, x1, y1 = crop_box
    crop = Image.open(frame_path).convert("RGB").crop((x0, y0, x1, y1))
    messages = [{"role": "user", "content": [{"type": "image", "image": crop},
                                             {"type": "text", "text": _VLM_PROMPT}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[crop], return_tensors="pt").to(net.device)
    out = net.generate(**inputs, max_new_tokens=128, do_sample=False)
    reply = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    try:
        start, end = reply.index("{"), reply.rindex("}") + 1
        data = json.loads(reply[start:end])
    except Exception:
        data = {}
    return {
        "contact_u": float(data.get("contact_u", (x1 - x0) / 2)) + x0,
        "contact_v": float(data.get("contact_v", (y1 - y0) / 2)) + y0,
        "object_part": str(data.get("object_part", "unknown")),
        "body_part": str(data.get("body_part", "unknown")),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
    }


def load_qwen(model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
    """Load Qwen3-VL (GPU). Imports are local so geometric runs need no transformers."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    net = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    net.eval()
    return proc, net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--source", choices=["geometric", "vlm"], default="geometric")
    ap.add_argument("--ball-trajectory-csv", type=Path, default=None)
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--score-thresh", type=float, default=0.3,
                    help="only emit contact points for onsets at/above this audio score")
    ap.add_argument("--vlm-model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--crop-margin", type=int, default=80, help="px margin around object+part for the VLM crop")
    args = ap.parse_args(argv)

    sample = args.sample_dir
    results = sample / "results"
    traj_csv = args.ball_trajectory_csv or (results / "pose6d_sharedcam_depthv3" / "ball_pose6d_sharedcam_trajectory.csv")

    traj = read_ball_trajectory(traj_csv)
    onsets = read_audio_onsets(results / "events" / "audio_events.csv",
                               results / "events" / "audio_semantics.csv")
    onsets = [o for o in onsets if o["score"] >= args.score_thresh and o["frame"] in traj]
    if not onsets:
        print("no audio onsets above threshold with a trajectory frame; nothing to do")
        return

    human = read_human_result(results / "gvhmr" / "result.pkl")
    K_all = np.asarray(human["K_fullimg"], dtype=np.float64)
    joints = build_body_joints(args.body_model_root, human)
    part_centers = build_contact_part_centers(joints)

    qwen = load_qwen(args.vlm_model_id) if args.source == "vlm" else None
    frames_dir = sample / "frames"

    rows = []
    for o in onsets:
        fr = o["frame"]
        idx = fr - 1
        K = K_all[idx] if K_all.ndim == 3 else K_all
        t = traj[fr]
        O = np.array([t["tx"], t["ty"], t["tz"]], dtype=np.float64)
        part_name, surface, part_cam = geometric_contact_point(O, t["radius_m"], part_centers, idx)
        ou, ov = _project(O, K)
        pu, pv = _project(part_cam, K)
        cu, cv = _project(surface, K)
        source, conf, obj_part = "geometric", o["score"], "surface"

        if args.source == "vlm":
            xs, ys = sorted([ou, pu]), sorted([ov, pv])
            box = (int(xs[0] - args.crop_margin), int(ys[0] - args.crop_margin),
                   int(xs[1] + args.crop_margin), int(ys[1] + args.crop_margin))
            frame_path = frames_dir / f"{fr:05d}.png"
            if frame_path.exists():
                v = vlm_contact_point(frame_path, box, qwen)
                cu, cv, obj_part, conf, source = v["contact_u"], v["contact_v"], v["object_part"], v["confidence"], "vlm"
                if v["body_part"] in part_centers:
                    part_name = v["body_part"]

        rows.append({
            "frame": fr, "time": f"{t['time']:.6f}", "source": source,
            "event_type": o["event_type"], "object_part": obj_part, "body_part": part_name,
            "confidence": f"{conf:.6f}",
            "contact_u": f"{cu:.3f}", "contact_v": f"{cv:.3f}",
            "object_u": f"{ou:.3f}", "object_v": f"{ov:.3f}",
            "part_u": f"{pu:.3f}", "part_v": f"{pv:.3f}",
        })

    out_csv = results / "events" / "contact_points.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "time", "source", "event_type", "object_part", "body_part", "confidence",
              "contact_u", "contact_v", "object_u", "object_v", "part_u", "part_v"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} contact points ({args.source}) -> {out_csv}")


if __name__ == "__main__":
    main()
