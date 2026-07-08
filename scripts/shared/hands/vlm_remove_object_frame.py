"""Remove the held object from selected frames via Gemini (Nano Banana) image edit.

Goal: HaMeR is anchored to GVHMR wrists and "detects" a hand on every frame, but when
the hand grips/occludes the object the predicted finger pose can be wrong. We regenerate
the chosen frame(s) with the object erased and the hand drawn naturally, so HaMeR sees an
unoccluded hand. Used for the stitch anchor frame (and optionally per grip frame).

The edit is instruction-based (Nano Banana needs no mask). If a SAM2 object mask exists it
is sent as a second image to localise what to remove. The original frame is never modified;
outputs go under results/<out-subdir>/.

Auth: reads GEMINI_API_KEY from the environment (query-param auth).

Example:
  python scripts/shared/hands/vlm_remove_object_frame.py \
    --sample-dir samples/basketball_01 --object-name basketball --frames 1
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

from PIL import Image

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

PROMPT_TMPL = (
    "Remove the {obj} that the person is holding from this image. "
    "Show the person's hand and fingers naturally and realistically as if the hand were "
    "empty, with correct skin and finger anatomy. Keep the person's body pose, clothing, "
    "the background, lighting, colours and camera viewpoint EXACTLY identical. "
    "Do not move or re-pose anything except removing the {obj} and completing the hand. "
    "Output a photorealistic image the same size as the input."
)


def b64_png(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode()


_LOCAL_PIPE = None


def call_local_sd(model_id: str, frame_png: Path, mask_png: Path | None, obj: str) -> bytes:
    """Local Stable-Diffusion inpaint: erase the masked object region and fill with a hand.

    Mask-guided (needs the SAM2 object mask). Runs on the 8GB GPU in fp16. NOTE: this paints a
    *hallucinated* hand into the object region guided by the prompt — it does not recover the
    true occluded hand, so treat the result as a plausibility aid, not ground truth.
    """
    global _LOCAL_PIPE
    import io
    import numpy as np
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    if mask_png is None or not mask_png.exists():
        raise RuntimeError("local SD inpaint needs --use-mask (SAM2 object mask) to know what to remove")
    if _LOCAL_PIPE is None:
        _LOCAL_PIPE = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        ).to("cuda")
    img = Image.open(frame_png).convert("RGB")
    w0, h0 = img.size
    mask = Image.open(mask_png).convert("L")
    # dilate the object mask a little so edges of the object are also repainted
    m = np.array(mask)
    import cv2
    m = cv2.dilate((m > 127).astype(np.uint8) * 255, np.ones((9, 9), np.uint8), iterations=2)
    mask = Image.fromarray(m)
    img_r, mask_r = img.resize((512, 512)), mask.resize((512, 512))
    out = _LOCAL_PIPE(
        prompt=f"a bare human hand with natural fingers, realistic skin, no {obj}, photorealistic",
        negative_prompt=f"{obj}, cup, mug, ceramic, object being held, blurry, deformed hand",
        image=img_r, mask_image=mask_r, num_inference_steps=35, guidance_scale=8.0,
    ).images[0].resize((w0, h0))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def call_gemini(model: str, key: str, frame_png: Path, mask_png: Path | None, obj: str) -> bytes:
    parts: list[dict] = [{"text": PROMPT_TMPL.format(obj=obj)}]
    parts.append({"inline_data": {"mime_type": "image/png", "data": b64_png(frame_png)}})
    if mask_png is not None and mask_png.exists():
        parts[0]["text"] += (
            " The white region in the second image marks where the object currently is."
        )
        parts.append({"inline_data": {"mime_type": "image/png", "data": b64_png(mask_png)}})
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    url = f"{API_ROOT}/{model}:generateContent?key={key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    # pull the first inline image out of the candidate parts
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise RuntimeError(
        "no image in Gemini response: "
        + json.dumps(resp)[:500]
    )


def side_by_side(orig: Path, edited: Path, out: Path) -> None:
    a = Image.open(orig).convert("RGB")
    b = Image.open(edited).convert("RGB").resize(a.size)
    canvas = Image.new("RGB", (a.width * 2, a.height), "black")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    canvas.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--object-name", default=None, help="defaults to metadata detection_text / 'object'")
    ap.add_argument("--frames", default="1", help="comma list of 1-based frame numbers, e.g. 1,60,120")
    ap.add_argument("--backend", choices=("gemini", "local"), default="gemini")
    ap.add_argument("--model", default="", help="default: gemini-2.5-flash-image | stabilityai/stable-diffusion-2-inpainting")
    ap.add_argument("--out-subdir", default="hands/anchor_clean")
    ap.add_argument("--use-mask", action="store_true", help="send SAM2 object mask (required for local backend)")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY") if args.backend == "gemini" else None
    if args.backend == "gemini" and not key:
        sys.exit("GEMINI_API_KEY not set in environment")
    model = args.model or ("gemini-2.5-flash-image" if args.backend == "gemini"
                           else "stabilityai/stable-diffusion-2-inpainting")

    sample = args.sample_dir
    obj = args.object_name
    if obj is None:
        meta = sample / "metadata.json"
        if meta.exists():
            obj = json.loads(meta.read_text()).get("detection_text", "object").strip(". ")
        else:
            obj = "object"

    frames_dir = sample / "frames"
    masks_dir = sample / "results" / "segmentation" / "masks"
    out_dir = sample / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = [int(x) for x in args.frames.split(",") if x.strip()]
    print(f"object='{obj}'  backend={args.backend}  model={model}  frames={frame_ids}  out={out_dir}")
    for fid in frame_ids:
        fp = frames_dir / f"{fid:05d}.png"
        if not fp.exists():
            print(f"  SKIP {fid}: {fp} missing")
            continue
        mp = (masks_dir / f"{fid:05d}_mask.png") if (args.use_mask or args.backend == "local") else None
        try:
            if args.backend == "local":
                png = call_local_sd(model, fp, mp, obj)
            else:
                png = call_gemini(model, key, fp, mp, obj)
        except Exception as e:
            print(f"  FAIL {fid}: {e}")
            continue
        out = out_dir / f"{fid:05d}.png"
        out.write_bytes(png)
        prev = out_dir / f"{fid:05d}_compare.png"
        side_by_side(fp, out, prev)
        print(f"  OK   {fid}: {out.name}  ({len(png)} bytes)  preview={prev.name}")


if __name__ == "__main__":
    main()
