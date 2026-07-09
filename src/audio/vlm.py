"""Stage 6 — VLM event analysis with audio-refined timing.

For every candidate event (audio- and/or visual-seeded) we cut a short window of frames
*around* the event — a few before, the event frame, a few after — montage them, and let a
VLM read "what happens, with which entities, how hard, is it relevant". The audio around the
moment then refines the *timing* (audio is temporally precise; vision gives the semantics),
and a relevance gate drops audio that is not an actual interaction.

Backends are pluggable:
  HeuristicBackend  no model — derives the structured analysis from the fused record + audio
                    energy (always runs; proves the plumbing and gives a sane baseline)
  MoondreamBackend  real local VLM (vikhyatk/moondream2) — must be enabled explicitly
                    (downloads + runs remote code), or point --vlm-model at a local path

The montage saved per event is exactly the image a VLM consumes, so the window extraction is
useful on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Frame-window extraction (the VLM input)
# ---------------------------------------------------------------------------

DEFAULT_OFFSETS = (-4, -2, 0, 2, 4)

# Adopted from the teammate's run_qwen_mug_contact_keyframes.py — fine object-side grasp prompt.
GRASP_PROMPT = (
    "Inspect this hand-object crop. Ignore overlays/text. Label only the visible hand-object "
    "grasp/contact. Do not treat mouth/rim drinking as a hand grasp. Return ONLY one-line valid JSON:\n"
    '{"contact_visible":true|false,"hand_side":"left_hand|right_hand|unknown",'
    '"contact_fingers":["thumb|index|middle|ring|pinky|palm|unknown"],'
    '"primary_contact_finger":"thumb|index|middle|ring|pinky|palm|unknown",'
    '"object_part":"handle|body|rim|bottom|side|top|unknown",'
    '"object_region":"upper|middle|lower|inner|outer|front|side|unknown",'
    '"grasp_type":"pinch|hook|palm_support|body_grasp|not_grasp|unknown",'
    '"impact":"none|light|medium|hard","relevant":true|false,"confidence":0.0-1.0}')


def _frame_files(sample_dir: Path) -> list[Path]:
    fdir = Path(sample_dir) / "frames"
    return sorted(fdir.glob("*.png")) or sorted(fdir.glob("*.jpg"))


def extract_window(sample_dir: Path, frame: int, offsets=DEFAULT_OFFSETS,
                   obj_uv: tuple[float, float] | None = None, out_path: Path | None = None):
    """Montage the before/event/after frames into one captioned image. Returns (PIL.Image, frames)."""
    import cv2
    from PIL import Image

    files = _frame_files(sample_dir)
    if not files:
        return None, []
    panels, used = [], []
    for off in offsets:
        i = int(np.clip(frame - 1 + off, 0, len(files) - 1))
        img = cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = 256.0 / h
        img = cv2.resize(img, (int(w * scale), 256))
        if off == 0 and obj_uv is not None:
            cv2.circle(img, (int(obj_uv[0] * scale), int(obj_uv[1] * scale)), 12, (255, 60, 60), 2)
        tag = "event" if off == 0 else (f"{off:+d}")
        cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(img, f"f{i+1} {tag}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if off == 0 else (220, 220, 220), 1)
        panels.append(img)
        used.append(i + 1)
    montage = np.concatenate(panels, axis=1)
    pil = Image.fromarray(montage)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)
    return pil, used


def extract_crop(sample_dir: Path, frame: int, center_uv: tuple[float, float] | None,
                 size: int = 420, out_path: Path | None = None):
    """Square crop around the contact (teammate-style: wrist+fingers+object in one window).
    Falls back to a centered crop when no contact point is known."""
    import cv2
    from PIL import Image

    files = _frame_files(sample_dir)
    if not files:
        return None
    i = int(np.clip(frame - 1, 0, len(files) - 1))
    img = cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    cx, cy = (int(center_uv[0]), int(center_uv[1])) if center_uv else (w // 2, h // 2)
    s = min(size, h, w)
    x1 = int(np.clip(cx - s // 2, 0, w - s))
    y1 = int(np.clip(cy - s // 2, 0, h - s))
    crop = img[y1:y1 + s, x1:x1 + s]
    pil = Image.fromarray(crop)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)
    return pil


# ---------------------------------------------------------------------------
# Audio-timing refinement
# ---------------------------------------------------------------------------

def refine_timing(x: np.ndarray, sr: int, frame: int, fps: float,
                  radius_frames: int = 4) -> tuple[int, float, float]:
    """Snap the event to the local audio-energy peak around it (audio is temporally precise).

    Returns (refined_frame, refined_time, audio_support) where audio_support ∈ [0,1] is the
    peak short-time RMS near the event relative to the clip's loudest moment.
    """
    t = (frame - 1) / fps
    c = int(round(t * sr))
    r = int(radius_frames / fps * sr)
    lo, hi = max(0, c - r), min(len(x), c + r)
    seg = x[lo:hi]
    if seg.size < 32:
        return frame, t, 0.0
    win = max(1, int(0.005 * sr))
    env = np.array([np.sqrt(np.mean(seg[i:i + win] ** 2)) for i in range(0, seg.size - win, win)])
    if env.size == 0:
        return frame, t, 0.0
    pk = int(np.argmax(env))
    peak_sample = lo + pk * win
    rt = peak_sample / sr
    clip_rms = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    support = float(np.clip(env[pk] / (clip_rms * 8.0), 0.0, 1.0))
    return int(round(rt * fps) + 1), float(rt), support


# ---------------------------------------------------------------------------
# Analysis schema + backends
# ---------------------------------------------------------------------------

@dataclass
class VLMAnalysis:
    relevant: bool          # is this an actual HOI interaction worth a loss term?
    interaction: str        # what happens (grasp / hit / place / bounce / slide / none ...)
    entities: list[str]     # involved entities (object, left_hand, ground, ...)
    impact: str             # none | light | medium | hard
    description: str        # short natural-language summary
    confidence: float
    source: str = "heuristic"
    extra: dict = field(default_factory=dict)


_IMPACT_FROM_WEIGHT = [(0.85, "hard"), (0.45, "medium"), (0.15, "light"), (-1.0, "none")]


def _impact(weight: float, audio_support: float) -> str:
    s = max(weight, audio_support)
    for thr, name in _IMPACT_FROM_WEIGHT:
        if s >= thr:
            return name
    return "none"


class Backend:
    name = "base"

    def analyze(self, montage, context: dict) -> VLMAnalysis:  # noqa: D401
        raise NotImplementedError


class HeuristicBackend(Backend):
    """No model — re-expresses the fused record + audio energy as a structured analysis.
    Lets the whole VLM pipeline run/test without a model."""
    name = "heuristic"

    def analyze(self, montage, context: dict) -> VLMAnalysis:
        et = context["fused_event_type"]
        target = context["target_entity"]
        ct = context["contact_target"]
        support = context.get("audio_support", 0.0)
        weight = context.get("manip_weight", 0.5)
        cue = context.get("visual_cue", "none")
        relevant = ct != "none" and not (et in ("none",) and support < 0.05 and cue == "none")
        entities = []
        if ct != "none":
            entities = ["object", target] if target not in ("none",) else ["object"]
        impact = _impact(weight, support)
        desc = (f"{target.replace('_',' ')} {('contacts' if ct=='part' else 'meets')} the object"
                if ct != "none" else "no clear interaction")
        return VLMAnalysis(relevant=relevant, interaction=et, entities=entities, impact=impact,
                           description=desc, confidence=context.get("fused_conf", 0.5),
                           source="heuristic", extra={"audio_support": round(support, 3)})


class MoondreamBackend(Backend):
    """Real local VLM. Disabled unless explicitly enabled (remote code / model download).
    Pass model_path to use a locally-trusted checkpoint."""
    name = "moondream"

    def __init__(self, model_path: str = "vikhyatk/moondream2", revision: str = "2024-08-26"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")

    _Q = ("You see 5 video frames around one moment (the middle is the event). "
          "Describe in one sentence the physical contact/interaction: which body part touches "
          "which object, and how hard (none/light/medium/hard). If nothing relevant happens, say 'none'.")

    def analyze(self, montage, context: dict) -> VLMAnalysis:
        enc = self.model.encode_image(montage)
        ans = self.model.answer_question(enc, self._Q, self.tok).strip()
        low = ans.lower()
        relevant = "none" not in low[:12] and "nothing" not in low
        impact = next((k for k in ("hard", "medium", "light") if k in low), "none")
        ents = [e for e in ("left_hand", "right_hand", "left_foot", "right_foot", "hand", "foot",
                            "ground", "floor", "object", "ball") if e in low]
        return VLMAnalysis(relevant=relevant, interaction=context["fused_event_type"], entities=ents,
                           impact=impact, description=ans, confidence=0.7, source="moondream")


class QwenVLBackend(Backend):
    """Official Qwen2-VL (no trust_remote_code). Fits an 8 GB GPU at 2B/fp16."""
    name = "qwen"

    _Q = ("These are 5 consecutive video frames around one moment; the middle frame (green "
          "label) is the event. Decide what physical interaction happens there. Reply with "
          "ONLY a compact JSON object and nothing else:\n"
          '{"contact": true|false, "part": "left_hand|right_hand|left_foot|right_foot|none", '
          '"object": "<one short noun>", "impact": "none|light|medium|hard", '
          '"relevant": true|false, "note": "<=8 words"}')

    def __init__(self, model_path: str = "Qwen/Qwen2-VL-2B-Instruct", prompt_mode: str = "event"):
        from transformers import AutoProcessor
        try:  # teammate-style generic loader → also works for Qwen3-VL-8B
            from transformers import AutoModelForImageTextToText as _M
        except Exception:
            from transformers import AutoModelForVision2Seq as _M
        self.prompt_mode = prompt_mode      # "event" (temporal montage) | "grasp" (fine object-side)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = _M.from_pretrained(model_path, torch_dtype="auto",
                                        device_map="cuda", trust_remote_code=True).eval()

    @staticmethod
    def _parse_json(ans: str) -> dict | None:
        import json
        import re

        m = re.search(r"\{.*\}", ans, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    @staticmethod
    def _grasp_analysis(j: dict, ans: str, context: dict) -> "VLMAnalysis":
        contact = bool(j.get("contact_visible", False))
        relevant = bool(j.get("relevant", contact))
        hand = str(j.get("hand_side", "unknown")).lower()
        part = str(j.get("object_part", "unknown")).lower()
        region = str(j.get("object_region", "")).lower()
        grasp = str(j.get("grasp_type", "unknown")).lower()
        fingers = j.get("contact_fingers", [])
        primary = str(j.get("primary_contact_finger", "")).lower()
        impact = str(j.get("impact", "none")).lower()
        if impact not in ("none", "light", "medium", "hard"):
            impact = "none"
        ents = [e for e in (hand, part) if e and e not in ("unknown", "none")]
        desc = f"{hand} {grasp} on {part}/{region} (primary {primary})"
        return VLMAnalysis(
            relevant=relevant, interaction=(grasp if grasp not in ("unknown", "not_grasp")
                                            else context["fused_event_type"]),
            entities=ents, impact=impact, description=desc[:80], confidence=float(j.get("confidence", 0.7) or 0.7),
            source="qwen_grasp",
            extra={"hand_side": hand, "object_part": part, "object_region": region,
                   "grasp_type": grasp, "contact_fingers": fingers, "primary_finger": primary})

    def analyze(self, montage, context: dict) -> VLMAnalysis:
        import torch

        prompt = GRASP_PROMPT if self.prompt_mode == "grasp" else self._Q
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[montage], padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        ans = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        j = self._parse_json(ans)
        if j is not None and self.prompt_mode == "grasp":
            return self._grasp_analysis(j, ans, context)
        if j is not None:
            contact = bool(j.get("contact", False))
            relevant = bool(j.get("relevant", contact))
            part = str(j.get("part", "none")).lower().strip()
            obj = str(j.get("object", "")).lower().strip()
            impact = str(j.get("impact", "none")).lower().strip()
            if impact not in ("none", "light", "medium", "hard"):
                impact = "none"
            ents = [e for e in (part, obj) if e and e != "none"]
            return VLMAnalysis(relevant=relevant, interaction=context["fused_event_type"],
                               entities=ents, impact=impact, description=str(j.get("note", ""))[:80],
                               confidence=0.8, source="qwen2vl",
                               extra={"contact": contact, "raw": ans[:120]})
        # fallback: unparseable → keyword heuristic on free text
        low = ans.lower()
        relevant = not low.startswith("none") and "nothing" not in low[:20]
        impact = next((k for k in ("hard", "medium", "light") if k in low), "none")
        ents = [e for e in ("left_hand", "right_hand", "left_foot", "right_foot") if e in low]
        return VLMAnalysis(relevant=relevant, interaction=context["fused_event_type"], entities=ents,
                           impact=impact, description=ans[:80], confidence=0.6, source="qwen2vl",
                           extra={"parse": "fallback"})


def get_backend(name: str, model_path: str | None = None) -> Backend:
    if name in ("heuristic", "none", None):
        return HeuristicBackend()
    if name == "moondream":
        return MoondreamBackend(model_path or "vikhyatk/moondream2")
    if name in ("qwen", "qwen2vl", "qwen2-vl"):
        return QwenVLBackend(model_path or "Qwen/Qwen2-VL-2B-Instruct", prompt_mode="event")
    if name in ("qwen_grasp", "qwen-grasp", "grasp"):   # adopted teammate fine grasp prompt
        return QwenVLBackend(model_path or "Qwen/Qwen2-VL-2B-Instruct", prompt_mode="grasp")
    raise ValueError(f"unknown VLM backend {name!r}")
