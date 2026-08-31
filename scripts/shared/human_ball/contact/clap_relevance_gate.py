#!/usr/bin/env python3
"""CLAP semantic relevance gate (SOTA import #4) — audio-text zero-shot check that an
onset actually SOUNDS like the interaction, complementing AKCA's kinematic gate.

For each part-contact row in contact_records.csv, take a +-160 ms window of audio.wav
around the (refined) event time, embed with CLAP (laion/clap-htsat-unfused), and score
similarity against positive prompts (impact/bounce/kick sounds) vs negative prompts
(speech, music, silence, wind). relevance = softmax margin pos - neg. Rows whose margin
falls below --margin are demoted (relevant=0, promote_anchor=0) — same record semantics
as the AKCA kinematic gate, so the two compose.

Usage (gvhmr env; downloads ~600 MB on first run):
  conda run -n gvhmr python scripts/shared/human_ball/contact/clap_relevance_gate.py \
      --sample-dir samples/football_10 --records-csv <in.csv> --out-csv <out.csv>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

POS_PROMPTS = ["a ball bouncing on the ground", "a ball being kicked",
               "an object impact or thud", "a basketball dribble"]
NEG_PROMPTS = ["a person speaking", "music playing", "silence or ambient room noise",
               "wind noise"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--records-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=0.16)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="demote rows with pos-neg similarity margin below this")
    args = ap.parse_args()

    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    wav_path = args.sample_dir / "audio.wav"
    audio, sr = librosa.load(str(wav_path), sr=48000, mono=True)

    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.eval()

    with torch.no_grad():
        ti = processor(text=POS_PROMPTS + NEG_PROMPTS, return_tensors="pt", padding=True)
        temb = model.get_text_features(**ti)
        temb = temb / temb.norm(dim=-1, keepdim=True)

    with args.records_csv.open() as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())

    n_pos = len(POS_PROMPTS)
    demoted = 0
    for row in rows:
        tgt = str(row.get("target_entity", "") or "")
        if tgt in ("", "none", "support"):
            continue  # only gate part-contact rows
        t = float(row.get("refined_time") or row.get("time") or 0.0)
        c = int(round(t * 48000))
        half = int(args.window_s * 48000)
        seg = audio[max(0, c - half):c + half]
        if len(seg) < 2400:  # <50 ms of audio
            continue
        with torch.no_grad():
            ai = processor(audios=[seg], sampling_rate=48000, return_tensors="pt")
            aemb = model.get_audio_features(**ai)
            aemb = aemb / aemb.norm(dim=-1, keepdim=True)
            sim = (aemb @ temb.T).squeeze(0).numpy()
        pos, neg = float(sim[:n_pos].max()), float(sim[n_pos:].max())
        margin = pos - neg
        row["clap_margin"] = f"{margin:.4f}"
        flag = ""
        if margin < args.margin:
            row["relevant"] = "0"
            row["promote_anchor"] = "0"
            demoted += 1
            flag = "  <== CLAP DEMOTED"
        print(f"  f{row['frame']:>4} {tgt:11s} pos={pos:.3f} neg={neg:.3f} "
              f"margin={margin:+.3f}{flag}", flush=True)

    if "clap_margin" not in fields:
        fields.append("clap_margin")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[clap] {args.sample_dir.name}: {demoted} row(s) demoted -> {args.out_csv}")


if __name__ == "__main__":
    main()
