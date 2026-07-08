"""Combined per-video source dashboard: every source in ONE image, for side-by-side comparison.

Tiles a representative frame from each per-source overlay (mask / cotracker / hamer) plus the
source plots (depth / audio) and a verdict panel parsed from SOURCES.md, so all input sources
for one video can be compared against each other at a glance.

Needs analyze_sources.py to have been run first (per_source/* + SOURCES.md).
Output: results/diagnostics/source_dashboard.png

Run:  python scripts/shared/diagnostics/build_source_dashboard.py --sample-dir samples/basketball_01
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mid_frame(mp4: Path):
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, fr = cap.read()
    cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None


def parse_verdicts(sources_md: Path) -> dict[str, str]:
    """Pull 'source -> verdict' from the SOURCES.md markdown table."""
    out = {}
    if not sources_md.exists():
        return out
    for line in sources_md.read_text().splitlines():
        if not line.startswith("|") or "verdict" in line.lower() or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 5:
            name, present, art, ok, verd = cells[0], cells[1], cells[2], cells[3], cells[4]
            if name and name != "source":
                out[name] = f"{ok} {verd}" if present != "—" else "absent"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    args = ap.parse_args()
    s = args.sample_dir
    V = s / "results" / "diagnostics" / "per_source"
    verds = parse_verdicts(s / "results" / "diagnostics" / "SOURCES.md")

    # ordered panels: (title, kind, path)
    panels = [
        ("SAM2 mask", "video", V / "mask.mp4"),
        ("CoTracker", "video", V / "cotracker.mp4"),
        ("HaMeR hands", "video", V / "hamer.mp4"),
        ("DA3 depth", "image", V / "depth.png"),
        ("Audio events", "image", V / "audio.png"),
    ]

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"Source comparison dashboard — {s.name}", fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.12)
    cells = [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1]]
    for (title, kind, path), cell in zip(panels, cells):
        ax = fig.add_subplot(cell)
        img = None
        if path.exists():
            img = mid_frame(path) if kind == "video" else plt.imread(str(path))
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "absent", ha="center", va="center", color="gray", fontsize=14)
        v = verds.get(title, "")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel(_wrap(v, 60), fontsize=8, color=("green" if v.startswith("✓") else
                                                       "red" if v.startswith("✗") else "gray"))
        ax.set_xticks([]); ax.set_yticks([])

    # verdict summary panel
    axv = fig.add_subplot(gs[1, 2]); axv.axis("off")
    txt = "Source verdicts\n" + "─" * 22 + "\n"
    for name in ("SAM2 mask", "CoTracker", "DA3 depth", "Audio events", "GVHMR body", "HaMeR hands", "Object 3D traj"):
        txt += f"{name}:\n  {_wrap(verds.get(name,'—'), 38)}\n"
    axv.text(0, 1, txt, va="top", ha="left", fontsize=8, family="monospace")

    out = s / "results" / "diagnostics" / "source_dashboard.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _wrap(t, n):
    words, lines, cur = t.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
