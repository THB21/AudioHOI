"""Clean before/after view of the LLM data-reasoner fix.

Left: the flagged field before (red) vs corrected (green). Right: the LLM flags + physics
reasoning + the verifier's max|accel| before->after. Reads the artifacts the reasoner already
wrote (flags_claude.json, verify_report.json, corrected_trajectory.csv) + the original CSV.

Run: python scripts/shared/llm/build_llm_before_after.py --sample-dir samples_known_object/02_mug_v2 \
       --data-csv samples_known_object/02_mug_v2/results/mug_oriented_pose.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def col(rows, k):
    return np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--data-csv", type=Path, required=True)
    args = ap.parse_args()
    d = args.sample_dir / "results" / "diagnostics" / "llm_reasoning"

    before = list(csv.DictReader(open(args.data_csv)))
    after = list(csv.DictReader(open(d / "corrected_trajectory.csv")))
    flags = json.loads((d / "flags_claude.json").read_text())
    verify = json.loads((d / "verify_report.json").read_text())
    fields = sorted({f["field"] for f in flags["flags"]})
    field = fields[0] if fields else "yaw"

    fr = col(before, "frame")
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
    fig.suptitle(f"LLM data-reasoner — before / after  ({args.sample_dir.name}, field: {field})",
                 fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(fr, col(before, field), color="tab:red", lw=2, alpha=0.7, label="before (raw solve)")
    ax.plot(col(after, "frame"), col(after, field), color="tab:green", lw=2, label="after (LLM-flagged fix)")
    ax.set_xlabel("frame"); ax.set_ylabel(field + " (rad)" if field in ("yaw", "pitch", "roll") else field)
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    acc = verify.get("accel", {}).get(field, {})
    if acc:
        ax.set_title(f"max|accel|  {acc.get('max_acc_before','?')} -> {acc.get('max_acc_after','?')}  (reduced)",
                     fontsize=10, loc="left")

    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")
    t = "LLM flags (physics-first)\n" + "─" * 30 + "\n"
    # collapse many identical flags (same field/action/issue) into one summary line
    from collections import defaultdict
    groups = defaultdict(list)
    for f in flags["flags"]:
        groups[(f["field"], f["suggested_action"], f["issue"])].append(f["frame"])
    for (field, act, issue), frs in groups.items():
        span = f"f{min(frs)}" if len(frs) == 1 else f"f{min(frs)}–{max(frs)} ({len(frs)} frames)"
        t += f"• {field} [{act}]  {span}\n"
        t += "  " + "\n  ".join(textwrap.wrap(issue, 42)) + "\n\n"
    t += "Applied fix\n" + "─" * 30 + "\n"
    for a in verify.get("applied", []):
        t += "  " + ", ".join(f"{k}={v}" for k, v in a.items()) + "\n"
    axt.text(0, 1, t, va="top", ha="left", fontsize=8.5, family="monospace")

    out = d / "llm_before_after_summary.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
