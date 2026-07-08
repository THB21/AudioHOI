"""Compare OUR per-term energy residuals against the teammate's generic-pipeline loss audit.

Left column  = ours (geometric residuals R_*, from energy_decomposition on the solved trajectory)
Right column = teammate `stage7_loss_residuals.csv` (E_total decomposition from the
               generic_pipeline_v2_llm_vlm_gate branch).

They are different formulations on different solves, so this is a qualitative comparison of which
terms carry the energy in each method (ours: depth/contact-driven; theirs: prior/reg-dominated).

Output: results/diagnostics/energy/energy_vs_teammate.png

Run: python scripts/shared/diagnostics/energy_compare.py --sample-dir samples/basketball_01 \
       --teammate-csv <stage7_loss_residuals.csv>
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ed", HERE / "energy_decomposition.py")
ed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ed)

TEAMMATE_TERMS = ["E_total", "E_2d", "E_depth", "E_contact", "E_support", "E_smooth", "E_reg", "E_prior"]


def read(p: Path):
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(r, k):
    v = r.get(k)
    try:
        return float(v) if v not in (None, "", "nan") else np.nan
    except ValueError:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--teammate-csv", type=Path, required=True)
    args = ap.parse_args()
    s = args.sample_dir

    # ours
    traj_path, kind = ed.pick_traj(s)
    rows = ed.read_csv(traj_path)
    depth = ed.read_csv(s / "results" / "depth" / "object_depth.csv")
    fr, terms, contact, _ = ed.compute_terms(rows, depth, kind)
    ours = [(n, y) for n, y in terms.items()]

    # teammate
    tm = read(args.teammate_csv)
    tfr = [int(fnum(r, "frame")) for r in tm]
    tm_terms = []
    for c in TEAMMATE_TERMS:
        y = np.array([fnum(r, c) for r in tm])
        if np.any(np.isfinite(y)) and np.nanmax(np.abs(y)) > 0:
            tm_terms.append((c, y))

    nrows = max(len(ours), len(tm_terms))
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 1.5 * nrows), squeeze=False)
    fig.suptitle(f"Energy / loss terms — {s.name}:  OURS (R_*)  vs  teammate generic-pipeline (E_*)",
                 fontsize=13, fontweight="bold")
    for i in range(nrows):
        # ours (left)
        ax = axes[i][0]
        if i < len(ours):
            n, y = ours[i]
            ax.plot(fr, y, color="tab:blue", lw=1.3)
            ax.set_title(f"OURS  {n}   (RMS {ed.rms(y):.3g})", fontsize=9, loc="left")
            ax.grid(alpha=0.3)
        else:
            ax.axis("off")
        # teammate (right)
        ax = axes[i][1]
        if i < len(tm_terms):
            n, y = tm_terms[i]
            ax.plot(tfr, y, color="tab:orange", lw=1.3)
            ax.set_title(f"TEAMMATE  {n}   (mean {np.nanmean(y):.3g}, max {np.nanmax(y):.3g})",
                         fontsize=9, loc="left")
            ax.grid(alpha=0.3)
        else:
            ax.axis("off")
    axes[-1][0].set_xlabel("frame"); axes[-1][1].set_xlabel("frame")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = s / "results" / "diagnostics" / "energy" / "energy_vs_teammate.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)
    print(f"  ours: {[n for n,_ in ours]}")
    print(f"  teammate: {[n for n,_ in tm_terms]}")


if __name__ == "__main__":
    main()
