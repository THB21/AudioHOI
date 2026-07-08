"""Our energy as a single combined loss curve, in the teammate's stage7 format.

Maps our geometric residuals to the teammate's E_* naming and plots ONE graph: E_total (bold)
plus each weighted component term over frames — like a training-loss decomposition, matching
`stage7_loss_residuals.csv` (E_2d, E_depth, E_contact, E_support, E_smooth, E_total). Also writes
a CSV in the same column layout so it lines up with the teammate's.

  R_center -> E_2d        R_depth -> E_depth      R_contact -> E_contact
  R_support -> E_support  R_reg -> E_smooth

Output: results/diagnostics/energy/energy_loss_curve.png + our_loss_residuals.csv
Run:    python scripts/shared/diagnostics/energy_loss_curve.py --sample-dir samples/basketball_01
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

# our term -> (teammate column, weight)
MAP = {
    "R_center": ("E_2d", 1.0),
    "R_depth": ("E_depth", 1.0),
    "R_contact": ("E_contact", 1.0),
    "R_support": ("E_support", 0.05),
    "R_reg": ("E_smooth", 1.0),
    "R_reg(trans)": ("E_smooth", 1.0),
    "R_reg(rot)": ("E_reg", 1.0),
}
COLORS = {"E_2d": "tab:blue", "E_depth": "tab:green", "E_contact": "tab:red",
          "E_support": "tab:purple", "E_smooth": "tab:orange", "E_reg": "tab:brown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    args = ap.parse_args()
    s = args.sample_dir
    traj_path, kind = ed.pick_traj(s)
    rows = ed.read_csv(traj_path)
    depth = ed.read_csv(s / "results" / "depth" / "object_depth.csv")
    fr, terms, contact, _ = ed.compute_terms(rows, depth, kind)

    # weighted components in teammate naming
    comp = {}
    for n, y in terms.items():
        if n in MAP:
            col, w = MAP[n]
            comp[col] = w * np.nan_to_num(np.asarray(y, float))
    E_total = np.sum([v for v in comp.values()], axis=0)

    # plot: E_total bold + components
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(fr, E_total, color="black", lw=2.4, label="E_total", zorder=5)
    for col, y in comp.items():
        ax.plot(fr, y, lw=1.5, color=COLORS.get(col, None), label=col, alpha=0.85)
    for f in np.asarray(fr)[contact]:
        ax.axvline(f, color="red", alpha=0.06)
    ax.set_xlabel("frame"); ax.set_ylabel("loss")
    ax.set_title(f"Energy loss curve — E_total + components — {s.name}",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", ncol=2, fontsize=9)
    out = s / "results" / "diagnostics" / "energy" / "energy_loss_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)

    # csv in teammate column order
    cols = ["frame", "time", "E_total", "E_2d", "E_depth", "E_contact", "E_support", "E_smooth", "E_reg"]
    csv_path = s / "results" / "diagnostics" / "energy" / "our_loss_residuals.csv"
    tcol = {int(r["frame"]): r.get("time", "") for r in rows}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for i, fid in enumerate(fr):
            row = {"frame": int(fid), "time": tcol.get(int(fid), ""), "E_total": f"{E_total[i]:.6f}"}
            for c in cols[3:]:
                row[c] = f"{comp[c][i]:.6f}" if c in comp else ""
            w.writerow(row)
    print("wrote", out)
    print("wrote", csv_path)
    print(f"  E_total mean={E_total.mean():.4f} max={E_total.max():.4f} | components: {list(comp)}")


if __name__ == "__main__":
    main()
