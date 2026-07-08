"""Per-term ENERGY decomposition for a solved trajectory.

Writes out every energy term in `method_losses.md` (the ones the available data supports),
so for each example you can see WHICH constraint is violated WHERE:

  R_center  center reprojection (px)          = ||(u,v)_proj - (u,v)_obs||      [method_losses Eq.§3]
  R_depth   metric depth (m)                  = |tz - DA3_aligned|              [§2, §5]
  R_contact contact gap at contact frames (m) = |contact_depth_gap|            [§4, §5]
  R_support floor reprojection at contact (px)= |bottom_proj_v - floor_v|       [§3]
  R_reg     acceleration smoothness           = ||t_{k+1}-2t_k+t_{k-1}||        [§6]
            (translation; and rotation for 6DOF objects)

Outputs (per sample, under results/diagnostics/energy/):
  energy_terms.png      per-term residual curves over time (contact frames marked)
  trajectory_3d.png     the 3D translation markers, coloured by frame, contacts ringed
  energy_report.md      table: term -> formula -> documented weight -> RMS/max/active -> verdict

Run:  python scripts/shared/diagnostics/energy_decomposition.py --sample-dir samples/basketball_01
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# documented default weights (method_losses.md §3-4); used only to show contribution
WEIGHTS = {"R_center": 1.0, "R_depth": 1.0, "R_contact": 1.0, "R_support": 0.05, "R_reg": 0.1}


def _f(v):
    if v in (None, "", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_csv(p: Path):
    if not p or not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def col(rows, name):
    return np.array([_f(r.get(name)) if _f(r.get(name)) is not None else np.nan for r in rows], float)


def accel_norm(*chans):
    """||second difference|| across the given per-frame channels."""
    stack = np.vstack(chans)
    a = np.zeros(stack.shape[1])
    a[1:-1] = np.linalg.norm(stack[:, 2:] - 2 * stack[:, 1:-1] + stack[:, :-2], axis=0)
    return np.nan_to_num(a)


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def pick_traj(sample: Path):
    for rel in ("pose6d_sharedcam_contactphase_depthv3/ball_pose6d_sharedcam_contactphase_trajectory.csv",
                "pose6d_sharedcam_depthv3/ball_pose6d_sharedcam_trajectory.csv",
                "pose6d_sharedcam/ball_pose6d_sharedcam_trajectory.csv"):
        p = sample / "results" / rel
        if p.exists():
            return p, "ball"
    for rel in ("mug_oriented_pose.csv", "mug_handle_grip_pose.csv"):
        p = sample / "results" / rel
        if p.exists():
            return p, "mug6d"
    return None, "none"


def compute_terms(rows, depth_rows, kind):
    """Return {name: per-frame residual array}, plus contact mask."""
    fr = col(rows, "frame").astype(int)
    tx, ty, tz = col(rows, "tx"), col(rows, "ty"), col(rows, "tz")
    terms = {}
    contact = np.zeros(len(rows), bool)
    if kind == "ball":
        u_o, v_o = col(rows, "u_obs"), col(rows, "v_obs")
        u_p, v_p = col(rows, "u_proj"), col(rows, "v_proj")
        terms["R_center"] = np.nan_to_num(np.hypot(u_p - u_o, v_p - v_o))
        human = np.nan_to_num(col(rows, "contact_frame")) >= 1          # hand-ball contacts
        floor = np.nan_to_num(col(rows, "floor_contact_state")) >= 1    # ball-floor bounces
        contact = human  # the primary interaction shown in plots/3D
        gap = np.abs(np.nan_to_num(col(rows, "contact_depth_gap")))
        gap[~human] = np.nan  # human contact gap only at hand-contact frames
        terms["R_contact"] = gap
        bp, fv = col(rows, "bottom_proj_v"), col(rows, "floor_v")
        sup = np.abs(bp - fv)
        sup[~floor] = np.nan  # floor reprojection only at FLOOR-contact frames (not hand contacts)
        terms["R_support"] = sup
        terms["R_reg"] = accel_norm(tx, ty, tz)
    else:  # mug 6DOF
        terms["R_reg(trans)"] = accel_norm(tx, ty, tz)
        terms["R_reg(rot)"] = accel_norm(col(rows, "yaw"), col(rows, "pitch"), col(rows, "roll"))
    # depth term (any kind with DA3)
    if depth_rows:
        dmap = {int(float(d["frame"])): _f(d.get("object_z_aligned_m")) for d in depth_rows}
        da3 = np.array([dmap.get(int(f), np.nan) for f in fr], float)
        terms["R_depth"] = np.abs(tz - da3)
    return fr, terms, contact, (tx, ty, tz)


def plot_terms(fr, terms, contact, out: Path):
    names = list(terms)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.6 * len(names)), sharex=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ax, n in zip(axes, names):
        y = terms[n]
        ax.plot(fr, y, lw=1.4, color="tab:blue")
        if "contact" in n.lower() or "support" in n.lower():
            ax.scatter(fr[contact], y[contact], s=14, color="red", zorder=3)
        for f in fr[contact]:
            ax.axvline(f, color="red", alpha=0.08)
        unit = "px" if ("center" in n or "support" in n) else ("m" if "depth" in n or "contact" in n else "")
        ax.set_title(f"{n}   RMS={rms(y):.3g} {unit}  max={np.nanmax(y) if np.any(np.isfinite(y)) else 0:.3g}",
                     fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame  (red = contact frames)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_traj3d(fr, t3, contact, out: Path):
    tx, ty, tz = t3
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(tx, tz, -ty, c=fr, cmap="viridis", s=12)
    ax.plot(tx, tz, -ty, color="gray", lw=0.6, alpha=0.6)
    if np.any(contact):
        ax.scatter(tx[contact], tz[contact], -ty[contact], s=60, facecolors="none",
                   edgecolors="red", linewidths=1.5, label="contact")
        ax.legend()
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z depth (m)"); ax.set_zlabel("-Y up (m)")
    ax.set_title("3D object trajectory markers (colour = frame)")
    fig.colorbar(p, ax=ax, shrink=0.5, label="frame")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def verdict(name, y, contact):
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return "no data"
    r, mx = rms(y), float(np.nanmax(y))
    if name == "R_center":
        return "OK — reprojection solved exactly (≈0)" if mx < 1.0 else f"2D mismatch up to {mx:.1f}px"
    if name == "R_depth":
        return f"weak/large — |tz-DA3| RMS {r:.2f} m" if r > 0.1 else f"tight (RMS {r:.2f} m)"
    if name == "R_contact":
        return f"gap→0 at contacts (RMS {r:.3f} m over {int(np.isfinite(y).sum())} frames)"
    if name == "R_support":
        return f"floor offset RMS {r:.1f}px at contacts"
    if "R_reg" in name:
        return f"smoothness max {mx:.3g} — spikes flag jumps" if mx > 5 * (r + 1e-9) else f"smooth (max {mx:.3g})"
    return f"RMS {r:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    args = ap.parse_args()
    sample = args.sample_dir
    traj_path, kind = pick_traj(sample)
    if not traj_path:
        raise SystemExit(f"no trajectory in {sample}")
    rows = read_csv(traj_path)
    depth_rows = read_csv(sample / "results" / "depth" / "object_depth.csv")
    fr, terms, contact, t3 = compute_terms(rows, depth_rows, kind)

    out = sample / "results" / "diagnostics" / "energy"
    plot_terms(fr, terms, contact, out / "energy_terms.png")
    plot_traj3d(fr, t3, contact, out / "trajectory_3d.png")

    lines = [f"# Energy decomposition — {sample.name}", "",
             f"- trajectory: `{traj_path.relative_to(sample)}` (kind={kind})",
             f"- frames: {len(rows)} | contact frames: {int(contact.sum())} | depth: {'yes' if depth_rows else 'no'}",
             "",
             "E_total = Σ wᵢ·Rᵢ  (weights = documented defaults, method_losses.md §3-4)", "",
             "| term | meaning | weight | RMS | max | weighted RMS | verdict |",
             "|---|---|---|---|---|---|---|"]
    wsum = 0.0
    for n, y in terms.items():
        w = WEIGHTS.get(n.split("(")[0], WEIGHTS.get(n, 1.0))
        r, mx = rms(y), (float(np.nanmax(y)) if np.any(np.isfinite(y)) else 0.0)
        wr = w * r
        wsum += wr
        unit = "px" if ("center" in n or "support" in n) else ("m" if ("depth" in n or "contact" in n) else "")
        lines.append(f"| {n} | {_meaning(n)} | {w} | {r:.4g} {unit} | {mx:.4g} | {wr:.4g} | {verdict(n, y, contact)} |")
    lines += ["", f"**Σ weighted-RMS ≈ {wsum:.4g}** (rough scalar energy level; for cross-frame trend see `energy_terms.png`).",
              "", "Figures: `energy_terms.png` (per-term over time), `trajectory_3d.png` (3D markers)."]
    (out / "energy_report.md").write_text("\n".join(lines))
    print(f"{sample.name}: kind={kind} terms={list(terms)}")
    for n, y in terms.items():
        print(f"  {n:<14} RMS={rms(y):.4g}  -> {verdict(n, y, contact)}")
    print("wrote", out / "energy_report.md")


def _meaning(n):
    return {
        "R_center": "center reprojection ‖proj−obs‖",
        "R_depth": "|tz − DA3 metric depth|",
        "R_contact": "contact depth gap (active@contact)",
        "R_support": "floor reproj |bottom−floor|",
        "R_reg": "‖accel‖ translation",
        "R_reg(trans)": "‖accel‖ translation",
        "R_reg(rot)": "‖accel‖ rotation",
    }.get(n, n)


if __name__ == "__main__":
    main()
