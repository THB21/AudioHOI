"""Energy-over-time video for a talk: the scene frame on top, the live loss curve below with a
moving time cursor and the current E_total + dominant term. Ties the optimisation energy to the
motion frame-by-frame.

Output: results/renders/energy_video.mp4
Run: python scripts/shared/diagnostics/render_energy_video.py --sample-dir samples/basketball_01
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ed", HERE / "energy_decomposition.py")
ed = importlib.util.module_from_spec(spec); spec.loader.exec_module(ed)

MAP = {"R_center": ("E_2d", 1.0), "R_depth": ("E_depth", 1.0), "R_contact": ("E_contact", 1.0),
       "R_support": ("E_support", 0.05), "R_reg": ("E_smooth", 1.0),
       "R_reg(trans)": ("E_smooth", 1.0), "R_reg(rot)": ("E_reg", 1.0)}
COLORS = {"E_2d": "tab:blue", "E_depth": "tab:green", "E_contact": "tab:red",
          "E_support": "tab:purple", "E_smooth": "tab:orange", "E_reg": "tab:brown"}
FPS = 24.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--width", type=int, default=900)
    args = ap.parse_args()
    s = args.sample_dir

    traj_path, kind = ed.pick_traj(s)
    rows = ed.read_csv(traj_path)
    depth = ed.read_csv(s / "results" / "depth" / "object_depth.csv")
    fr, terms, contact, _ = ed.compute_terms(rows, depth, kind)
    fr = np.asarray(fr)
    comp = {}
    for n, y in terms.items():
        if n in MAP:
            col, w = MAP[n]
            comp[col] = w * np.nan_to_num(np.asarray(y, float))
    E_total = np.sum([v for v in comp.values()], axis=0)
    dom = np.array(list(comp))[np.argmax(np.vstack([comp[c] for c in comp]), axis=0)]

    # static loss figure (we redraw the cursor per frame)
    W = args.width
    fig = plt.figure(figsize=(W / 100, 3.6), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(fr, E_total, color="black", lw=2.2, label="E_total", zorder=5)
    for c, y in comp.items():
        ax.plot(fr, y, color=COLORS.get(c), lw=1.3, label=c, alpha=0.85)
    for f in fr[contact]:
        ax.axvline(f, color="red", alpha=0.06)
    ax.set_xlim(fr.min(), fr.max()); ax.set_ylim(0, float(E_total.max()) * 1.15 + 1e-3)
    ax.set_xlabel("frame"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_title(f"Optimisation energy — {s.name}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)

    frames_dir = s / "frames"
    fpaths = sorted(frames_dir.glob("*.png"))
    f0 = int(fpaths[0].stem)
    # size: scene scaled to width W, loss panel ~360px
    sample_img = cv2.imread(str(fpaths[0]))
    sh, sw = sample_img.shape[:2]
    scene_h = int(sh * W / sw)
    out_dir = s / "results" / "renders"; out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "energy_video.mp4"
    vw = None

    for i, fid in enumerate(fr):
        # scene frame (overlay object obs circle if available)
        fp = frames_dir / f"{int(fid):05d}.png"
        img = cv2.imread(str(fp)) if fp.exists() else np.zeros((sh, sw, 3), np.uint8)
        r = rows[i]
        try:
            uo = float(r.get("u_obs")); vo = float(r.get("v_obs"))
            cv2.circle(img, (int(uo), int(vo)), 14, (0, 255, 0), 2)
        except (TypeError, ValueError):
            pass
        img = cv2.resize(img, (W, scene_h))

        # loss panel with cursor
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        loss = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        xpix = int(ax.transData.transform((fid, 0))[0])
        yh = loss.shape[0]
        ytop = int(yh - ax.transData.transform((0, ax.get_ylim()[1]))[1])
        cv2.line(loss, (xpix, ytop), (xpix, yh - 35), (0, 0, 255), 2)
        cv2.putText(loss, f"f{int(fid)}  E_total={E_total[i]:.2f}  dom={dom[i]}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        loss = cv2.resize(loss, (W, loss.shape[0]))

        frame = np.vstack([img, loss])
        if vw is None:
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                 (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    if vw:
        vw.release()
    plt.close(fig)
    print("wrote", out, f"({len(fr)} frames)")


if __name__ == "__main__":
    main()
