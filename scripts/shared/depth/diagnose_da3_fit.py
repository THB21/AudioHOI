"""Diagnose WHY the DA3->GVHMR affine fit fails.

For a subset of frames it re-runs DA3, samples depth at the projected SMPL-X body joints, and
reports the conditioning of the per-frame affine Z = a*D + b:
  - true joint depth spread        (ptp Z_k)  : the metric baseline we fit on
  - DA3-at-joints spread           (ptp d_k)
  - corr(d_k, Z_k)                 : is DA3 even monotonic with true depth on the body?
  - object DA3 value vs joint range: are we INTERPOLATING or EXTRAPOLATING to the object?
  - slope a and resulting object Z

Run (gvhmr env):
  python scripts/shared/depth/diagnose_da3_fit.py --sample-dir samples/football_10 --stride 16
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("da3", HERE / "run_depth_anything_v3.py")
da3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(da3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--body-model-root", type=Path,
                    default=Path("scripts/third-party/GVHMR/inputs/checkpoints/body_models"))
    ap.add_argument("--model-id", default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--stride", type=int, default=16)
    args = ap.parse_args()

    s = args.sample_dir
    results = s / "results"
    gvhmr = da3.load_gvhmr(results)
    joints = da3.build_smplx_joints(args.body_model_root, gvhmr)   # [N,22,3] cam frame
    K = np.asarray(gvhmr["K_fullimg"])                              # [N,3,3]
    masks_dir = results / "segmentation" / "masks"
    obs = da3.read_object_observations(results / "object_observations" / "object_observations.csv")
    frame_paths = sorted((s / "frames").glob("*.png"))
    frame_ids = [int(p.stem) for p in frame_paths]
    f0 = frame_ids[0]

    import torch
    from depth_anything_3.api import DepthAnything3
    print(f"loading {args.model_id} ...")
    model = DepthAnything3.from_pretrained(args.model_id).to("cuda").eval()

    print(f"\n{'frame':>5} {'Zspread':>8} {'dspread':>8} {'corr':>6} {'slope':>7} "
          f"{'objInRange':>10} {'extrapX':>8} {'objZ_m':>8}")
    rows = []
    for i in range(0, len(frame_paths), args.stride):
        fp = frame_paths[i]
        img = cv2.imread(str(fp)); H, W = img.shape[:2]
        with torch.inference_mode():
            pred = model.inference([str(fp)])
        dmap = np.asarray(pred.depth)[0].astype(np.float32)
        dmap = cv2.resize(dmap, (W, H), interpolation=cv2.INTER_LINEAR)

        j_idx = frame_ids[i] - f0
        jc = joints[j_idx]
        uv = da3.project(jc, K[j_idx])
        uu = np.round(uv[:, 0]).astype(int); vv = np.round(uv[:, 1]).astype(int)
        inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (jc[:, 2] > 1e-3)
        if inb.sum() < 6:
            continue
        d_k = dmap[vv[inb], uu[inb]].astype(np.float64)
        Z_k = jc[inb, 2].astype(np.float64)
        good = np.isfinite(d_k) & (d_k > 0)
        d_k, Z_k = d_k[good], Z_k[good]
        if d_k.size < 6 or np.ptp(d_k) < 1e-6:
            continue
        corr = float(np.corrcoef(d_k, Z_k)[0, 1])
        a, b, rms, n = da3.robust_affine_fit(d_k, Z_k)
        obj_raw, *_ = da3.sample_object_raw_depth(dmap, frame_ids[i], masks_dir, obs)
        if np.isfinite(obj_raw):
            lo, hi = float(np.min(d_k)), float(np.max(d_k))
            in_range = lo <= obj_raw <= hi
            extrap = max(0.0, lo - obj_raw, obj_raw - hi) / max(hi - lo, 1e-6)
            objZ = a * obj_raw + b
        else:
            in_range, extrap, objZ = False, np.nan, np.nan
        rows.append((np.ptp(Z_k), np.ptp(d_k), corr, a, float(in_range), extrap, objZ))
        print(f"{frame_ids[i]:>5} {np.ptp(Z_k):>8.3f} {np.ptp(d_k):>8.3f} {corr:>6.2f} {a:>7.3f} "
              f"{str(bool(in_range)):>10} {extrap:>8.2f} {objZ:>8.2f}")

    if not rows:
        print("no frames diagnosed"); return
    arr = np.array(rows, float)
    print("\n==== AGGREGATE ====")
    print(f"  joint TRUE depth spread (m):     median {np.median(arr[:,0]):.3f}   <- small = ill-conditioned baseline")
    print(f"  DA3-at-joints spread:            median {np.median(arr[:,1]):.3f}")
    print(f"  corr(DA3, trueZ) on body:        median {np.median(arr[:,2]):+.2f}   <- ~0 / negative = DA3 not monotonic here")
    print(f"  affine slope a:                  median {np.median(arr[:,3]):+.3f}")
    print(f"  object DA3 in joint range:       {int(arr[:,4].sum())}/{len(arr)} frames   <- else EXTRAPOLATING")
    print(f"  extrapolation (x joint span):    median {np.nanmedian(arr[:,5]):.2f}")
    print(f"  resulting object Z (m):          {np.nanmin(arr[:,6]):.2f} .. {np.nanmax(arr[:,6]):.2f}")


if __name__ == "__main__":
    main()
