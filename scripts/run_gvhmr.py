#!/usr/bin/env python3
"""Run GVHMR on a sample video and save SMPL-X predictions.

Layout:
    AudioHOI/
    ├── scripts/
    │   ├── run_gvhmr.py
    │   └── third-party/GVHMR/
    └── samples/<name>/
        ├── video.mp4
        └── results/gvhmr/
            ├── preprocess/
            │   ├── bbx.pt
            │   ├── vitpose.pt
            │   ├── vit_features.pt
            │   └── slam.pt
            ├── result.pt
            └── result.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
GVHMR_ROOT = (_SCRIPT_DIR / "third-party" / "GVHMR").resolve()
assert GVHMR_ROOT.exists(), f"GVHMR not found at {GVHMR_ROOT}"

_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat  # type: ignore

sys.path.insert(0, str(GVHMR_ROOT))

import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.geo.hmr_cam import (
    get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor,
)
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import detach_to_cpu
from hmr4d.utils.pylogger import Log
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL


def build_cfg(
    sample_dir: Path,
    static_cam: bool,
    f_mm: int | None,
    person: int,
    fps: int,
):
    register_store_gvhmr()
    GlobalHydra.instance().clear()

    cfg_dir = (GVHMR_ROOT / "hmr4d" / "configs").resolve()
    video_path = sample_dir / "video.mp4"
    assert video_path.exists(), f"No video at {video_path}"

    results_dir = sample_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        overrides = [
            f"video_name=gvhmr",
            f"output_root={results_dir}",
            f"static_cam={static_cam}",
            f"verbose=False",
            f"person={person}",
            f"fps={fps}",
            f"use_dpvo=False",
            f"ignore_render=True",
            f"+video_dir={sample_dir}",
            f"video_path={video_path}",
        ]
        if f_mm is not None:
            overrides.append(f"f_mm={f_mm}")
        cfg = compose(config_name="demo", overrides=overrides)

    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def clear_preprocess_cache(cfg) -> None:
    paths = cfg.paths
    targets = [paths.bbx, paths.vitpose, paths.vit_features]
    if not cfg.static_cam:
        targets.append(paths.slam)
    for p in targets:
        if Path(p).exists():
            Path(p).unlink()
            Log.info(f"[Preprocess] Removed cache: {p}")


@torch.no_grad()
def run_preprocess(cfg, clear_cache: bool = False) -> None:
    Log.info("[Preprocess] Start")
    if clear_cache:
        clear_preprocess_cache(cfg)

    paths = cfg.paths
    video_path = cfg.video_path
    person = cfg.person

    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path, person).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    bbx_xys = torch.load(paths.bbx)["bbx_xys"]

    if not Path(paths.vitpose).exists():
        extractor = VitPoseExtractor()
        vitpose = extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del extractor

    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor

    if not cfg.static_cam and not Path(paths.slam).exists():
        vo = SimpleVO(video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
        torch.save(vo.compute(), paths.slam)

    Log.info("[Preprocess] Done")


def load_data_dict(cfg) -> dict:
    length, width, height = get_video_lwh(cfg.video_path)

    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if not isinstance(traj, torch.Tensor):
            traj = torch.from_numpy(traj)
        R_w2c = traj[:, :3, :3].clone()

    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    return {
        "length":     torch.tensor(length),
        "bbx_xys":    torch.load(cfg.paths.bbx)["bbx_xys"],
        "kp2d":       torch.load(cfg.paths.vitpose),
        "K_fullimg":  K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq":   torch.load(cfg.paths.vit_features),
    }


def save_outputs(pred: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    result_pt = out_dir / "result.pt"
    torch.save(pred, result_pt)
    Log.info(f"[HMR4D] Saved → {result_pt}")

    pred_np = {
        "smpl_params_global": {
            k: v.cpu().numpy() for k, v in pred["smpl_params_global"].items()
        },
        "smpl_params_incam": {
            k: v.cpu().numpy() for k, v in pred["smpl_params_incam"].items()
        },
        "K_fullimg": pred["K_fullimg"].cpu().numpy(),
    }
    result_pkl = out_dir / "result.pkl"
    with open(result_pkl, "wb") as f:
        pickle.dump(pred_np, f, protocol=pickle.HIGHEST_PROTOCOL)
    Log.info(f"[HMR4D] Saved → {result_pkl}")

    return result_pt, result_pkl


def run_gvhmr(
    sample_dir: Path,
    static_cam: bool = False,
    f_mm: int | None = None,
    clear_cache: bool = False,
    person: int = 0,
    fps: int = 30,
) -> dict:
    os.chdir(str(GVHMR_ROOT))

    cfg = build_cfg(
        sample_dir=sample_dir,
        static_cam=static_cam,
        f_mm=f_mm,
        person=person,
        fps=fps,
    )
    run_preprocess(cfg, clear_cache=clear_cache)
    data = load_data_dict(cfg)

    Log.info("[HMR4D] Predicting")
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()

    tic = Log.sync_time()
    pred = model.predict(data, static_cam=cfg.static_cam)
    pred = detach_to_cpu(pred)
    Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s")

    save_outputs(pred, sample_dir / "results" / "gvhmr")
    return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GVHMR on a sample video.")
    parser.add_argument("--sample-dir", type=Path, required=True,
                        help="Path to sample folder containing video.mp4")
    parser.add_argument("--static-cam", action="store_true",
                        help="Skip visual odometry (use for tripod-shot clips)")
    parser.add_argument("--f-mm", type=int, default=None,
                        help="Focal length in mm (default: estimated from image size)")
    parser.add_argument("--person", type=int, default=0,
                        help="Person index to track when multiple people are detected")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frame rate")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Force re-run of all preprocessing steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = _SCRIPT_DIR.parent
    sample_dir = args.sample_dir
    if not sample_dir.is_absolute():
        sample_dir = (project_root / sample_dir).resolve()
    assert sample_dir.exists(), f"Sample dir not found: {sample_dir}"

    pred = run_gvhmr(
        sample_dir,
        static_cam=args.static_cam,
        f_mm=args.f_mm,
        clear_cache=args.clear_cache,
        person=args.person,
        fps=args.fps,
    )

    g = pred["smpl_params_global"]
    print()
    print(f"sample:        {sample_dir.name}")
    print(f"frames:        {g['body_pose'].shape[0]}")
    print(f"body_pose:     {tuple(g['body_pose'].shape)}")
    print(f"global_orient: {tuple(g['global_orient'].shape)}")
    print(f"transl:        {tuple(g['transl'].shape)}")
    print(f"betas:         {tuple(g['betas'].shape)}")
    print(f"K_fullimg:     {tuple(pred['K_fullimg'].shape)}")
    print(f"output:        {sample_dir / 'results' / 'gvhmr'}")


if __name__ == "__main__":
    main()
