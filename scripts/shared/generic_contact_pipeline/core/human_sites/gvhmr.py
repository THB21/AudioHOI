from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ..contact_constraints.types import HumanSite
from ..measurements.types import CoordinateFrame, SourceRef
from .types import HumanSiteMeasurement


SITE_JOINT_IDS = {
    ("hand", "left"): (20, 25, 28, 31, 34),
    ("hand", "right"): (21, 40, 43, 46, 49),
    ("foot", "left"): (10,),
    ("foot", "right"): (11,),
}


@dataclass(frozen=True)
class GVHMRSiteExtractionResult:
    schema: str
    measurements: tuple[HumanSiteMeasurement, ...]
    source_artifact: str
    source_sha256: str
    body_model_artifact: str
    body_model_sha256: str
    read_only: bool = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_joints(result_pkl: Path, body_models_root: Path) -> np.ndarray:
    try:
        import smplx
        import torch
    except ImportError as exc:  # pragma: no cover - depends on declared audiohoi runtime
        raise RuntimeError("GVHMR site extraction requires the audiohoi runtime with smplx and torch") from exc

    body_model = body_models_root / "smplx" / "SMPLX_NEUTRAL.npz"
    if not result_pkl.is_file():
        raise FileNotFoundError(f"missing GVHMR result: {result_pkl}")
    if not body_model.is_file():
        raise FileNotFoundError(f"missing SMPL-X body model: {body_model}")
    with result_pkl.open("rb") as handle:
        data = pickle.load(handle)
    params = data["smpl_params_incam"]
    arrays = {
        key: np.asarray(params[key], dtype=np.float32)
        for key in ("body_pose", "betas", "global_orient", "transl")
    }
    frame_count = int(arrays["transl"].shape[0])
    model = smplx.create(
        str(body_models_root),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=frame_count,
    )
    with torch.inference_mode():
        output = model(
            body_pose=torch.from_numpy(arrays["body_pose"]),
            betas=torch.from_numpy(arrays["betas"]),
            global_orient=torch.from_numpy(arrays["global_orient"]),
            transl=torch.from_numpy(arrays["transl"]),
            return_verts=False,
        )
    joints = output.joints.detach().cpu().numpy().astype(np.float64)
    if joints.ndim != 3 or joints.shape[0] != frame_count or joints.shape[2] != 3:
        raise ValueError("GVHMR SMPL-X forward pass returned an invalid joint trajectory")
    return joints


def extract_gvhmr_site_measurements(
    *,
    sample_id: str,
    result_pkl: Path,
    body_models_root: Path,
    frame_times: Mapping[int, float],
) -> GVHMRSiteExtractionResult:
    """Read GVHMR once and expose fixed skeleton sites for object factors."""

    joints = _load_joints(result_pkl, body_models_root)
    measurements: list[HumanSiteMeasurement] = []
    for frame in sorted(frame_times):
        index = frame - 1
        if index < 0 or index >= len(joints):
            continue
        for (body_part, side), joint_ids in SITE_JOINT_IDS.items():
            xyz = tuple(float(value) for value in joints[index, list(joint_ids), :].mean(axis=0))
            site_id = f"{side}_{body_part}"
            measurements.append(
                HumanSiteMeasurement(
                    measurement_id=f"{sample_id}:{frame}:gvhmr_site:{site_id}",
                    sample_id=sample_id,
                    frame=frame,
                    time=float(frame_times[frame]),
                    site=HumanSite(body_part, side),
                    xyz_m=xyz,
                    coordinate_frame=CoordinateFrame.CAMERA_METERS,
                    confidence=1.0,
                    source=SourceRef(
                        str(result_pkl),
                        ("smpl_params_incam", "body_pose", "betas", "global_orient", "transl"),
                        producer="gvhmr_read_only_skeleton_site_adapter",
                    ),
                )
            )
    if not measurements:
        raise ValueError("GVHMR site extraction produced no frame-aligned measurements")
    return GVHMRSiteExtractionResult(
        schema="gvhmr_skeleton_site_xyz_v1",
        measurements=tuple(measurements),
        source_artifact=str(result_pkl),
        source_sha256=_sha256(result_pkl),
        body_model_artifact=str(body_models_root / "smplx" / "SMPLX_NEUTRAL.npz"),
        body_model_sha256=_sha256(body_models_root / "smplx" / "SMPLX_NEUTRAL.npz"),
        read_only=True,
    )
