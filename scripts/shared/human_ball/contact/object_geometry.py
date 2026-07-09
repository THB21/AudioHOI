"""Object geometry abstraction shared by body-side contact refinement and HOI metrics.

The ball pipeline modeled the object as a per-frame sphere (center + radius). Yixin's
generic SE3 mainline emits full poses (tx,ty,tz,qw,qx,qy,qz) for line objects (stick)
and rigid objects (mug/chair). This module maps a trajectory CSV onto a differentiable
"distance to object surface skeleton" so the same hinge/touch losses work for:

  sphere   skeleton = center point,        surface at `radius`
  capsule  skeleton = axis segment (SE3-rotated local axis, length L), surface at `radius`

All tensors are torch; distances are differentiable w.r.t. the query points (the body
joints), the object pose is fixed input.
"""
from __future__ import annotations

import numpy as np
import torch

_AXES = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}


def quat_to_rotmat(qwxyz: np.ndarray) -> np.ndarray:
    """[N,4] (w,x,y,z) -> [N,3,3]."""
    q = qwxyz / np.linalg.norm(qwxyz, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


class SphereGeometry:
    """Per-frame centers [N,3]; radius [N] (surface radius, from radius_m column)."""

    kind = "sphere"

    def __init__(self, centers: torch.Tensor, radius: torch.Tensor):
        self.centers = centers
        self.radius = radius
        self.valid = ~torch.isnan(centers[:, 0])

    def point_distances(self, points: torch.Tensor) -> torch.Tensor:
        """points [N,P,3] -> distance to skeleton [N,P]."""
        return torch.linalg.norm(points - self.centers[:, None, :], dim=2)


class CapsuleGeometry:
    """Per-frame axis segment from SE3 pose; constant surface radius.

    e1/e2 [N,3] world endpoints = t ± R @ axis_local * length/2.
    For the stick: axis_local='x' (verified against line_correspondence reprojection),
    length 1.86 m (case config), radius 0.018 m (URDF main cylinder).
    """

    kind = "capsule"

    def __init__(self, centers: np.ndarray, quats_wxyz: np.ndarray, *,
                 length_m: float, radius_m: float, axis_local: str = "x",
                 device: str = "cpu"):
        rot = quat_to_rotmat(quats_wxyz)
        axis_world = rot @ np.asarray(_AXES[axis_local], dtype=np.float64)
        half = axis_world * (length_m / 2.0)
        self.e1 = torch.tensor(centers - half, dtype=torch.float32, device=device)
        self.e2 = torch.tensor(centers + half, dtype=torch.float32, device=device)
        n = len(centers)
        self.centers = torch.tensor(centers, dtype=torch.float32, device=device)
        self.radius = torch.full((n,), float(radius_m), device=device)
        self.valid = ~torch.isnan(self.centers[:, 0])

    def point_distances(self, points: torch.Tensor) -> torch.Tensor:
        """points [N,P,3] -> distance to axis segment [N,P]."""
        d = (self.e2 - self.e1)[:, None, :]                      # [N,1,3]
        v = points - self.e1[:, None, :]                         # [N,P,3]
        denom = (d * d).sum(dim=2).clamp_min(1e-9)               # [N,1]
        t = ((v * d).sum(dim=2) / denom).clamp(0.0, 1.0)         # [N,P]
        closest = self.e1[:, None, :] + t[:, :, None] * d        # [N,P,3]
        return torch.linalg.norm(points - closest, dim=2)


class MeshGeometry:
    """Mesh-SDF geometry (METRICS ONLY — trimesh queries are not differentiable).

    One static mesh in the object/URDF frame + per-frame SE3; query points are
    inverse-transformed into the mesh frame. point_distances returns SIGNED distance
    to the surface (negative inside), and radius is 0 so downstream
    penetration = (radius - d)+ and gap = |d - radius| formulas hold unchanged.
    """

    kind = "mesh"

    def __init__(self, centers: np.ndarray, quats_wxyz: np.ndarray, mesh_path: str,
                 device: str = "cpu", scales: np.ndarray | None = None):
        import trimesh
        self._mesh = trimesh.load(mesh_path, force="mesh")
        self._pq = trimesh.proximity.ProximityQuery(self._mesh)
        self._R = quat_to_rotmat(quats_wxyz)
        self._t = centers
        n = len(centers)
        # per-frame uniform mesh scale (mesh is the unit/base object; pose scale maps
        # it to metric size). SDF of a uniformly-scaled mesh: d_s(p) = s · d_1(p/s).
        self._s = np.ones(n) if scales is None else np.asarray(scales, dtype=float)
        self.centers = torch.tensor(centers, dtype=torch.float32, device=device)
        self.radius = torch.zeros(n, device=device)
        self.valid = ~self.centers[:, 0].isnan()
        self._device = device

    def point_distances(self, points: torch.Tensor) -> torch.Tensor:
        p = points.detach().cpu().numpy()                       # [N,P,3] camera frame
        n, P, _ = p.shape
        local = np.einsum("nij,npj->npi", self._R.transpose(0, 2, 1), p - self._t[:, None, :])
        local = local / self._s[:, None, None]                  # into unit mesh frame
        sd = self._pq.signed_distance(local.reshape(-1, 3))     # + inside (trimesh)
        dist = -sd.reshape(n, P) * self._s[:, None]             # + outside, - inside (metric)
        return torch.tensor(dist, dtype=torch.float32, device=self._device)


class SDFGridGeometry:
    """Differentiable mesh geometry for Stage C: a precomputed voxel SDF grid
    (trimesh signed distance, one-off ~1-2 min) queried with trilinear interpolation
    (torch grid_sample), so the penetration/touch hinges get gradients w.r.t. the
    body joints. Sign convention like MeshGeometry: + outside, − inside; radius 0.

    EasyHOI-style surface-level upgrade of the center/skeleton-distance hinge
    (docs/research_grasp_placement_sota.md import #3 / #1 target-map slot).
    """

    kind = "mesh_sdf"

    def __init__(self, centers: np.ndarray, quats_wxyz: np.ndarray, mesh_path: str,
                 device: str = "cpu", res: int = 48, pad_m: float = 0.15,
                 scales: np.ndarray | None = None):
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        pq = trimesh.proximity.ProximityQuery(mesh)
        lo = mesh.bounds[0] - pad_m
        hi = mesh.bounds[1] + pad_m
        xs = [np.linspace(lo[k], hi[k], res) for k in range(3)]
        X, Y, Z = np.meshgrid(*xs, indexing="ij")
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        sd = np.empty(len(pts))
        for s in range(0, len(pts), 20000):
            sd[s:s + 20000] = pq.signed_distance(pts[s:s + 20000])
        dist = (-sd).reshape(res, res, res)                      # + outside
        # grid_sample layout: input [1,1,D,H,W] indexed (z,y,x); coords (x,y,z)
        self._sdf = torch.tensor(dist.transpose(2, 1, 0), dtype=torch.float32,
                                 device=device)[None, None]
        self._lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self._hi = torch.tensor(hi, dtype=torch.float32, device=device)
        self._R = torch.tensor(quat_to_rotmat(quats_wxyz), dtype=torch.float32, device=device)
        self._t = torch.tensor(centers, dtype=torch.float32, device=device)
        n = len(centers)
        s = np.ones(n) if scales is None else np.asarray(scales, dtype=float)
        self._s = torch.tensor(s, dtype=torch.float32, device=device)   # per-frame uniform scale
        self.centers = self._t
        self.radius = torch.zeros(n, device=device)
        self.valid = ~self.centers[:, 0].isnan()

    def point_distances(self, points: torch.Tensor) -> torch.Tensor:
        local = torch.einsum("nji,npj->npi", self._R, points - self._t[:, None, :])
        local = local / self._s[:, None, None]                       # into unit mesh frame
        g = (local - self._lo) / (self._hi - self._lo) * 2.0 - 1.0   # [-1,1] (x,y,z)
        n, P, _ = g.shape
        out = torch.nn.functional.grid_sample(
            self._sdf, g.view(1, n, P, 1, 3), mode="bilinear",
            padding_mode="border", align_corners=True)
        return out.view(n, P) * self._s[:, None]                     # unit → metric distance


def geometry_from_rows(rows: list[dict], n: int, *, object_type: str = "sphere",
                       length_m: float | None = None, radius_m: float | None = None,
                       axis_local: str = "x", mesh_path: str | None = None,
                       device: str = "cpu"):
    """Build geometry from trajectory CSV rows (1-based `frame` column).

    sphere: uses tx/ty/tz + radius_m column (radius_m arg overrides).
    capsule: needs qw..qz columns + explicit length_m; radius_m defaults to 0.018.
    mesh: needs qw..qz columns + mesh_path (URDF-frame GLB, e.g. bake_urdf_to_glb
          --keep-origin output). Metrics only — not differentiable.
    """
    centers = np.full((n, 3), np.nan)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    rad = np.zeros(n)
    scales = np.ones(n)                                    # per-frame uniform mesh scale
    for r in rows:
        i = int(r["frame"]) - 1
        if not (0 <= i < n):
            continue
        centers[i] = [float(r["tx"]), float(r["ty"]), float(r["tz"])]
        rad[i] = float(r.get("radius_m", 0.0) or 0.0)
        if "qw" in r and r.get("qw") not in ("", None):
            quats[i] = [float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]
        if r.get("scale") not in ("", None):
            scales[i] = float(r["scale"])
    if object_type == "sphere":
        centers_t = torch.tensor(centers, dtype=torch.float32, device=device)
        if radius_m is not None:
            rad[:] = radius_m
        return SphereGeometry(centers_t, torch.tensor(rad, dtype=torch.float32, device=device))
    if object_type == "capsule":
        if length_m is None:
            raise ValueError("capsule geometry needs --object-length-m")
        return CapsuleGeometry(centers, quats, length_m=length_m,
                               radius_m=radius_m if radius_m is not None else 0.018,
                               axis_local=axis_local, device=device)
    if object_type == "mesh":
        if mesh_path is None:
            raise ValueError("mesh geometry needs --object-mesh-path")
        return MeshGeometry(centers, quats, mesh_path, device=device, scales=scales)
    if object_type == "mesh_sdf":
        if mesh_path is None:
            raise ValueError("mesh_sdf geometry needs --object-mesh-path")
        return SDFGridGeometry(centers, quats, mesh_path, device=device, scales=scales)
    raise ValueError(f"unknown object_type {object_type!r}")
