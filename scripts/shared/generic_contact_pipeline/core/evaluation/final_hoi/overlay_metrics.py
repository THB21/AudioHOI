from __future__ import annotations

import re
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .schemas import EvaluationPaths, MetricBlock
from .utils import f, mean, read_rows, write_json, write_rows


Mask = list[list[bool]]
DEFAULT_CAMERA = {"fx": 1468.604736328125, "fy": 1468.604736328125, "cx": 640.0, "cy": 360.0}


def _frame_number(path: Path) -> int | None:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _collect_masks(dirs: list[Path]) -> dict[int, Path]:
    masks: dict[int, Path] = {}
    for directory in dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {".pgm", ".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            frame = _frame_number(path)
            if frame is not None and frame not in masks:
                masks[frame] = path
    return masks


def _write_pgm(path: Path, mask: Mask) -> None:
    height = len(mask)
    width = len(mask[0]) if height else 0
    payload = bytes(255 if mask[y][x] else 0 for y in range(height) for x in range(width))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + payload)


def _pgm_tokens(data: bytes) -> tuple[list[bytes], int]:
    tokens: list[bytes] = []
    i = 0
    n = len(data)
    while i < n and len(tokens) < 4:
        while i < n and data[i] in b" \t\r\n":
            i += 1
        if i < n and data[i] == ord("#"):
            while i < n and data[i] not in b"\r\n":
                i += 1
            continue
        start = i
        while i < n and data[i] not in b" \t\r\n":
            i += 1
        if start < i:
            tokens.append(data[start:i])
    while i < n and data[i] in b" \t\r\n":
        i += 1
    return tokens, i


def _read_pgm(path: Path) -> Mask | None:
    data = path.read_bytes()
    tokens, offset = _pgm_tokens(data)
    if len(tokens) < 4 or tokens[0] not in {b"P5", b"P2"}:
        return None
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = max(1, int(tokens[3]))
    if tokens[0] == b"P5":
        payload = data[offset : offset + width * height]
        if len(payload) < width * height:
            return None
        return [[payload[y * width + x] > max_value * 0.5 for x in range(width)] for y in range(height)]
    values = [int(token) for token in data[offset:].split()]
    if len(values) < width * height:
        return None
    return [[values[y * width + x] > max_value * 0.5 for x in range(width)] for y in range(height)]


def _read_image_mask(path: Path) -> Mask | None:
    if path.suffix.lower() == ".pgm":
        return _read_pgm(path)
    try:
        from PIL import Image  # type: ignore

        image = Image.open(path).convert("L")
        width, height = image.size
        pixels = list(image.getdata())
        return [[pixels[y * width + x] > 127 for x in range(width)] for y in range(height)]
    except Exception:
        return None


def _read_mask_array(path: Path):
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        image = Image.open(path).convert("L")
        return np.asarray(image) > 127
    except Exception:
        return None


def _write_mask_array_pgm(path: Path, mask) -> None:
    try:
        import numpy as np  # type: ignore

        arr = (np.asarray(mask, dtype=np.uint8) * 255)
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = arr.shape
        path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + arr.tobytes())
    except Exception:
        _write_pgm(path, [[bool(v) for v in row] for row in mask])


def _parse_floats(text: str | None, default: tuple[float, ...]) -> list[float]:
    if not text:
        return list(default)
    try:
        values = [float(v) for v in text.split()]
        return values if values else list(default)
    except Exception:
        return list(default)


def _rpy_matrix(rpy: list[float]):
    import numpy as np  # type: ignore

    rx, ry, rz = rpy
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return mz @ my @ mx


def _quat_matrix(row: dict[str, str]):
    import numpy as np  # type: ignore

    qw = f(row.get("qw"), 1.0) or 1.0
    qx = f(row.get("qx"), 0.0) or 0.0
    qy = f(row.get("qy"), 0.0) or 0.0
    qz = f(row.get("qz"), 0.0) or 0.0
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-8:
        return np.eye(3)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _box_vertices_faces(size: list[float]):
    import numpy as np  # type: ignore

    sx, sy, sz = [v * 0.5 for v in size[:3]]
    vertices = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    return vertices, faces


def _cylinder_vertices_faces(radius: float, length: float, sections: int = 24):
    import numpy as np  # type: ignore

    angles = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
    bottom = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(sections, -length * 0.5)])
    top = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(sections, length * 0.5)])
    vertices = np.vstack([bottom, top, [[0, 0, -length * 0.5], [0, 0, length * 0.5]]])
    faces = []
    b_center = 2 * sections
    t_center = 2 * sections + 1
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([i, j, sections + j])
        faces.append([i, sections + j, sections + i])
        faces.append([b_center, j, i])
        faces.append([t_center, sections + i, sections + j])
    return vertices, np.asarray(faces, dtype=int)


def _sphere_vertices_faces(radius: float):
    try:
        import trimesh  # type: ignore

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        return mesh.vertices, mesh.faces
    except Exception:
        return _box_vertices_faces([radius * 2, radius * 2, radius * 2])


def _load_mesh_vertices_faces(path: Path):
    try:
        import numpy as np  # type: ignore
        import trimesh  # type: ignore

        mesh = trimesh.load_mesh(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int)
    except Exception:
        return None


def _apply_origin(vertices, origin_el: ET.Element | None):
    import numpy as np  # type: ignore

    xyz = _parse_floats(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0))
    rpy = _parse_floats(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0))
    return vertices @ _rpy_matrix(rpy).T + np.asarray(xyz[:3], dtype=float)


def _resolve_urdf(paths: EvaluationPaths) -> Path | None:
    candidates = [
        paths.sample_dir / "articraft" / "model.urdf",
        paths.sample_dir / "articraft" / "materialized_mug_mesh" / "model.urdf",
        paths.sample_dir / "articraft" / "generated_record" / "rec_fork-6911e934-oldtopology-rearup-stretcherlevel-pass_20260619_1056" / "model.urdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    globbed = sorted(paths.sample_dir.glob("articraft/**/model.urdf"), key=lambda p: (0 if "materialized" in str(p) else 1, len(str(p))))
    return globbed[0] if globbed else None


def _load_urdf_visuals(urdf: Path):
    import numpy as np  # type: ignore

    visuals = []
    robot = ET.parse(urdf).getroot()
    for visual in robot.findall(".//visual"):
        geom = visual.find("geometry")
        if geom is None:
            continue
        vertices_faces = None
        box = geom.find("box")
        cylinder = geom.find("cylinder")
        sphere = geom.find("sphere")
        mesh_el = geom.find("mesh")
        if box is not None:
            vertices_faces = _box_vertices_faces(_parse_floats(box.get("size"), (1.0, 1.0, 1.0)))
        elif cylinder is not None:
            vertices_faces = _cylinder_vertices_faces(float(cylinder.get("radius", "0.01")), float(cylinder.get("length", "0.1")))
        elif sphere is not None:
            vertices_faces = _sphere_vertices_faces(float(sphere.get("radius", "0.01")))
        elif mesh_el is not None and mesh_el.get("filename"):
            mesh_path = Path(str(mesh_el.get("filename")))
            if not mesh_path.is_absolute():
                mesh_path = urdf.parent / mesh_path
            vertices_faces = _load_mesh_vertices_faces(mesh_path)
        if vertices_faces is None:
            continue
        vertices, faces = vertices_faces
        visuals.append((_apply_origin(np.asarray(vertices, dtype=float), visual.find("origin")), np.asarray(faces, dtype=int)))
    return visuals


def _render_urdf_mask_array(paths: EvaluationPaths, row: dict[str, str], observed_shape, visuals_cache: dict[str, object]):
    try:
        import numpy as np  # type: ignore
        from PIL import Image, ImageDraw  # type: ignore

        urdf = visuals_cache.get("urdf")
        visuals = visuals_cache.get("visuals")
        if urdf is None:
            urdf = _resolve_urdf(paths)
            visuals_cache["urdf"] = urdf
        if urdf is None:
            return None
        if visuals is None:
            visuals = _load_urdf_visuals(urdf)
            visuals_cache["visuals"] = visuals
        if not visuals:
            return None
        height, width = observed_shape.shape[:2]
        tx = f(row.get("tx"))
        ty = f(row.get("ty"))
        tz = f(row.get("tz"))
        if tx is None or ty is None or tz is None:
            return None
        rot = _quat_matrix(row)
        trans = np.asarray([tx, ty, tz], dtype=float)
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        fx = f(row.get("fx"), DEFAULT_CAMERA["fx"]) or DEFAULT_CAMERA["fx"]
        fy = f(row.get("fy"), DEFAULT_CAMERA["fy"]) or DEFAULT_CAMERA["fy"]
        cx = f(row.get("cx"), DEFAULT_CAMERA["cx"]) or DEFAULT_CAMERA["cx"]
        cy = f(row.get("cy"), DEFAULT_CAMERA["cy"]) or DEFAULT_CAMERA["cy"]
        for vertices, faces in visuals:  # type: ignore[assignment]
            cam = np.asarray(vertices, dtype=float) @ rot.T + trans
            z = cam[:, 2]
            valid = z > 1e-6
            zz = np.clip(z, 1e-6, None)
            uv = np.column_stack([fx * cam[:, 0] / zz + cx, fy * cam[:, 1] / zz + cy])
            for face in np.asarray(faces, dtype=int):
                if len(face) < 3 or not np.all(valid[face]):
                    continue
                pts = [(float(uv[i, 0]), float(uv[i, 1])) for i in face]
                if all((x < -50 or x > width + 50 or y < -50 or y > height + 50) for x, y in pts):
                    continue
                draw.polygon(pts, fill=255)
        arr = np.asarray(image) > 127
        return arr if np.any(arr) else None
    except Exception:
        return None


def _mask_pair_metrics_array(observed, rendered) -> dict[str, float] | None:
    try:
        import numpy as np  # type: ignore

        h = min(observed.shape[0], rendered.shape[0])
        w = min(observed.shape[1], rendered.shape[1])
        if h <= 0 or w <= 0:
            return None
        obs = observed[:h, :w].astype(bool)
        ren = rendered[:h, :w].astype(bool)
        intersection = int(np.count_nonzero(obs & ren))
        union = int(np.count_nonzero(obs | ren))
        if union == 0:
            return None
        obs_area = int(np.count_nonzero(obs))
        render_area = int(np.count_nonzero(ren))
        false_positive = int(np.count_nonzero(ren & ~obs))
        return {
            "iou": intersection / union,
            "coverage": intersection / obs_area if obs_area else None,
            "render_false_coverage": false_positive / render_area if render_area else None,
            "observed_area_px": float(obs_area),
            "rendered_area_px": float(render_area),
        }
    except Exception:
        return None


def _mask_pair_metrics(observed: Mask, rendered: Mask) -> dict[str, float] | None:
    height = min(len(observed), len(rendered))
    if height <= 0:
        return None
    width = min(min((len(row) for row in observed[:height]), default=0), min((len(row) for row in rendered[:height]), default=0))
    if width <= 0:
        return None
    obs_area = 0
    render_area = 0
    intersection = 0
    union = 0
    false_positive = 0
    for y in range(height):
        for x in range(width):
            obs = observed[y][x]
            ren = rendered[y][x]
            obs_area += int(obs)
            render_area += int(ren)
            intersection += int(obs and ren)
            union += int(obs or ren)
            false_positive += int(ren and not obs)
    if union == 0:
        return None
    return {
        "iou": intersection / union,
        "coverage": intersection / obs_area if obs_area else None,
        "render_false_coverage": false_positive / render_area if render_area else None,
        "observed_area_px": float(obs_area),
        "rendered_area_px": float(render_area),
    }


def _blank_like(mask: Mask) -> Mask:
    return [[False for _ in row] for row in mask]


def _draw_circle(mask: Mask, cx: float, cy: float, radius: float) -> None:
    radius = max(0.0, radius)
    r2 = radius * radius
    height = len(mask)
    width = len(mask[0]) if height else 0
    y0 = max(0, int(cy - radius - 1))
    y1 = min(height - 1, int(cy + radius + 1))
    x0 = max(0, int(cx - radius - 1))
    x1 = min(width - 1, int(cx + radius + 1))
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2 + 1e-6:
                mask[y][x] = True


def _draw_line(mask: Mask, cx: float, cy: float, length: float, angle_rad: float, thickness: float) -> None:
    height = len(mask)
    width = len(mask[0]) if height else 0
    dx = 0.5 * length * math.cos(angle_rad)
    dy = 0.5 * length * math.sin(angle_rad)
    x1, y1 = cx - dx, cy - dy
    x2, y2 = cx + dx, cy + dy
    vx, vy = x2 - x1, y2 - y1
    denom = max(vx * vx + vy * vy, 1e-6)
    half_t = max(1.0, thickness * 0.5)
    min_x = max(0, int(min(x1, x2) - half_t - 1))
    max_x = min(width - 1, int(max(x1, x2) + half_t + 1))
    min_y = max(0, int(min(y1, y2) - half_t - 1))
    max_y = min(height - 1, int(max(y1, y2) + half_t + 1))
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            t = max(0.0, min(1.0, ((x - x1) * vx + (y - y1) * vy) / denom))
            px = x1 + t * vx
            py = y1 + t * vy
            if (x - px) * (x - px) + (y - py) * (y - py) <= half_t * half_t:
                mask[y][x] = True


def _pose_by_frame(paths: EvaluationPaths) -> dict[int, dict[str, str]]:
    rows = read_rows(paths.result_dir / "object_pose.csv")
    obs_rows = read_rows(paths.result_dir / "object_observations.csv")
    obs_by_frame: dict[int, dict[str, str]] = {}
    for row in obs_rows:
        frame = f(row.get("frame"))
        if frame is not None:
            obs_by_frame[int(frame)] = row
    by_frame: dict[int, dict[str, str]] = {}
    for row in rows:
        frame = f(row.get("frame"))
        if frame is not None:
            merged = dict(obs_by_frame.get(int(frame), {}))
            merged.update(row)
            for pose_key, obs_key in (
                ("u_proj", "ref_u"),
                ("v_proj", "ref_v"),
                ("u_obs", "ref_u"),
                ("v_obs", "ref_v"),
                ("radius_proj_px", "radius_px"),
                ("radius_obs_px", "radius_px"),
                ("u_proj", "center_x"),
                ("v_proj", "center_y"),
                ("radius_proj_px", "enclosing_radius_px"),
            ):
                if not merged.get(pose_key) and merged.get(obs_key):
                    merged[pose_key] = merged[obs_key]
            by_frame[int(frame)] = merged
    return by_frame


def _generate_mask_from_pose(row: dict[str, str], observed_shape: Mask) -> Mask | None:
    cx = f(row.get("u_proj"), f(row.get("u_obs"), f(row.get("ref_u"))))
    cy = f(row.get("v_proj"), f(row.get("v_obs"), f(row.get("ref_v"))))
    bbox = [f(row.get(key)) for key in ("mask_bbox_x1", "mask_bbox_y1", "mask_bbox_x2", "mask_bbox_y2")]
    if cx is None or cy is None:
        if all(v is not None for v in bbox):
            out = _blank_like(observed_shape)
            h = len(out)
            w = len(out[0]) if h else 0
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]  # type: ignore[arg-type]
            x1, x2 = max(0, min(x1, x2)), min(w - 1, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h - 1, max(y1, y2))
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    out[y][x] = True
            return out
        return None
    out = _blank_like(observed_shape)
    radius = f(row.get("radius_proj_px"), f(row.get("radius_obs_px"), f(row.get("radius_px"))))
    if radius is not None and radius > 0:
        _draw_circle(out, cx, cy, radius)
        return out
    line_len = f(row.get("object_pixel_len"), f(row.get("contact_lock_pixel_len"), f(row.get("visible_len_px"))))
    angle = f(row.get("contact_lock_angle_rad"))
    if line_len is not None and angle is not None and line_len > 0:
        thickness = max(5.0, min(28.0, line_len * 0.04))
        _draw_line(out, cx, cy, line_len, angle, thickness)
        return out
    return None


def _generate_mask_array_from_pose(row: dict[str, str], observed_shape):
    try:
        import numpy as np  # type: ignore

        cx = f(row.get("u_proj"), f(row.get("u_obs"), f(row.get("ref_u"))))
        cy = f(row.get("v_proj"), f(row.get("v_obs"), f(row.get("ref_v"))))
        height, width = observed_shape.shape[:2]
        if cx is None or cy is None:
            bbox = [f(row.get(key)) for key in ("mask_bbox_x1", "mask_bbox_y1", "mask_bbox_x2", "mask_bbox_y2")]
            if all(v is not None for v in bbox):
                out = np.zeros((height, width), dtype=bool)
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]  # type: ignore[arg-type]
                x1, x2 = max(0, min(x1, x2)), min(width - 1, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(height - 1, max(y1, y2))
                out[y1 : y2 + 1, x1 : x2 + 1] = True
                return out
            return None
        yy, xx = np.ogrid[:height, :width]
        radius = f(row.get("radius_proj_px"), f(row.get("radius_obs_px"), f(row.get("radius_px"))))
        if radius is not None and radius > 0:
            return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius
        line_len = f(row.get("object_pixel_len"), f(row.get("contact_lock_pixel_len"), f(row.get("visible_len_px"))))
        angle = f(row.get("contact_lock_angle_rad"))
        if line_len is None or angle is None or line_len <= 0:
            return None
        dx = 0.5 * line_len * math.cos(angle)
        dy = 0.5 * line_len * math.sin(angle)
        x1, y1 = cx - dx, cy - dy
        x2, y2 = cx + dx, cy + dy
        vx, vy = x2 - x1, y2 - y1
        denom = max(vx * vx + vy * vy, 1e-6)
        t = np.clip(((xx - x1) * vx + (yy - y1) * vy) / denom, 0.0, 1.0)
        px = x1 + t * vx
        py = y1 + t * vy
        thickness = max(5.0, min(28.0, line_len * 0.04))
        return ((xx - px) ** 2 + (yy - py) ** 2) <= (0.5 * thickness) ** 2
    except Exception:
        return None


def _generate_evaluation_render_masks(paths: EvaluationPaths, observed_masks: dict[int, Path], rendered_masks: dict[int, Path]) -> tuple[int, int]:
    pose_rows = _pose_by_frame(paths)
    out_dir = paths.evaluation_dir / "render_masks"
    proxy_generated = 0
    full_geometry_generated = 0
    visuals_cache: dict[str, object] = {}
    urdf_available = _resolve_urdf(paths) is not None
    for frame, observed_path in sorted(observed_masks.items()):
        if frame in rendered_masks:
            try:
                rendered_masks[frame].relative_to(paths.evaluation_dir / "render_masks")
                if not urdf_available:
                    continue
            except ValueError:
                continue
        row = pose_rows.get(frame)
        if row is None:
            continue
        observed_array = _read_mask_array(observed_path)
        if observed_array is not None:
            rendered_full = _render_urdf_mask_array(paths, row, observed_array, visuals_cache)
            if rendered_full is not None:
                target = out_dir / f"{frame:05d}_mask.pgm"
                _write_mask_array_pgm(target, rendered_full)
                full_geometry_generated += 1
                rendered_masks[frame] = target
                continue
            rendered_array = _generate_mask_array_from_pose(row, observed_array)
            if rendered_array is not None:
                target = out_dir / f"{frame:05d}_mask.pgm"
                _write_mask_array_pgm(target, rendered_array)
                proxy_generated += 1
                rendered_masks[frame] = target
                continue
        observed = _read_image_mask(observed_path)
        if observed is None:
            continue
        rendered = _generate_mask_from_pose(row, observed)
        if rendered is None:
            continue
        target = out_dir / f"{frame:05d}_mask.pgm"
        _write_pgm(target, rendered)
        proxy_generated += 1
        rendered_masks[frame] = target
    return proxy_generated, full_geometry_generated


def _compute_mask_overlay_metrics(paths: EvaluationPaths) -> tuple[list[dict[str, object]], int, int]:
    observed_masks = _collect_masks(
        [
            paths.sample_dir / "results" / "segmentation" / "masks",
            paths.result_dir / "segmentation" / "masks",
            paths.result_dir / "object_masks",
        ]
    )
    rendered_masks = _collect_masks(
        [
            paths.render_dir / "object_masks",
            paths.render_dir / "masks",
            paths.result_dir / "render_masks",
            paths.evaluation_dir / "render_masks",
        ]
    )
    proxy_generated_count, full_geometry_generated_count = _generate_evaluation_render_masks(paths, observed_masks, rendered_masks)
    rows: list[dict[str, object]] = []
    evaluation_render_mask_count = 0
    for frame in sorted(set(observed_masks) & set(rendered_masks)):
        try:
            rendered_masks[frame].relative_to(paths.evaluation_dir / "render_masks")
            evaluation_render_mask_count += 1
        except ValueError:
            pass
        observed_array = _read_mask_array(observed_masks[frame])
        rendered_array = _read_mask_array(rendered_masks[frame])
        metrics = _mask_pair_metrics_array(observed_array, rendered_array) if observed_array is not None and rendered_array is not None else None
        if metrics is None:
            observed = _read_image_mask(observed_masks[frame])
            rendered = _read_image_mask(rendered_masks[frame])
            if observed is None or rendered is None:
                continue
            metrics = _mask_pair_metrics(observed, rendered)
        if metrics is None:
            continue
        rows.append(
            {
                "frame": frame,
                "observed_mask": str(observed_masks[frame]),
                "rendered_mask": str(rendered_masks[frame]),
                **metrics,
            }
        )
    cached_generated = max(0, evaluation_render_mask_count - full_geometry_generated_count)
    return rows, max(proxy_generated_count, cached_generated), full_geometry_generated_count


def compute_overlay_metrics(paths: EvaluationPaths) -> MetricBlock:
    mask_rows, proxy_generated_count, full_geometry_generated_count = _compute_mask_overlay_metrics(paths)
    if mask_rows:
        overlay_hard_score = mean(f(row.get("iou")) for row in mask_rows)
        if full_geometry_generated_count:
            source = "generated_eval_full_geometry_mask_iou"
        elif proxy_generated_count:
            source = "generated_eval_proxy_render_mask_iou"
        else:
            source = "mask_pair_iou"
        metrics = {
            "overlay_hard_score": overlay_hard_score,
            "overlay_hard_metric_source": source,
            "overlay_mask_pair_count": len(mask_rows),
            "overlay_generated_render_mask_count": proxy_generated_count + full_geometry_generated_count,
            "overlay_generated_proxy_render_mask_count": proxy_generated_count,
            "overlay_generated_full_geometry_mask_count": full_geometry_generated_count,
            "overlay_mask_coverage": mean(f(row.get("coverage")) for row in mask_rows),
            "overlay_render_false_coverage": mean(f(row.get("render_false_coverage")) for row in mask_rows),
            "overlay_vlm_is_primary": False,
            "overlay_conflict_flag": "",
        }
        out_json = write_json(paths.evaluation_dir / "overlay_metrics.json", metrics)
        out_csv = write_rows(paths.evaluation_dir / "overlay_metrics.csv", [metrics])
        mask_csv = write_rows(paths.evaluation_dir / "overlay_mask_metrics.csv", mask_rows)
        return MetricBlock("overlay", metrics, {"json": str(out_json), "csv": str(out_csv), "mask_csv": str(mask_csv)})

    obs = read_rows(paths.result_dir / "object_observations.csv")
    line = read_rows(paths.result_dir / "line_correspondence.csv")
    values = []
    for row in obs:
        value = f(row.get("mask_iou"), f(row.get("object_mask_iou"), f(row.get("observation_conf"), f(row.get("mask_conf")))))
        if value is not None:
            values.append(max(0.0, min(1.0, value)))
    if not values:
        for row in line:
            value = f(row.get("line_observation_trusted"), f(row.get("endpoint_track_conf")))
            if value is not None:
                values.append(max(0.0, min(1.0, value)))
    overlay_hard_score = mean(values)
    metrics = {
        "overlay_hard_score": overlay_hard_score,
        "overlay_hard_metric_source": "mask_or_observation_conf" if obs and overlay_hard_score is not None else "line_confidence" if overlay_hard_score is not None else "missing",
        "overlay_mask_pair_count": 0,
        "overlay_generated_render_mask_count": 0,
        "overlay_generated_proxy_render_mask_count": 0,
        "overlay_generated_full_geometry_mask_count": 0,
        "overlay_mask_coverage": None,
        "overlay_render_false_coverage": None,
        "overlay_vlm_is_primary": False,
        "overlay_conflict_flag": "",
    }
    out_json = write_json(paths.evaluation_dir / "overlay_metrics.json", metrics)
    out_csv = write_rows(paths.evaluation_dir / "overlay_metrics.csv", [metrics])
    return MetricBlock("overlay", metrics, {"json": str(out_json), "csv": str(out_csv)})
