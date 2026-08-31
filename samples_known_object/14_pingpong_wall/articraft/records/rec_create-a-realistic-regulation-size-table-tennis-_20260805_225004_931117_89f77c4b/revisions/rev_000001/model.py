from __future__ import annotations

import math

import cadquery as cq

from sdk import (
    ArticulatedObject,
    Box,
    Origin,
    TestContext,
    TestReport,
    mesh_from_cadquery,
)


def _superellipse_points(
    rx: float,
    ry: float,
    cy: float,
    *,
    exponent: float = 2.65,
    count: int = 96,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    power = 2.0 / exponent
    for index in range(count):
        theta = 2.0 * math.pi * index / count
        cosine = math.cos(theta)
        sine = math.sin(theta)
        x = rx * math.copysign(abs(cosine) ** power, cosine)
        y = cy + ry * math.copysign(abs(sine) ** power, sine)
        points.append((x, y))
    return points


def _truncated_superellipse_points(
    rx: float,
    ry: float,
    cy: float,
    bottom: float,
    *,
    exponent: float = 2.65,
    count: int = 84,
) -> list[tuple[float, float]]:
    ratio = (cy - bottom) / ry
    crossing = math.asin(ratio ** (exponent / 2.0))
    start = 2.0 * math.pi - crossing
    stop = math.pi + crossing + 2.0 * math.pi
    power = 2.0 / exponent
    points: list[tuple[float, float]] = []
    for index in range(count):
        theta = start + (stop - start) * index / (count - 1)
        cosine = math.cos(theta)
        sine = math.sin(theta)
        x = rx * math.copysign(abs(cosine) ** power, cosine)
        y = cy + ry * math.copysign(abs(sine) ** power, sine)
        points.append((x, y))
    return points


def _profile(points: list[tuple[float, float]], z: float = 0.0) -> cq.Workplane:
    return cq.Workplane("XY", origin=(0.0, 0.0, z)).polyline(points).close()


def _rubber_texture_points(
    rx: float,
    ry: float,
    cy: float,
    bottom: float,
    exponent: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    margin_x = rx - 0.0030
    margin_y = ry - 0.0030
    spacing = 0.0085
    for row in range(18):
        y = bottom + 0.0045 + row * spacing
        if y > cy + margin_y:
            continue
        stagger = 0.5 * spacing if row % 2 else 0.0
        for column in range(-9, 10):
            x = column * spacing + stagger
            if abs(x) >= margin_x:
                continue
            inside = (abs(x) / margin_x) ** exponent + (abs(y - cy) / margin_y) ** exponent
            if inside <= 1.0:
                points.append((x, y))
    return points


def _make_rubber(
    *,
    front: bool,
    rx: float,
    ry: float,
    cy: float,
    bottom: float,
    exponent: float,
) -> cq.Workplane:
    points = _truncated_superellipse_points(
        rx, ry, cy, bottom, exponent=exponent
    )
    core_face = 0.0030 if front else -0.0030
    rubber_depth = 0.00275 if front else -0.00275
    rubber = _profile(points, core_face).extrude(rubber_depth)

    outer_face = core_face + rubber_depth
    texture_start = outer_face - 0.00003 if front else outer_face + 0.00003
    texture_depth = 0.00015 if front else -0.00015
    texture_points = _rubber_texture_points(rx, ry, cy, bottom, exponent)
    texture = (
        cq.Workplane("XY", origin=(0.0, 0.0, texture_start))
        .pushPoints(texture_points)
        .circle(0.00032)
        .extrude(texture_depth)
    )
    return rubber.union(texture)


def _make_handle() -> cq.Workplane:
    # Explicit plane gives +Y as the loft direction. The varying elliptical
    # sections form a flared, palm-filling laminated table-tennis grip.
    plane = cq.Plane(origin=(0.0, -0.130, 0.0), xDir=(1.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
    return (
        cq.Workplane(plane)
        .ellipse(0.0160, 0.0095)
        .workplane(offset=0.015)
        .ellipse(0.0180, 0.0105)
        .workplane(offset=0.040)
        .ellipse(0.0140, 0.0115)
        .workplane(offset=0.040)
        .ellipse(0.0155, 0.0105)
        .workplane(offset=0.015)
        .ellipse(0.0250, 0.0070)
        .loft(ruled=False, combine=True)
    )


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(
        name="regulation_table_tennis_paddle",
        meta={
            "coordinate_frame": "+X is blade width, +Y runs from handle toward blade crown, +Z is the red/front face normal",
            "nominal_dimensions_m": {
                "overall_length": 0.260,
                "blade_width": 0.150,
                "blade_height": 0.160,
                "blade_face_stack": 0.0118,
            },
        },
    )

    light_wood = model.material("blade_ply", rgba=(0.78, 0.57, 0.33, 1.0))
    edge_wood = model.material("blade_edge_wood", rgba=(0.68, 0.43, 0.22, 1.0))
    handle_wood = model.material("laminated_handle_wood", rgba=(0.72, 0.50, 0.28, 1.0))
    dark_ply = model.material("dark_handle_ply", rgba=(0.25, 0.12, 0.055, 1.0))
    red_rubber = model.material("front_red_rubber_material", rgba=(0.82, 0.025, 0.025, 1.0))
    black_rubber = model.material("rear_black_rubber_material", rgba=(0.025, 0.027, 0.029, 1.0))

    paddle_root = model.part(
        "paddle_root",
        meta={
            "rigid_tool": True,
            "face_normal_axis": "z",
            "handle_axis": "y",
            "blade_width_axis": "x",
        },
    )

    exponent = 2.65
    outer_points = _superellipse_points(0.0750, 0.0800, 0.0500, exponent=exponent)
    inner_points = _superellipse_points(0.0728, 0.0778, 0.0500, exponent=exponent)

    # Six-millimetre wooden blade core: `both=True` extrudes this distance
    # symmetrically, so 0.003 m gives faces at z = +/-0.003 m.
    outer_core = _profile(outer_points).extrude(0.0030, both=True)
    inner_core = _profile(inner_points).extrude(0.0030, both=True)
    blade_edge = outer_core.cut(inner_core)

    front_rubber = _make_rubber(
        front=True,
        rx=0.0735,
        ry=0.0787,
        cy=0.0503,
        bottom=-0.0120,
        exponent=exponent,
    )
    rear_rubber = _make_rubber(
        front=False,
        rx=0.0735,
        ry=0.0787,
        cy=0.0503,
        bottom=-0.0120,
        exponent=exponent,
    )
    handle = _make_handle()

    paddle_root.visual(
        mesh_from_cadquery(inner_core, "blade_core", tolerance=0.00020, angular_tolerance=0.12),
        material=light_wood,
        name="blade_core",
    )
    paddle_root.visual(
        mesh_from_cadquery(blade_edge, "blade_edge", tolerance=0.00018, angular_tolerance=0.10),
        material=edge_wood,
        name="blade_edge",
    )
    paddle_root.visual(
        mesh_from_cadquery(front_rubber, "front_red_rubber", tolerance=0.00008, angular_tolerance=0.16),
        material=red_rubber,
        name="front_red_rubber",
    )
    paddle_root.visual(
        mesh_from_cadquery(rear_rubber, "rear_black_rubber", tolerance=0.00008, angular_tolerance=0.16),
        material=black_rubber,
        name="rear_black_rubber",
    )
    paddle_root.visual(
        mesh_from_cadquery(handle, "handle", tolerance=0.00018, angular_tolerance=0.10),
        material=handle_wood,
        name="handle",
    )

    # Narrow contrasting plies are shallow, physically embedded in the grip.
    paddle_root.visual(
        Box((0.0040, 0.0780, 0.0008)),
        origin=Origin(xyz=(0.0, -0.0720, 0.01065)),
        material=dark_ply,
        name="handle_laminate_front",
    )
    paddle_root.visual(
        Box((0.0040, 0.0780, 0.0008)),
        origin=Origin(xyz=(0.0, -0.0720, -0.01065)),
        material=dark_ply,
        name="handle_laminate_rear",
    )

    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    root = object_model.get_part("paddle_root")

    semantic_visuals = {visual.name for visual in root.visuals}
    ctx.check(
        "semantic paddle construction is present",
        {"blade_core", "blade_edge", "front_red_rubber", "rear_black_rubber", "handle"}.issubset(semantic_visuals),
        details=f"visuals={sorted(semantic_visuals)}",
    )
    ctx.check(
        "single fixed rigid root",
        len(object_model.parts) == 1
        and len(object_model.root_parts()) == 1
        and object_model.root_parts()[0].name == "paddle_root",
        details=f"parts={[part.name for part in object_model.parts]}, roots={[part.name for part in object_model.root_parts()]}",
    )
    ctx.check(
        "paddle has no moving or fixed child joints",
        len(object_model.articulations) == 0,
        details=f"articulations={[joint.name for joint in object_model.articulations]}",
    )

    overall = ctx.part_world_aabb(root)
    if overall is None:
        ctx.fail("overall paddle bounds available", "paddle_root has no world AABB")
    else:
        lower, upper = overall
        width = upper[0] - lower[0]
        length = upper[1] - lower[1]
        depth = upper[2] - lower[2]
        ctx.check(
            "regulation-scale overall metric bounds",
            0.148 <= width <= 0.152
            and 0.257 <= length <= 0.263
            and 0.018 <= depth <= 0.025,
            details=f"width={width:.6f}, length={length:.6f}, max_depth={depth:.6f}",
        )

    core_bounds = ctx.part_element_world_aabb(root, elem="blade_core")
    red_bounds = ctx.part_element_world_aabb(root, elem="front_red_rubber")
    black_bounds = ctx.part_element_world_aabb(root, elem="rear_black_rubber")
    handle_bounds = ctx.part_element_world_aabb(root, elem="handle")

    if core_bounds and red_bounds and black_bounds:
        blade_width = max(
            red_bounds[1][0] - red_bounds[0][0],
            core_bounds[1][0] - core_bounds[0][0],
        )
        face_stack = red_bounds[1][2] - black_bounds[0][2]
        ctx.check(
            "thin regulation blade face stack",
            0.0110 <= face_stack <= 0.0125 and face_stack / blade_width < 0.085,
            details=f"face_stack={face_stack:.6f}, blade_width={blade_width:.6f}, ratio={face_stack / blade_width:.4f}",
        )
        ctx.check(
            "red front and black rear occupy opposite faces",
            red_bounds[0][2] >= core_bounds[1][2] - 0.00005
            and black_bounds[1][2] <= core_bounds[0][2] + 0.00005
            and red_bounds[1][2] > 0.0
            and black_bounds[0][2] < 0.0,
            details=f"red_z={red_bounds[0][2]:.6f}..{red_bounds[1][2]:.6f}, core_z={core_bounds[0][2]:.6f}..{core_bounds[1][2]:.6f}, black_z={black_bounds[0][2]:.6f}..{black_bounds[1][2]:.6f}",
        )

    front_visual = root.get_visual("front_red_rubber")
    rear_visual = root.get_visual("rear_black_rubber")
    front_color = front_visual.material.rgba if front_visual.material is not None else None
    rear_color = rear_visual.material.rgba if rear_visual.material is not None else None
    ctx.check(
        "face materials preserve red black identity",
        front_color is not None
        and rear_color is not None
        and front_color[0] > 0.65
        and front_color[1] < 0.12
        and front_color[2] < 0.12
        and max(rear_color[:3]) < 0.08,
        details=f"front_rgba={front_color}, rear_rgba={rear_color}",
    )

    if handle_bounds and core_bounds:
        handle_center_x = (handle_bounds[0][0] + handle_bounds[1][0]) * 0.5
        ctx.check(
            "handle is centered on blade axis",
            abs(handle_center_x) <= 0.0005
            and handle_bounds[1][1] > core_bounds[0][1]
            and handle_bounds[0][1] <= -0.129,
            details=f"handle_center_x={handle_center_x:.6f}, handle_y={handle_bounds[0][1]:.6f}..{handle_bounds[1][1]:.6f}, blade_bottom={core_bounds[0][1]:.6f}",
        )

    ctx.expect_contact(
        root,
        root,
        elem_a="handle",
        elem_b="blade_core",
        contact_tol=0.00005,
        name="handle is physically attached to blade core",
    )
    ctx.expect_contact(
        root,
        root,
        elem_a="blade_core",
        elem_b="blade_edge",
        contact_tol=0.00005,
        name="wood core is connected to perimeter edge",
    )
    ctx.expect_contact(
        root,
        root,
        elem_a="front_red_rubber",
        elem_b="blade_core",
        contact_tol=0.00005,
        name="front rubber is bonded to blade",
    )
    ctx.expect_contact(
        root,
        root,
        elem_a="rear_black_rubber",
        elem_b="blade_core",
        contact_tol=0.00005,
        name="rear rubber is bonded to blade",
    )
    ctx.expect_overlap(
        root,
        root,
        axes="x",
        elem_a="handle",
        elem_b="blade_core",
        min_overlap=0.045,
        name="flared handle attachment overlaps blade width",
    )

    return ctx.report()


object_model = build_object_model()
