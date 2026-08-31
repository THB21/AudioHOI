from __future__ import annotations

from math import isclose

from sdk import (
    ArticulatedObject,
    ArticulationType,
    Box,
    Cylinder,
    Material,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)


BODY_WIDTH = 0.38
BODY_DEPTH = 0.23
BODY_HEIGHT = 0.46
WHEEL_RADIUS = 0.035
WHEEL_WIDTH = 0.025
OVERALL_HEIGHT = BODY_HEIGHT + WHEEL_RADIUS * 2.0
HANDLE_TRAVEL = 0.28

SHELL = Material(name="dark_charcoal_shell", color=(0.08, 0.085, 0.08, 1.0))
TRIM = Material(name="black_trim", color=(0.01, 0.01, 0.01, 1.0))
TIRE = Material(name="black_rubber_tire", color=(0.015, 0.014, 0.013, 1.0))
HUB = Material(name="dark_wheel_hub", color=(0.05, 0.05, 0.05, 1.0))
METAL = Material(name="brushed_silver_rails", color=(0.72, 0.74, 0.72, 1.0))


def _add_shell_box(part, size, xyz, name, material=SHELL):
    part.visual(Box(size), origin=Origin(xyz=xyz), name=name, material=material)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="black_hardshell_carry_on")

    body = model.part("suitcase_body")

    # Layered box construction gives a hard-shell suitcase silhouette with rounded-corner blocks.
    _add_shell_box(body, (BODY_WIDTH - 0.045, BODY_DEPTH, BODY_HEIGHT), (0.0, 0.0, WHEEL_RADIUS * 2.0 + BODY_HEIGHT / 2.0), "main_shell")
    _add_shell_box(body, (BODY_WIDTH, BODY_DEPTH - 0.045, BODY_HEIGHT), (0.0, 0.0, WHEEL_RADIUS * 2.0 + BODY_HEIGHT / 2.0), "cross_shell")

    corner_w = 0.045
    corner_d = 0.045
    zc = WHEEL_RADIUS * 2.0 + BODY_HEIGHT / 2.0
    for ix, x in enumerate((-BODY_WIDTH / 2.0 + corner_w / 2.0, BODY_WIDTH / 2.0 - corner_w / 2.0)):
        for iy, y in enumerate((-BODY_DEPTH / 2.0 + corner_d / 2.0, BODY_DEPTH / 2.0 - corner_d / 2.0)):
            _add_shell_box(body, (corner_w, corner_d, BODY_HEIGHT - 0.025), (x, y, zc), f"rounded_corner_{ix}_{iy}")

    # Raised horizontal ribs and center zipper seam from the reference sheet.
    rib_zs = [0.145, 0.195, 0.245, 0.295, 0.345, 0.395, 0.445]
    for i, z in enumerate(rib_zs):
        _add_shell_box(body, (BODY_WIDTH + 0.006, 0.010, 0.010), (0.0, -BODY_DEPTH / 2.0 - 0.004, z), f"front_rib_{i}", TRIM)
        _add_shell_box(body, (0.085, 0.010, 0.010), (-0.148, BODY_DEPTH / 2.0 + 0.004, z), f"rear_rib_{i}_0", TRIM)
        _add_shell_box(body, (0.090, 0.010, 0.010), (0.0, BODY_DEPTH / 2.0 + 0.004, z), f"rear_rib_{i}_1", TRIM)
        _add_shell_box(body, (0.085, 0.010, 0.010), (0.148, BODY_DEPTH / 2.0 + 0.004, z), f"rear_rib_{i}_2", TRIM)

    _add_shell_box(body, (BODY_WIDTH + 0.012, 0.020, 0.014), (0.0, 0.0, WHEEL_RADIUS * 2.0 + BODY_HEIGHT * 0.53), "zipper_belt", TRIM)
    _add_shell_box(body, (0.012, BODY_DEPTH + 0.020, BODY_HEIGHT - 0.040), (BODY_WIDTH / 2.0 + 0.004, 0.0, zc), "side_zipper_track", TRIM)

    # Flush top handle mounted into a shallow recess.
    top_z = WHEEL_RADIUS * 2.0 + BODY_HEIGHT
    _add_shell_box(body, (0.180, 0.055, 0.016), (0.0, 0.0, top_z + 0.006), "top_handle_recess", TRIM)
    _add_shell_box(body, (0.140, 0.024, 0.018), (0.0, 0.0, top_z + 0.020), "top_carry_handle", TRIM)
    _add_shell_box(body, (0.020, 0.040, 0.018), (-0.080, 0.0, top_z + 0.018), "top_handle_mount_0", TRIM)
    _add_shell_box(body, (0.020, 0.040, 0.018), (0.080, 0.0, top_z + 0.018), "top_handle_mount_1", TRIM)

    # Telescoping sockets remain on the body so the child rails are visibly retained.
    rail_xs = (-0.070, 0.070)
    socket_y = BODY_DEPTH / 2.0 + 0.010
    for i, x in enumerate(rail_xs):
        body.visual(Cylinder(radius=0.014, length=0.080), origin=Origin(xyz=(x, socket_y, top_z + 0.025), rpy=(0.0, 0.0, 0.0)), name=f"rail_socket_{i}", material=TRIM)
        _add_shell_box(body, (0.034, 0.026, 0.075), (x, BODY_DEPTH / 2.0 + 0.003, top_z - 0.020), f"socket_base_{i}", TRIM)

    # Small side combination lock and zipper pulls, fused to body for support.
    _add_shell_box(body, (0.010, 0.045, 0.060), (BODY_WIDTH / 2.0 + 0.010, 0.010, zc + 0.030), "combination_lock", TRIM)
    for i, z in enumerate((zc + 0.015, zc + 0.050)):
        _add_shell_box(body, (0.011, 0.030, 0.014), (BODY_WIDTH / 2.0 + 0.014, -0.028, z), f"zipper_pull_{i}", TRIM)

    # Fixed spinner wheels: each link is mounted by a fork block and axle, no wheel articulation.
    wheel_positions = [
        (-BODY_WIDTH / 2.0 + 0.045, -BODY_DEPTH / 2.0 + 0.030),
        (BODY_WIDTH / 2.0 - 0.045, -BODY_DEPTH / 2.0 + 0.030),
        (-BODY_WIDTH / 2.0 + 0.045, BODY_DEPTH / 2.0 - 0.030),
        (BODY_WIDTH / 2.0 - 0.045, BODY_DEPTH / 2.0 - 0.030),
    ]
    for i, (x, y) in enumerate(wheel_positions):
        wheel = model.part(f"wheel_{i}")
        wheel.visual(Cylinder(radius=WHEEL_RADIUS, length=WHEEL_WIDTH), origin=Origin(xyz=(0.0, 0.0, 0.0), rpy=(1.5708, 0.0, 0.0)), name="tire", material=TIRE)
        wheel.visual(Cylinder(radius=0.018, length=WHEEL_WIDTH + 0.004), origin=Origin(xyz=(0.0, 0.0, 0.0), rpy=(1.5708, 0.0, 0.0)), name="hub", material=HUB)
        wheel.visual(Box((0.040, 0.030, 0.026)), origin=Origin(xyz=(0.0, 0.0, WHEEL_RADIUS - 0.013)), name="caster_fork", material=TRIM)
        model.articulation(
            f"body_to_wheel_{i}",
            ArticulationType.FIXED,
            parent=body,
            child=wheel,
            origin=Origin(xyz=(x, y, WHEEL_RADIUS)),
        )

    handle = model.part("telescoping_handle")
    rail_y = socket_y
    rail_height = 0.320
    for i, x in enumerate(rail_xs):
        handle.visual(Cylinder(radius=0.007, length=rail_height), origin=Origin(xyz=(x, 0.0, -0.025), rpy=(0.0, 0.0, 0.0)), name=f"silver_rail_{i}", material=METAL)
    handle.visual(Box((0.175, 0.038, 0.026)), origin=Origin(xyz=(0.0, 0.0, 0.136)), name="black_grip", material=TRIM)
    handle.visual(Box((0.044, 0.030, 0.024)), origin=Origin(xyz=(-0.070, 0.0, 0.137)), name="grip_post_0", material=TRIM)
    handle.visual(Box((0.044, 0.030, 0.024)), origin=Origin(xyz=(0.070, 0.0, 0.137)), name="grip_post_1", material=TRIM)

    model.articulation(
        "handle_extension",
        ArticulationType.PRISMATIC,
        parent=body,
        child=handle,
        origin=Origin(xyz=(0.0, rail_y, top_z - 0.055)),
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(effort=20.0, velocity=0.35, lower=0.0, upper=HANDLE_TRAVEL),
    )

    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    body = object_model.get_part("suitcase_body")
    handle = object_model.get_part("telescoping_handle")
    joint = object_model.get_articulation("handle_extension")
    wheels = [object_model.get_part(f"wheel_{i}") for i in range(4)]

    roots = list(object_model.root_parts())
    ctx.check("exactly one root suitcase_body", len(roots) == 1 and roots[0].name == "suitcase_body", details=f"roots={[p.name for p in roots]}")
    ctx.check("four semantic wheel links", all(w is not None for w in wheels) and len(wheels) == 4, details="wheel_0 through wheel_3 must exist")

    body_aabb = ctx.part_world_aabb(body)
    ctx.check("plausible carry-on width", body_aabb is not None and 0.36 <= (body_aabb[1][0] - body_aabb[0][0]) <= 0.42, details=f"body_aabb={body_aabb}")
    ctx.check("plausible carry-on depth", body_aabb is not None and 0.21 <= (body_aabb[1][1] - body_aabb[0][1]) <= 0.27, details=f"body_aabb={body_aabb}")

    wheel_aabbs = [ctx.part_world_aabb(w) for w in wheels]
    bottoms = [aabb[0][2] for aabb in wheel_aabbs if aabb is not None]
    ctx.check("four coplanar wheel bottoms", len(bottoms) == 4 and max(bottoms) - min(bottoms) <= 0.002 and all(abs(v) <= 0.002 for v in bottoms), details=f"bottoms={bottoms}")

    ctx.expect_overlap(handle, body, axes="x", elem_a="silver_rail_0", elem_b="rail_socket_0", min_overlap=0.010, name="rail 0 centered in socket footprint")
    ctx.expect_overlap(handle, body, axes="x", elem_a="silver_rail_1", elem_b="rail_socket_1", min_overlap=0.010, name="rail 1 centered in socket footprint")
    ctx.expect_within(handle, body, axes="x", inner_elem="black_grip", outer_elem="top_carry_handle", margin=0.040, name="grip centered over suitcase body")

    rest_pos = ctx.part_world_position(handle)
    with ctx.pose({joint: HANDLE_TRAVEL}):
        extended_pos = ctx.part_world_position(handle)
        ctx.expect_overlap(handle, body, axes="z", elem_a="silver_rail_0", elem_b="rail_socket_0", min_overlap=0.020, name="rail 0 retained at full extension")
        ctx.expect_overlap(handle, body, axes="z", elem_a="silver_rail_1", elem_b="rail_socket_1", min_overlap=0.020, name="rail 1 retained at full extension")
        full_aabb = ctx.part_world_aabb(handle)
        ctx.check("extended reconstruction height plausible", full_aabb is not None and full_aabb[1][2] <= 0.91, details=f"handle_aabb={full_aabb}")

    ctx.check("handle extends upward", rest_pos is not None and extended_pos is not None and extended_pos[2] > rest_pos[2] + 0.25, details=f"rest={rest_pos}, extended={extended_pos}")

    # The only intentional intersections are retained telescoping rails seated in body sockets.
    ctx.allow_overlap(body, handle, elem_a="rail_socket_0", elem_b="silver_rail_0", reason="The silver rail is intentionally retained inside the black telescoping socket.")
    ctx.allow_overlap(body, handle, elem_a="rail_socket_1", elem_b="silver_rail_1", reason="The silver rail is intentionally retained inside the black telescoping socket.")
    ctx.allow_overlap(body, handle, elem_a="socket_base_0", elem_b="silver_rail_0", reason="The lower rail passes through the reinforced socket base that captures the telescoping member.")
    ctx.allow_overlap(body, handle, elem_a="socket_base_1", elem_b="silver_rail_1", reason="The lower rail passes through the reinforced socket base that captures the telescoping member.")

    return ctx.report()


object_model = build_object_model()
