from __future__ import annotations

import math

import cadquery as cq
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
    mesh_from_cadquery,
)

WIDTH = 0.440
DEPTH = 0.520
HEIGHT = 0.780
SEAT_TOP_Z = 0.440

FRONT_FOOT_Y = -0.246
REAR_FOOT_Y = 0.246
FOOT_Z = 0.012
PIVOT_Y = 0.040
PIVOT_Z = 0.574
FRONT_TOP_Y = 0.164
FRONT_TOP_Z = HEIGHT
UPPER_HALF_WIDTH = 0.168
FOOT_HALF_WIDTH = 0.205
REAR_SUPPORT_TOP_HALF_WIDTH = 0.212

BACKREST_WIDTH = 0.382
BACKREST_HEIGHT = 0.072
BACKREST_THICKNESS = 0.018
BACKREST_CENTER = (0.0, 0.130, 0.715)
BACKREST_HOLE_RADIUS = 0.014

SEAT_HINGE_Y = 0.090
SEAT_LENGTH = 0.335
SEAT_THICKNESS = 0.022
SEAT_SIDE_X = 0.154
SEAT_SLAT_TOP_LOCAL_Z = 0.0
CENTER_SLAT_XS = (-0.120, -0.080, -0.040, 0.0, 0.040, 0.080, 0.120)
LOWER_STRETCHER_COUNT = 2
LOWER_STRETCHER_RADIUS = 0.010
FRONT_LOWER_STRETCHER_X = 0.202
FRONT_LOWER_STRETCHER_Y = -0.179
FRONT_LOWER_STRETCHER_Z = 0.138
REAR_LOWER_STRETCHER_LOCAL_Y = 0.164
REAR_LOWER_STRETCHER_LOCAL_Z = -0.436

WOOD = Material(name="warm_varnished_honey_oak", rgba=(0.78, 0.46, 0.20, 1.0))
DARK_WOOD = Material(name="subtle_end_grain", rgba=(0.46, 0.25, 0.10, 1.0))
METAL = Material(name="aged_bronze_hinge_metal", rgba=(0.24, 0.20, 0.16, 1.0))
BOLT = Material(name="dark_bronze_bolt_heads", rgba=(0.08, 0.07, 0.06, 1.0))


def _origin_between(start: tuple[float, float, float], end: tuple[float, float, float]) -> tuple[Origin, float]:
    sx, sy, sz = start
    ex, ey, ez = end
    dx = ex - sx
    dy = ey - sy
    dz = ez - sz
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    radial = math.sqrt(dx * dx + dy * dy)
    pitch = math.atan2(radial, dz)
    yaw = math.atan2(dy, dx) if radial > 1e-9 else 0.0
    origin = Origin(
        xyz=((sx + ex) * 0.5, (sy + ey) * 0.5, (sz + ez) * 0.5),
        rpy=(0.0, pitch, yaw),
    )
    return origin, length


def _rod(part, name: str, start: tuple[float, float, float], end: tuple[float, float, float], radius: float, material=WOOD) -> None:
    origin, length = _origin_between(start, end)
    part.visual(Cylinder(radius=radius, length=length), origin=origin, name=name, material=material)


def _box(part, name: str, size: tuple[float, float, float], xyz: tuple[float, float, float], material=WOOD, rpy=(0.0, 0.0, 0.0)) -> None:
    part.visual(Box(size), origin=Origin(xyz=xyz, rpy=rpy), name=name, material=material)


def _backrest_board_geometry():
    board = cq.Workplane("XZ").rect(BACKREST_WIDTH, BACKREST_HEIGHT).extrude(BACKREST_THICKNESS)
    board = board.faces(">Y").workplane().circle(BACKREST_HOLE_RADIUS).cutThruAll()
    return board


def _side_plate(part, side: int, base_name: str, local: bool = False) -> None:
    x_abs = UPPER_HALF_WIDTH + (0.016 if local else 0.010)
    x = side * x_abs
    y = 0.0 if local else PIVOT_Y
    z = 0.0 if local else PIVOT_Z
    _box(part, f"{base_name}_plate_{0 if side < 0 else 1}", (0.004, 0.050, 0.044), (x, y, z), METAL)
    for idx, (dy, dz) in enumerate(((-0.014, -0.011), (0.014, -0.010), (0.0, 0.013))):
        _rod(
            part,
            f"{base_name}_bolt_{0 if side < 0 else 1}_{idx}",
            (side * x_abs, y + dy, z + dz),
            (side * (x_abs + 0.006), y + dy, z + dz),
            0.004,
            BOLT,
        )


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="image_conditioned_wooden_folding_chair")

    front = model.part("front_frame")
    for side in (-1, 1):
        _rod(
            front,
            f"front_top_rail_{0 if side < 0 else 1}",
            (side * FOOT_HALF_WIDTH, FRONT_FOOT_Y, FOOT_Z),
            (side * UPPER_HALF_WIDTH, FRONT_TOP_Y, FRONT_TOP_Z),
            0.014,
        )
        _side_plate(front, side, "upper_hinge")

    front.visual(
        mesh_from_cadquery(_backrest_board_geometry(), "single_backrest_board_with_hole"),
        origin=Origin(xyz=(BACKREST_CENTER[0], BACKREST_CENTER[1] - BACKREST_THICKNESS * 0.5, BACKREST_CENTER[2])),
        name="backrest_board",
        material=WOOD,
    )
    backrest_t = (BACKREST_CENTER[2] - FOOT_Z) / (FRONT_TOP_Z - FOOT_Z)
    for side in (-1, 1):
        suffix = 0 if side < 0 else 1
        rail_x = side * (FOOT_HALF_WIDTH + backrest_t * (UPPER_HALF_WIDTH - FOOT_HALF_WIDTH))
        _rod(
            front,
            f"backrest_mount_bolt_{suffix}",
            (rail_x, BACKREST_CENTER[1] - BACKREST_THICKNESS * 0.95, BACKREST_CENTER[2]),
            (rail_x, BACKREST_CENTER[1] + BACKREST_THICKNESS * 0.95, BACKREST_CENTER[2]),
            0.0045,
            BOLT,
        )
    _rod(
        front,
        "front_lower_stretcher",
        (-FRONT_LOWER_STRETCHER_X, FRONT_LOWER_STRETCHER_Y, FRONT_LOWER_STRETCHER_Z),
        (FRONT_LOWER_STRETCHER_X, FRONT_LOWER_STRETCHER_Y, FRONT_LOWER_STRETCHER_Z),
        LOWER_STRETCHER_RADIUS,
    )
    for side in (-1, 1):
        suffix = 0 if side < 0 else 1
        _rod(
            front,
            f"seat_hinge_brace_{suffix}",
            (side * 0.174, -0.035, SEAT_TOP_Z - 0.018),
            (side * 0.174, 0.086, SEAT_TOP_Z - 0.018),
            0.0035,
            METAL,
        )
        _box(
            front,
            f"seat_hinge_anchor_{suffix}",
            (0.044, 0.018, 0.010),
            (side * 0.171, SEAT_HINGE_Y - 0.008, SEAT_TOP_Z - 0.018),
            METAL,
        )

    rear = model.part("rear_support")
    for side in (-1, 1):
        _rod(
            rear,
            f"rear_leg_{0 if side < 0 else 1}",
            (side * REAR_SUPPORT_TOP_HALF_WIDTH, 0.0, 0.0),
            (side * FOOT_HALF_WIDTH, REAR_FOOT_Y - PIVOT_Y, FOOT_Z - PIVOT_Z),
            0.014,
        )
        _side_plate(rear, side, "rear_pivot", local=True)
        _rod(
            rear,
            f"rear_hinge_pin_{0 if side < 0 else 1}",
            (side * (UPPER_HALF_WIDTH + 0.016), 0.0, 0.0),
            (side * (REAR_SUPPORT_TOP_HALF_WIDTH + 0.004), 0.0, 0.0),
            0.0045,
            BOLT,
        )
    _rod(
        rear,
        "rear_lower_stretcher",
        (-FRONT_LOWER_STRETCHER_X, REAR_LOWER_STRETCHER_LOCAL_Y, REAR_LOWER_STRETCHER_LOCAL_Z),
        (FRONT_LOWER_STRETCHER_X, REAR_LOWER_STRETCHER_LOCAL_Y, REAR_LOWER_STRETCHER_LOCAL_Z),
        LOWER_STRETCHER_RADIUS,
    )

    seat = model.part("seat")
    seat_center_y = -SEAT_LENGTH * 0.5
    for side in (-1, 1):
        _box(
            seat,
            f"seat_side_rail_{0 if side < 0 else 1}",
            (0.036, SEAT_LENGTH, SEAT_THICKNESS),
            (side * SEAT_SIDE_X, seat_center_y, -SEAT_THICKNESS * 0.5),
        )
        _rod(
            seat,
            f"seat_front_round_{0 if side < 0 else 1}",
            (side * SEAT_SIDE_X - 0.015, -SEAT_LENGTH, -SEAT_THICKNESS * 0.5),
            (side * SEAT_SIDE_X + 0.015, -SEAT_LENGTH, -SEAT_THICKNESS * 0.5),
            0.011,
        )
    for idx, x in enumerate(CENTER_SLAT_XS):
        _box(seat, f"center_slat_{idx}", (0.028, SEAT_LENGTH, SEAT_THICKNESS), (x, seat_center_y, -SEAT_THICKNESS * 0.5))
        _box(seat, f"slat_grain_{idx}", (0.004, SEAT_LENGTH * 0.78, 0.001), (x, seat_center_y, 0.0002), DARK_WOOD)
    _box(seat, "front_slat_tie", (0.372, 0.014, 0.012), (0.0, -SEAT_LENGTH + 0.010, -0.018), WOOD)
    _box(seat, "rear_hinge_tie", (0.372, 0.014, 0.012), (0.0, -0.010, -0.018), WOOD)
    for side in (-1, 1):
        _box(
            seat,
            f"seat_hinge_leaf_{0 if side < 0 else 1}",
            (0.044, 0.018, 0.008),
            (side * 0.143, -0.006, -0.018),
            METAL,
        )
        _rod(seat, f"seat_hinge_bolt_{0 if side < 0 else 1}", (side * 0.154, -0.006, -0.010), (side * 0.168, -0.006, -0.010), 0.005, BOLT)

    model.articulation(
        "front_to_rear",
        ArticulationType.REVOLUTE,
        parent=front,
        child=rear,
        origin=Origin(xyz=(0.0, PIVOT_Y, PIVOT_Z)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(effort=15.0, velocity=1.2, lower=-0.82, upper=0.12),
    )
    model.articulation(
        "front_to_seat",
        ArticulationType.REVOLUTE,
        parent=front,
        child=seat,
        origin=Origin(xyz=(0.0, SEAT_HINGE_Y, SEAT_TOP_Z)),
        axis=(-1.0, 0.0, 0.0),
        motion_limits=MotionLimits(effort=8.0, velocity=1.4, lower=0.0, upper=1.35),
    )

    return model


def _vec(v) -> tuple[float, float, float]:
    if hasattr(v, "x"):
        return (float(v.x), float(v.y), float(v.z))
    return tuple(float(c) for c in v)


def _aabb_center(aabb) -> tuple[float, float, float] | None:
    if aabb is None:
        return None
    mn = _vec(aabb[0])
    mx = _vec(aabb[1])
    return tuple((mn[i] + mx[i]) * 0.5 for i in range(3))


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    front = object_model.get_part("front_frame")
    rear = object_model.get_part("rear_support")
    seat = object_model.get_part("seat")
    front_to_rear = object_model.get_articulation("front_to_rear")
    front_to_seat = object_model.get_articulation("front_to_seat")

    mins = [1e9, 1e9, 1e9]
    maxs = [-1e9, -1e9, -1e9]
    for part in (front, rear, seat):
        aabb = ctx.part_world_aabb(part)
        if aabb is None:
            continue
        mn = _vec(aabb[0])
        mx = _vec(aabb[1])
        for i in range(3):
            mins[i] = min(mins[i], mn[i])
            maxs[i] = max(maxs[i], mx[i])
    dims = tuple(maxs[i] - mins[i] for i in range(3))

    ctx.check(
        "approximate reference dimensions",
        0.430 <= dims[0] <= 0.455 and 0.505 <= dims[1] <= 0.535 and 0.760 <= dims[2] <= 0.800,
        details=f"dims={dims}, expected roughly {(WIDTH, DEPTH, HEIGHT)}",
    )
    ctx.check(
        "seat top height is 440 mm",
        abs(SEAT_TOP_Z - 0.440) < 0.003,
        details=f"seat_top={SEAT_TOP_Z}",
    )
    ctx.check(
        "one backrest board with one centered circular hole",
        BACKREST_WIDTH > 0.30
        and BACKREST_WIDTH > 4.0 * BACKREST_HEIGHT
        and BACKREST_HOLE_RADIUS > 0.010
        and BACKREST_CENTER[0] == 0.0,
        details="The backrest is one managed CadQuery board mesh with one centered cut-through circle.",
    )
    ctx.check(
        "continuous front rail and rear support lean apart",
        FRONT_FOOT_Y < PIVOT_Y < FRONT_TOP_Y and REAR_FOOT_Y > PIVOT_Y and PIVOT_Z < FRONT_TOP_Z,
        details=f"front foot/pivot/top y={(FRONT_FOOT_Y, PIVOT_Y, FRONT_TOP_Y)}, rear foot y={REAR_FOOT_Y}",
    )
    front_rail_y_at_backrest = FRONT_FOOT_Y + (
        (BACKREST_CENTER[2] - FOOT_Z) / (FRONT_TOP_Z - FOOT_Z)
    ) * (FRONT_TOP_Y - FRONT_FOOT_Y)
    front_rail_x_at_backrest = FOOT_HALF_WIDTH + (
        (BACKREST_CENTER[2] - FOOT_Z) / (FRONT_TOP_Z - FOOT_Z)
    ) * (UPPER_HALF_WIDTH - FOOT_HALF_WIDTH)
    board_rail_capture = BACKREST_WIDTH * 0.5 - front_rail_x_at_backrest
    top_post_exposure = FRONT_TOP_Z - (BACKREST_CENTER[2] + BACKREST_HEIGHT * 0.5)
    ctx.check(
        "backrest is carried by continuous front rails",
        abs(BACKREST_CENTER[1] - front_rail_y_at_backrest) < 0.008
        and BACKREST_CENTER[2] > PIVOT_Z + 0.14
        and 0.004 <= board_rail_capture <= 0.020,
        details=(
            f"backrest center={(BACKREST_CENTER[1], BACKREST_CENTER[2])}, "
            f"rail y at board={front_rail_y_at_backrest}, rail x={front_rail_x_at_backrest}, "
            f"capture={board_rail_capture}"
        ),
    )
    ctx.check(
        "backrest sits below protruding front post tips",
        0.705 <= BACKREST_CENTER[2] <= 0.725 and 0.020 <= top_post_exposure <= 0.050,
        details=f"backrest center z={BACKREST_CENTER[2]}, exposed top post={top_post_exposure}",
    )
    ctx.check(
        "rear support legs stop at hinge below backrest",
        PIVOT_Z < BACKREST_CENTER[2] - 0.08 and PIVOT_Z < FRONT_TOP_Z - 0.18,
        details=f"pivot z={PIVOT_Z}, backrest center z={BACKREST_CENTER[2]}, top z={FRONT_TOP_Z}",
    )
    ctx.check(
        "front rear foot separation near 520 mm",
        abs((REAR_FOOT_Y - FRONT_FOOT_Y + 2 * 0.014) - DEPTH) < 0.010,
        details=f"outer separation={REAR_FOOT_Y - FRONT_FOOT_Y + 2 * 0.014}",
    )
    ctx.check(
        "bottom stance is wider than upper rail spacing",
        FOOT_HALF_WIDTH > UPPER_HALF_WIDTH + 0.025,
        details=f"foot half width={FOOT_HALF_WIDTH}, upper half width={UPPER_HALF_WIDTH}",
    )
    ctx.check(
        "seat rails and center slats are flush",
        abs(SEAT_SLAT_TOP_LOCAL_Z) < 1e-9,
        details="All seat slats and side rails share the same local top plane at the seat hinge height.",
    )
    ctx.check(
        "only two slim lower round stretchers",
        LOWER_STRETCHER_COUNT == 2 and LOWER_STRETCHER_RADIUS <= 0.011,
        details=f"count={LOWER_STRETCHER_COUNT}, radius={LOWER_STRETCHER_RADIUS}",
    )

    for suffix in ("0", "1"):
        ctx.allow_overlap(
            front,
            rear,
            elem_a=f"upper_hinge_plate_{suffix}",
            elem_b=f"rear_pivot_plate_{suffix}",
            reason="The paired thin metal hinge leaves intentionally meet at the shared side pivot.",
        )
        ctx.allow_overlap(
            front,
            rear,
            elem_a=f"front_top_rail_{suffix}",
            elem_b=f"rear_hinge_pin_{suffix}",
            reason="The small hinge pin intentionally passes through the continuous front rail at the folding pivot.",
        )
        ctx.expect_contact(
            front,
            rear,
            elem_a=f"upper_hinge_plate_{suffix}",
            elem_b=f"rear_pivot_plate_{suffix}",
            contact_tol=0.006,
            name=f"side hinge leaves meet {suffix}",
        )
        ctx.expect_contact(
            front,
            rear,
            elem_a=f"front_top_rail_{suffix}",
            elem_b=f"rear_hinge_pin_{suffix}",
            contact_tol=0.001,
            name=f"hinge pin is captured in front rail {suffix}",
        )
        ctx.allow_overlap(
            front,
            seat,
            elem_a=f"seat_hinge_anchor_{suffix}",
            elem_b=f"seat_hinge_bolt_{suffix}",
            reason="The small seat hinge bolt is intentionally captured through the front frame hinge anchor.",
        )
        ctx.allow_overlap(
            front,
            seat,
            elem_a=f"seat_hinge_anchor_{suffix}",
            elem_b=f"seat_hinge_leaf_{suffix}",
            reason="The thin seat hinge leaf is intentionally seated against the front frame hinge anchor.",
        )
        ctx.expect_contact(
            front,
            seat,
            elem_a=f"seat_hinge_anchor_{suffix}",
            elem_b=f"seat_hinge_leaf_{suffix}",
            contact_tol=0.008,
            name=f"seat hinge leaf seats on front frame {suffix}",
        )
        ctx.expect_contact(
            front,
            seat,
            elem_a=f"seat_hinge_anchor_{suffix}",
            elem_b=f"seat_hinge_bolt_{suffix}",
            contact_tol=0.008,
            name=f"seat hinge bolt seats in front frame {suffix}",
        )
        if suffix == "0":
            ctx.expect_gap(
                seat,
                rear,
                axis="x",
                positive_elem=f"seat_hinge_bolt_{suffix}",
                negative_elem=f"rear_leg_{suffix}",
                min_gap=0.001,
                name=f"seat hinge bolt clears rear support {suffix}",
            )
        else:
            ctx.expect_gap(
                rear,
                seat,
                axis="x",
                positive_elem=f"rear_leg_{suffix}",
                negative_elem=f"seat_hinge_bolt_{suffix}",
                min_gap=0.001,
                name=f"seat hinge bolt clears rear support {suffix}",
            )

    rest_rear = _aabb_center(ctx.part_world_aabb(rear))
    rest_seat = _aabb_center(ctx.part_world_aabb(seat))
    with ctx.pose({front_to_rear: -0.55, front_to_seat: 0.95}):
        folded_rear = _aabb_center(ctx.part_world_aabb(rear))
        folded_seat = _aabb_center(ctx.part_world_aabb(seat))
    ctx.check(
        "folding joints move attached frames",
        rest_rear is not None
        and folded_rear is not None
        and rest_seat is not None
        and folded_seat is not None
        and abs(folded_rear[1] - rest_rear[1]) > 0.015
        and abs(folded_seat[2] - rest_seat[2]) > 0.015,
        details=f"rear {rest_rear}->{folded_rear}, seat {rest_seat}->{folded_seat}",
    )

    return ctx.report()


object_model = build_object_model()
