from __future__ import annotations

import cadquery as cq

from sdk import (
    ArticulatedObject,
    ArticulationType,
    Box,
    Material,
    Origin,
    TestContext,
    TestReport,
    mesh_from_cadquery,
)


OUTER_RADIUS = 0.040
INNER_RADIUS = 0.0355
BOTTOM_HEIGHT = 0.006
BODY_HEIGHT = 0.095
RIM_HEIGHT = 0.007
RIM_OUTER_RADIUS = 0.044
RIM_INNER_RADIUS = 0.0335
HANDLE_EMBED = 0.002


def _ring(name: str, outer_radius: float, inner_radius: float, z0: float, height: float):
    solid = (
        cq.Workplane("XY")
        .circle(outer_radius)
        .circle(inner_radius)
        .extrude(height)
        .translate((0.0, 0.0, z0))
    )
    return mesh_from_cadquery(solid, name)


def _disk(name: str, radius: float, z0: float, height: float):
    solid = cq.Workplane("XY").circle(radius).extrude(height).translate((0.0, 0.0, z0))
    return mesh_from_cadquery(solid, name)


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="mug_contact_proxy")

    ceramic = Material(name="warm_white_ceramic", rgba=(0.92, 0.90, 0.84, 1.0))
    rim_ceramic = Material(name="slightly_lighter_rim", rgba=(0.98, 0.97, 0.93, 1.0))
    underside = Material(name="unglazed_bottom", rgba=(0.74, 0.70, 0.62, 1.0))

    body = model.part("body")
    body.visual(
        _ring("body_shell", OUTER_RADIUS, INNER_RADIUS, BOTTOM_HEIGHT, BODY_HEIGHT - BOTTOM_HEIGHT),
        name="body_shell",
        material=ceramic,
    )

    bottom = model.part("bottom")
    bottom.visual(
        _disk("bottom_disk", OUTER_RADIUS, 0.0, BOTTOM_HEIGHT),
        name="bottom_disk",
        material=underside,
    )

    rim = model.part("rim")
    rim.visual(
        _ring("rim_lip", RIM_OUTER_RADIUS, RIM_INNER_RADIUS, BODY_HEIGHT, RIM_HEIGHT),
        name="rim_lip",
        material=rim_ceramic,
    )

    handle = model.part("handle")
    tube_y = 0.014
    tube_z = 0.010
    upper_z = 0.071
    lower_z = 0.035
    bridge_x_min = OUTER_RADIUS - HANDLE_EMBED
    bridge_x_max = 0.083
    bridge_len = bridge_x_max - bridge_x_min
    bridge_x = bridge_x_min + bridge_len / 2.0
    grip_x = 0.083
    grip_height = (upper_z - lower_z) + tube_z
    grip_center_z = (upper_z + lower_z) / 2.0

    handle.visual(
        Box((bridge_len, tube_y, tube_z)),
        origin=Origin(xyz=(bridge_x, 0.0, upper_z)),
        name="upper_mount",
        material=ceramic,
    )
    handle.visual(
        Box((bridge_len, tube_y, tube_z)),
        origin=Origin(xyz=(bridge_x, 0.0, lower_z)),
        name="lower_mount",
        material=ceramic,
    )
    handle.visual(
        Box((0.012, tube_y, grip_height)),
        origin=Origin(xyz=(grip_x, 0.0, grip_center_z)),
        name="outer_grip",
        material=ceramic,
    )

    for child in (bottom, rim, handle):
        model.articulation(
            f"body_to_{child.name}",
            ArticulationType.FIXED,
            parent=body,
            child=child,
            origin=Origin(),
        )

    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)

    body = object_model.get_part("body")
    bottom = object_model.get_part("bottom")
    rim = object_model.get_part("rim")
    handle = object_model.get_part("handle")

    ctx.allow_overlap(
        handle,
        body,
        elem_a="upper_mount",
        elem_b="body_shell",
        reason="The upper handle mount is intentionally seated a few millimeters into the cup wall.",
    )
    ctx.allow_overlap(
        handle,
        body,
        elem_a="lower_mount",
        elem_b="body_shell",
        reason="The lower handle mount is intentionally seated a few millimeters into the cup wall.",
    )

    missing = []
    for part_name in ("body", "handle", "rim", "bottom"):
        try:
            object_model.get_part(part_name)
        except Exception:
            missing.append(part_name)
    ctx.check("semantic mug parts exist", not missing, details=f"missing={missing}")

    ctx.expect_gap(
        body,
        bottom,
        axis="z",
        max_gap=0.001,
        max_penetration=0.0,
        positive_elem="body_shell",
        negative_elem="bottom_disk",
        name="bottom disk seats under cup body",
    )
    ctx.expect_overlap(
        bottom,
        body,
        axes="xy",
        min_overlap=0.070,
        elem_a="bottom_disk",
        elem_b="body_shell",
        name="bottom footprint supports cylindrical body",
    )
    ctx.expect_gap(
        rim,
        body,
        axis="z",
        max_gap=0.001,
        max_penetration=0.0,
        positive_elem="rim_lip",
        negative_elem="body_shell",
        name="rim sits on top of body shell",
    )
    ctx.expect_overlap(
        rim,
        body,
        axes="xy",
        min_overlap=0.070,
        elem_a="rim_lip",
        elem_b="body_shell",
        name="rim follows cylindrical body footprint",
    )

    for elem_name, check_name in (
        ("upper_mount", "upper handle mount is seated in side wall"),
        ("lower_mount", "lower handle mount is seated in side wall"),
    ):
        ctx.expect_gap(
            handle,
            body,
            axis="x",
            max_gap=0.001,
            max_penetration=0.004,
            positive_elem=elem_name,
            negative_elem="body_shell",
            name=check_name,
        )
        ctx.expect_overlap(
            handle,
            body,
            axes="yz",
            min_overlap=0.008,
            elem_a=elem_name,
            elem_b="body_shell",
            name=f"{elem_name} overlaps cup side projection",
        )

    return ctx.report()


object_model = build_object_model()
