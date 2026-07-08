from __future__ import annotations

import math

from sdk import ArticulatedObject, Box, Cylinder, Inertial, Origin, Sphere, TestContext, TestReport


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="wooden_bo_staff")

    warm_wood = model.material("warm_wood", rgba=(0.63, 0.36, 0.16, 1.0))
    end_grain = model.material("end_grain", rgba=(0.48, 0.27, 0.12, 1.0))
    dark_wrap = model.material("dark_grip_wrap", rgba=(0.075, 0.052, 0.035, 1.0))
    fine_line = model.material("fine_wood_line", rgba=(0.34, 0.19, 0.08, 1.0))

    staff = model.part("staff")
    staff.visual(
        Cylinder(radius=0.018, length=1.720),
        origin=Origin(xyz=(0.0, 0.0, 0.0), rpy=(0.0, math.pi / 2.0, 0.0)),
        material=warm_wood,
        name="shaft",
    )

    for x, name in [(-0.890, "tip_region_0"), (0.890, "tip_region_1")]:
        staff.visual(
            Cylinder(radius=0.017, length=0.080),
            origin=Origin(xyz=(x, 0.0, 0.0), rpy=(0.0, math.pi / 2.0, 0.0)),
            material=end_grain,
            name=name,
        )
        staff.visual(
            Sphere(radius=0.017),
            origin=Origin(xyz=(x + (-0.040 if x < 0.0 else 0.040), 0.0, 0.0)),
            material=end_grain,
            name=f"{name}_rounded_end",
        )

    for x, name in [(-0.115, "grip_band_0"), (0.115, "grip_band_1")]:
        staff.visual(
            Cylinder(radius=0.0205, length=0.095),
            origin=Origin(xyz=(x, 0.0, 0.0), rpy=(0.0, math.pi / 2.0, 0.0)),
            material=dark_wrap,
            name=name,
        )

    for x in (-0.520, -0.260, 0.260, 0.520):
        staff.visual(
            Cylinder(radius=0.0185, length=0.012),
            origin=Origin(xyz=(x, 0.0, 0.0), rpy=(0.0, math.pi / 2.0, 0.0)),
            material=fine_line,
            name=f"wood_grain_ring_{int((x + 1.0) * 1000)}",
        )

    staff.inertial = Inertial.from_geometry(
        Box((1.84, 0.042, 0.042)),
        mass=0.72,
        origin=Origin(xyz=(0.0, 0.0, 0.0)),
    )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    staff = object_model.get_part("staff")
    visual_names = {visual.name for visual in staff.visuals}
    ctx.check("single rigid staff link", len(object_model.parts) == 1, "The staff asset should be one connected rigid link.")
    ctx.check("long shaft visual present", "shaft" in visual_names, "The main wooden shaft visual is missing.")
    ctx.check(
        "two grip bands present",
        {"grip_band_0", "grip_band_1"}.issubset(visual_names),
        "The darker central grip/contact bands are missing.",
    )
    ctx.check(
        "two tip regions present",
        {"tip_region_0", "tip_region_1"}.issubset(visual_names),
        "The semantic floor/contact tip regions are missing.",
    )
    ctx.check(
        "staff has realistic length scale",
        staff.inertial is not None and staff.inertial.mass > 0.5,
        "The staff should carry plausible mass/inertial metadata for a wooden bo staff.",
    )
    return ctx.report()


object_model = build_object_model()
