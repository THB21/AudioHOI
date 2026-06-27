from __future__ import annotations

K_FULLIMG = {
    "fx": 1468.604736328125,
    "fy": 1468.604736328125,
    "cx": 640.0,
    "cy": 360.0,
}


def backproject_uvz(u: float, v: float, z: float, k: dict[str, float] | None = None) -> tuple[float, float, float]:
    k = k or K_FULLIMG
    x = (u - k["cx"]) * z / k["fx"]
    y = (v - k["cy"]) * z / k["fy"]
    return x, y, z
