from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from ..contact_constraints.types import HumanSite
from ..measurements.types import CoordinateFrame, SourceRef


@dataclass(frozen=True)
class HumanSiteMeasurement:
    measurement_id: str
    sample_id: str
    frame: int
    time: float
    site: HumanSite
    xyz_m: tuple[float, float, float]
    coordinate_frame: CoordinateFrame
    confidence: float | None
    source: SourceRef

    def __post_init__(self) -> None:
        if not self.measurement_id or not self.sample_id or self.frame < 1 or not isfinite(self.time):
            raise ValueError("invalid human-site measurement identity/frame/time")
        if not all(isfinite(value) for value in self.xyz_m):
            raise ValueError("human-site coordinates must be finite")
        if self.coordinate_frame != CoordinateFrame.CAMERA_METERS:
            raise ValueError("human-site measurements must use camera_meters")
        if self.confidence is not None and (not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("human-site confidence must be absent or within [0, 1]")


def human_site_record(measurement: HumanSiteMeasurement) -> dict[str, object]:
    return asdict(measurement)
