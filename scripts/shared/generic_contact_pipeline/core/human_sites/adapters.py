from __future__ import annotations

from dataclasses import dataclass

from ..contact_constraints.types import HumanSite
from ..measurements.types import CoordinateFrame, SourceRef
from .types import HumanSiteMeasurement


@dataclass(frozen=True)
class HumanSiteAdaptationResult:
    schema: str
    measurements: tuple[HumanSiteMeasurement, ...]
    mapped_fields: tuple[str, ...]
    unmapped_nonempty_fields: tuple[str, ...]


def adapt_human_site_rows(
    sample_id: str,
    rows: list[dict[str, str]],
    artifact: str,
) -> HumanSiteAdaptationResult:
    if not rows:
        raise ValueError("cannot adapt an empty human-site table")
    required = {"frame", "time", "site_id", "body_part", "side", "x_m", "y_m", "z_m", "coordinate_frame"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"human-site table is missing fields: {sorted(missing)}")
    measurements: list[HumanSiteMeasurement] = []
    for row in rows:
        if row.get("coordinate_frame") not in {"gvhmr_incam", "camera_meters"}:
            raise ValueError(f"unsupported human-site coordinate frame: {row.get('coordinate_frame')!r}")
        confidence = float(row["confidence"]) if row.get("confidence", "") not in {"", None} else None
        frame = int(row["frame"])
        site_id = row["site_id"]
        measurements.append(
            HumanSiteMeasurement(
                measurement_id=f"{sample_id}:{frame}:human_site:{site_id}",
                sample_id=sample_id,
                frame=frame,
                time=float(row["time"]),
                site=HumanSite(row["body_part"], row["side"]),
                xyz_m=(float(row["x_m"]), float(row["y_m"]), float(row["z_m"])),
                coordinate_frame=CoordinateFrame.CAMERA_METERS,
                confidence=confidence,
                source=SourceRef(
                    artifact,
                    ("site_id", "body_part", "side", "x_m", "y_m", "z_m", "coordinate_frame", "confidence", "source"),
                    producer=row.get("source", "gvhmr_human_site_adapter") or "gvhmr_human_site_adapter",
                ),
            )
        )
    mapped = required | {"confidence", "source"}
    nonempty = {field for field in rows[0] if any(row.get(field, "") not in {"", None} for row in rows)}
    return HumanSiteAdaptationResult(
        schema="human_site_xyz_v1",
        measurements=tuple(measurements),
        mapped_fields=tuple(sorted(mapped)),
        unmapped_nonempty_fields=tuple(sorted(nonempty - mapped)),
    )
