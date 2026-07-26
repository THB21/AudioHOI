from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..measurements.types import FeatureRef, SourceRef
from .types import ContactMode, FrameInterval, HumanSite


@dataclass(frozen=True)
class ContactEventConstraint:
    event_id: str
    sample_id: str
    peak_frame: int
    interval: FrameInterval
    peak_time: float
    human_site: HumanSite
    object_feature: FeatureRef
    mode: ContactMode
    confidence: float | None
    source: SourceRef

    def __post_init__(self) -> None:
        if not self.event_id or not self.sample_id or self.peak_frame < 1 or not isfinite(self.peak_time):
            raise ValueError("invalid contact event identity/frame/time")
        if not self.interval.start_frame <= self.peak_frame <= self.interval.end_frame:
            raise ValueError("contact event peak must lie inside its interval")
        if self.confidence is not None and (not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("contact event confidence must be absent or within [0, 1]")


@dataclass(frozen=True)
class ContactStateSample:
    sample_id: str
    frame: int
    time: float
    human_active: bool
    support_active: bool
    human_site: HumanSite
    contact_depth_offset_m: float
    object_u: float | None
    object_v: float | None
    confidence: float | None
    source: SourceRef

    def __post_init__(self) -> None:
        if not self.sample_id or self.frame < 1 or not isfinite(self.time) or not isfinite(self.contact_depth_offset_m):
            raise ValueError("invalid contact-state sample")
        if self.object_u is not None and not isfinite(self.object_u):
            raise ValueError("contact object_u must be finite or absent")
        if self.object_v is not None and not isfinite(self.object_v):
            raise ValueError("contact object_v must be finite or absent")
        if self.confidence is not None and (not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("contact-state confidence must be absent or within [0, 1]")


def _site(value: str) -> HumanSite:
    clean = value.strip().lower()
    if clean == "floor" or clean in {"unknown_plane", "plane", "support"}:
        return HumanSite("environment", "none")
    if "_" in clean:
        side, body_part = clean.split("_", 1)
        if side in {"left", "right"}:
            return HumanSite(body_part, side)
    return HumanSite(clean or "unknown", "unknown")


def _number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return None if value in {"", None} else float(value)


def adapt_contact_event_rows(
    sample_id: str,
    rows: list[dict[str, str]],
    artifact: str,
) -> tuple[ContactEventConstraint, ...]:
    events: list[ContactEventConstraint] = []
    for ordinal, row in enumerate(rows):
        event_type = row.get("contact_type", "")
        target = row.get("target", "")
        peak = int(row["frame"])
        is_support = event_type in {"floor_contact_event", "plane_support_contact_event"} or target in {"floor", "unknown_plane"}
        site = HumanSite("environment", "none") if is_support else _site(target)
        mode = ContactMode.SUPPORT if is_support else (ContactMode.IMPACT if site.body_part == "foot" else ContactMode.GRASP)
        score = _number(row, "score")
        confidence = min(1.0, max(0.0, score)) if score is not None else None
        events.append(
            ContactEventConstraint(
                event_id=f"{sample_id}:{peak}:event:{ordinal}",
                sample_id=sample_id,
                peak_frame=peak,
                interval=FrameInterval(int(row.get("window_start", peak) or peak), int(row.get("window_end", peak) or peak)),
                peak_time=float(row["time"]),
                human_site=site,
                object_feature=FeatureRef("support_point" if is_support else "object_surface", "object:support" if is_support else "object:surface"),
                mode=mode,
                confidence=confidence,
                source=SourceRef(
                    artifact,
                    tuple(field for field in ("frame", "window_start", "window_end", "time", "contact_type", "target", "score", "confidence_level") if field in row),
                    producer="contact_event_interval_adapter",
                ),
            )
        )
    return tuple(events)


def adapt_contact_state_rows(
    sample_id: str,
    rows: list[dict[str, str]],
    artifact: str,
) -> tuple[ContactStateSample, ...]:
    states: list[ContactStateSample] = []
    for row in rows:
        label = row.get("contact_label", "") or "unknown"
        confidence = _number(row, "support_conf")
        if confidence is None:
            confidence = _number(row, "anchor_score")
        if confidence is not None:
            confidence = min(1.0, max(0.0, confidence))
        states.append(
            ContactStateSample(
                sample_id=sample_id,
                frame=int(row["frame"]),
                time=float(row["time"]),
                human_active=int(float(row.get("human_contact_state", row.get("anchor_contact_state", "0")) or 0)) == 1,
                support_active=int(float(row.get("floor_contact_state", row.get("plane_support_state", "0")) or 0)) == 1,
                human_site=_site(label),
                contact_depth_offset_m=float(row.get("contact_depth_offset_m", "0") or 0),
                object_u=_number(row, "active_object_u"),
                object_v=_number(row, "active_object_v"),
                confidence=confidence,
                source=SourceRef(
                    artifact,
                    tuple(field for field in ("frame", "time", "human_contact_state", "floor_contact_state", "contact_label", "contact_depth_offset_m", "active_object_u", "active_object_v", "anchor_score", "support_conf") if field in row),
                    producer="contact_state_timeline_adapter",
                ),
            )
        )
    return tuple(states)
