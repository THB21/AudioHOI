from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite

from ..measurements.types import SourceRef


class AudioEventType(str, Enum):
    IMPACT = "impact"
    CONTACT_ONSET = "contact_onset"
    CONTACT_OFFSET = "contact_offset"
    SUSTAINED_MOTION = "sustained_motion"
    SILENCE = "silence"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AudioEvent:
    event_id: str
    sample_id: str
    frame: int
    peak_time_s: float
    event_type: AudioEventType
    confidence: float | None
    energy: float | None
    prominence: float | None
    source: SourceRef

    def __post_init__(self) -> None:
        if not self.event_id or not self.sample_id or self.frame < 1 or not isfinite(self.peak_time_s):
            raise ValueError("invalid audio event identity/frame/time")
        if self.confidence is not None and (not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("audio event confidence must be absent or within [0, 1]")
        if self.energy is not None and not isfinite(self.energy):
            raise ValueError("audio event energy must be finite when present")
        if self.prominence is not None and not isfinite(self.prominence):
            raise ValueError("audio event prominence must be finite when present")


def audio_event_record(event: AudioEvent) -> dict[str, object]:
    record = asdict(event)
    record["event_type"] = event.event_type.value
    return record
