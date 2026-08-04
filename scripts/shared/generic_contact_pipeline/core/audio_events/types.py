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
    MOTION_ONSET = "motion_onset"
    MOTION_OFFSET = "motion_offset"
    SHORT_TUG = "short_tug"
    SEAM_CLICK = "seam_click"
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
    start_frame: int | None = None
    end_frame: int | None = None
    start_time_s: float | None = None
    end_time_s: float | None = None
    snr: float | None = None
    band_profile: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.sample_id or self.frame < 1 or not isfinite(self.peak_time_s):
            raise ValueError("invalid audio event identity/frame/time")
        if self.confidence is not None and (not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("audio event confidence must be absent or within [0, 1]")
        if self.energy is not None and not isfinite(self.energy):
            raise ValueError("audio event energy must be finite when present")
        if self.prominence is not None and not isfinite(self.prominence):
            raise ValueError("audio event prominence must be finite when present")
        start_frame = self.frame if self.start_frame is None else self.start_frame
        end_frame = self.frame if self.end_frame is None else self.end_frame
        start_time = self.peak_time_s if self.start_time_s is None else self.start_time_s
        end_time = self.peak_time_s if self.end_time_s is None else self.end_time_s
        if start_frame < 1 or end_frame < start_frame:
            raise ValueError("audio event interval frames must be ordered and positive")
        if not isfinite(start_time) or not isfinite(end_time) or end_time < start_time:
            raise ValueError("audio event interval times must be finite and ordered")
        if self.snr is not None and not isfinite(self.snr):
            raise ValueError("audio event SNR must be finite when present")

    def contains_frame(self, frame: int) -> bool:
        start = self.start_frame if self.start_frame is not None else self.frame
        end = self.end_frame if self.end_frame is not None else self.frame
        return start <= frame <= end


def audio_event_record(event: AudioEvent) -> dict[str, object]:
    record = asdict(event)
    record["event_type"] = event.event_type.value
    return record
