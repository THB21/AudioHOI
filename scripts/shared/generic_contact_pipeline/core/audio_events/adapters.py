from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import csv
from pathlib import Path

from ..measurements.types import SourceRef
from .types import AudioEvent, AudioEventType


@dataclass(frozen=True)
class AudioEventAdaptationResult:
    schema: str
    events: tuple[AudioEvent, ...]
    mapped_fields: tuple[str, ...]
    unmapped_nonempty_fields: tuple[str, ...]


def resolve_audio_event_artifact(result_dir: Path) -> Path | None:
    for path in (
        result_dir / "contact_candidates_internal/audio_events.csv",
        result_dir / "events/audio_events.csv",
        result_dir.parent / "events/audio_events.csv",
    ):
        if path.is_file():
            return path
    return None


def load_audio_events(sample_id: str, result_dir: Path) -> AudioEventAdaptationResult:
    path = resolve_audio_event_artifact(result_dir)
    if path is None:
        return AudioEventAdaptationResult("audio_peak_events_v1", (), (), ())
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return adapt_audio_event_rows(sample_id, rows, str(path))


def _number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in {"", None}:
        return None
    parsed = float(value)
    return parsed if isfinite(parsed) else None


def adapt_audio_event_rows(
    sample_id: str,
    rows: list[dict[str, str]],
    artifact: str,
) -> AudioEventAdaptationResult:
    """Adapt detector events without inventing semantic event labels."""

    events: list[AudioEvent] = []
    v2 = bool(rows and {"event_type", "start_frame", "end_frame"}.issubset(rows[0]))
    mapped = {
        "event", "audio_time", "audio_frame", "peak", "prominence", "audio_score",
        "event_type", "start_frame", "end_frame", "start_time_s", "end_time_s",
        "snr", "band_profile", "detector", "source",
    }
    for index, row in enumerate(rows):
        time_s = _number(row, "audio_time")
        frame = _number(row, "audio_frame")
        if time_s is None or frame is None:
            continue
        raw_type = str(row.get("event_type", "unknown")).strip() if v2 else "unknown"
        try:
            event_type = AudioEventType(raw_type or "unknown")
        except ValueError as exc:
            raise ValueError(f"unknown audio event type {raw_type!r} at row {index + 1}") from exc
        events.append(
            AudioEvent(
                event_id=str(row.get("event", "")).strip() or f"audio_event_{index + 1}",
                sample_id=sample_id,
                frame=int(frame),
                peak_time_s=time_s,
                event_type=event_type,
                confidence=_number(row, "audio_score"),
                energy=_number(row, "peak"),
                prominence=_number(row, "prominence"),
                source=SourceRef(
                    artifact,
                    tuple(sorted(mapped & set(row))),
                    producer="audio_event_detector_adapter",
                ),
                start_frame=int(_number(row, "start_frame") or frame),
                end_frame=int(_number(row, "end_frame") or frame),
                start_time_s=_number(row, "start_time_s") if v2 else time_s,
                end_time_s=_number(row, "end_time_s") if v2 else time_s,
                snr=_number(row, "snr"),
                band_profile=str(row.get("band_profile", "")).strip() or None,
            )
        )
    nonempty = {
        field
        for field in (rows[0] if rows else {})
        if any(row.get(field, "") not in {"", None} for row in rows)
    }
    return AudioEventAdaptationResult(
        schema="audio_events_v2" if v2 else "audio_peak_events_v1",
        events=tuple(events),
        mapped_fields=tuple(sorted(mapped & nonempty)),
        unmapped_nonempty_fields=tuple(sorted(nonempty - mapped)),
    )
