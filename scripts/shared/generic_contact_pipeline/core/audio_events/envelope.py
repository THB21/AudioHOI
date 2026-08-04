from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


@dataclass(frozen=True)
class AudioEnvelopeConfig:
    window_ms: float = 80.0
    hop_ms: float = 1000.0 / 24.0
    motion_on_z: float = 0.55
    motion_off_z: float = -0.10
    min_motion_ms: float = 150.0
    min_silence_ms: float = 250.0
    impulse_z: float = 3.0

    def __post_init__(self) -> None:
        if self.window_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("audio envelope window and hop must be positive")
        if self.motion_on_z <= self.motion_off_z:
            raise ValueError("motion-on threshold must exceed motion-off threshold")
        if self.min_motion_ms <= 0 or self.min_silence_ms <= 0:
            raise ValueError("minimum interval durations must be positive")


def _read_pcm_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    dtype = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}.get(width)
    if dtype is None:
        raise ValueError(f"unsupported PCM sample width: {width}")
    values = np.frombuffer(frames, dtype=dtype).astype(np.float64)
    if width == 1:
        values = values - 128.0
        scale = 128.0
    else:
        scale = float(2 ** (8 * width - 1))
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return rate, values / scale


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if scale < 1e-9:
        scale = float(np.std(values))
    return (values - median) / max(scale, 1e-9)


def _intervals(mask: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    padded = np.r_[False, mask.astype(bool), False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [
        (int(start), int(end - 1))
        for start, end in zip(edges[0::2], edges[1::2])
        if end - start >= minimum_frames
    ]


def _hysteresis(values: np.ndarray, on: float, off: float) -> np.ndarray:
    active = False
    result = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        if not active and value >= on:
            active = True
        elif active and value <= off:
            active = False
        result[index] = active
    return result


def extract_audio_evidence(
    audio_path: Path,
    fps: float,
    config: AudioEnvelopeConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return typed event rows and one envelope row per video frame."""

    if fps <= 0:
        raise ValueError("video FPS must be positive")
    rate, samples = _read_pcm_mono(audio_path)
    # Audio containers commonly carry a few milliseconds of encoder padding.
    # Map duration to the nearest video-frame count so sub-half-frame padding
    # does not create an impossible extra object frame.
    frame_count = max(1, int(round(len(samples) / rate * fps)))
    half_window = max(1, int(round(config.window_ms * rate / 2000.0)))
    rms = np.zeros(frame_count, dtype=np.float64)
    flux = np.zeros(frame_count, dtype=np.float64)
    hf_ratio = np.zeros(frame_count, dtype=np.float64)
    previous_spectrum: np.ndarray | None = None
    for index in range(frame_count):
        center = int(round((index + 0.5) * rate / fps))
        start, end = max(0, center - half_window), min(len(samples), center + half_window)
        window = samples[start:end]
        if not len(window):
            continue
        rms[index] = float(np.sqrt(np.mean(window * window) + 1e-12))
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        total = float(spectrum.sum()) + 1e-12
        split = int(round(0.35 * len(spectrum)))
        hf_ratio[index] = float(spectrum[split:].sum() / total)
        if previous_spectrum is not None:
            common = min(len(previous_spectrum), len(spectrum))
            flux[index] = float(np.maximum(0.0, spectrum[:common] - previous_spectrum[:common]).sum() / total)
        previous_spectrum = spectrum

    rms_z = _robust_z(np.log(rms + 1e-8))
    flux_z = _robust_z(flux)
    motion_score = 0.78 * rms_z + 0.22 * flux_z
    motion_probability = 1.0 / (1.0 + np.exp(-np.clip(motion_score, -30.0, 30.0)))
    motion_mask = _hysteresis(motion_score, config.motion_on_z, config.motion_off_z)
    min_motion = max(1, int(np.ceil(config.min_motion_ms * fps / 1000.0)))
    min_silence = max(1, int(np.ceil(config.min_silence_ms * fps / 1000.0)))
    motion_intervals = _intervals(motion_mask, min_motion)
    silence_intervals = _intervals(~motion_mask, min_silence)

    rows: list[dict[str, object]] = []

    def add_interval(kind: str, start: int, end: int, confidence: float) -> None:
        peak_index = start + int(np.argmax(motion_score[start : end + 1]))
        rows.append(
            {
                "event": f"{kind}_{len(rows) + 1:04d}",
                "event_type": kind,
                "audio_time": (peak_index + 0.5) / fps,
                "audio_frame": peak_index + 1,
                "start_time_s": start / fps,
                "end_time_s": (end + 1) / fps,
                "start_frame": start + 1,
                "end_frame": end + 1,
                "peak": float(rms[peak_index]),
                "prominence": float(max(motion_score[peak_index], 0.0)),
                "rms_rise": float(max(rms_z[peak_index], 0.0)),
                "sharpness": float(hf_ratio[peak_index]),
                "audio_score": float(np.clip(confidence, 0.0, 1.0)),
                "snr": float(rms_z[peak_index]),
                "band_profile": "broadband" if hf_ratio[peak_index] > 0.30 else "low_mid",
                "detector": "robust_interval_envelope_v2",
                "source": str(audio_path),
            }
        )

    for start, end in motion_intervals:
        duration_ms = (end - start + 1) * 1000.0 / fps
        kind = "short_tug" if duration_ms < max(450.0, 2.0 * config.min_motion_ms) else "sustained_motion"
        confidence = float(np.mean(motion_probability[start : end + 1]))
        add_interval(kind, start, end, confidence)
        add_interval("motion_onset", start, start, confidence)
        add_interval("motion_offset", end, end, confidence)
    for start, end in silence_intervals:
        confidence = float(np.mean(1.0 - motion_probability[start : end + 1]))
        add_interval("silence", start, end, confidence)

    impulse_candidates = (flux_z >= config.impulse_z) & motion_mask
    for start, end in _intervals(impulse_candidates, 1):
        peak = start + int(np.argmax(flux_z[start : end + 1]))
        add_interval("seam_click", peak, peak, float(motion_probability[peak]))

    envelope = [
        {
            "frame": index + 1,
            "time_s": (index + 0.5) / fps,
            "rms_z": float(rms_z[index]),
            "flux_z": float(flux_z[index]),
            "hf_ratio": float(hf_ratio[index]),
            "motion_probability": float(motion_probability[index]),
            "source": str(audio_path),
        }
        for index in range(frame_count)
    ]
    rows.sort(key=lambda row: (int(row["start_frame"]), str(row["event_type"])))
    return rows, envelope
