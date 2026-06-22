"""Stage 1 — acoustic event detection (interchangeable detectors).

Each detector returns a list of ``Onset(time_s, score)`` with score in [0, 1].
Generic (no object assumptions). We compare:
  - ``onset_strength``  librosa spectral-flux onset envelope (broad, musical onsets)
  - ``spectral_flux``   half-wave-rectified flux with adaptive median threshold
  - ``hf_transient``    high-frequency energy derivative (sharp impacts/clicks)
  - ``combined``        union of the three with non-max suppression (best recall)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOP = 256
SR = 16000


@dataclass(frozen=True)
class Onset:
    time_s: float
    score: float          # 0..1, normalized within the clip
    detector: str


def _peaks(env: np.ndarray, times: np.ndarray, *, min_gap_s: float,
           height_pct: float, detector: str) -> list[Onset]:
    from scipy.signal import find_peaks

    if env.size == 0:
        return []
    distance = max(1, int(round(min_gap_s / (HOP / SR))))
    height = float(np.percentile(env, height_pct))
    prom = max(1e-6, float(np.std(env) * 0.5))
    idx, _ = find_peaks(env, distance=distance, height=height, prominence=prom)
    if idx.size == 0:
        return []
    e = env[idx]
    lo, hi = float(e.min()), float(e.max())
    norm = (e - lo) / (hi - lo + 1e-12)
    return [Onset(float(times[i]), float(s), detector) for i, s in zip(idx, norm)]


def detect_onset_strength(x: np.ndarray, *, min_gap_s: float = 0.12) -> list[Onset]:
    import librosa

    env = librosa.onset.onset_strength(y=x.astype(np.float32), sr=SR, hop_length=HOP)
    times = librosa.frames_to_time(np.arange(len(env)), sr=SR, hop_length=HOP)
    return _peaks(env, times, min_gap_s=min_gap_s, height_pct=70.0, detector="onset_strength")


def detect_spectral_flux(x: np.ndarray, *, min_gap_s: float = 0.12) -> list[Onset]:
    import librosa

    S = np.abs(librosa.stft(x.astype(np.float32), n_fft=1024, hop_length=HOP))
    logS = np.log1p(S)
    flux = np.maximum(0.0, np.diff(logS, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    # subtract a local median (adaptive threshold) to suppress sustained energy
    from scipy.ndimage import median_filter

    flux = np.maximum(0.0, flux - median_filter(flux, size=21))
    times = librosa.frames_to_time(np.arange(len(flux)), sr=SR, hop_length=HOP)
    return _peaks(flux, times, min_gap_s=min_gap_s, height_pct=75.0, detector="spectral_flux")


def detect_hf_transient(x: np.ndarray, *, min_gap_s: float = 0.12) -> list[Onset]:
    """High-frequency (>2 kHz) energy rise — sensitive to sharp clicks/impacts."""
    import librosa

    S = np.abs(librosa.stft(x.astype(np.float32), n_fft=1024, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    hf = S[freqs >= 2000.0].sum(axis=0)
    d = np.maximum(0.0, np.diff(hf))
    d = np.concatenate([[0.0], d])
    times = librosa.frames_to_time(np.arange(len(d)), sr=SR, hop_length=HOP)
    return _peaks(d, times, min_gap_s=min_gap_s, height_pct=80.0, detector="hf_transient")


DETECTORS = {
    "onset_strength": detect_onset_strength,
    "spectral_flux": detect_spectral_flux,
    "hf_transient": detect_hf_transient,
}


def detect_combined(x: np.ndarray, *, min_gap_s: float = 0.12) -> list[Onset]:
    """Union of all detectors with non-max suppression; score = mean of contributors."""
    allo = [o for fn in DETECTORS.values() for o in fn(x, min_gap_s=min_gap_s)]
    allo.sort(key=lambda o: o.time_s)
    merged: list[Onset] = []
    for o in allo:
        if merged and abs(o.time_s - merged[-1].time_s) < min_gap_s:
            prev = merged[-1]
            merged[-1] = Onset(
                time_s=(prev.time_s + o.time_s) / 2.0,
                score=max(prev.score, o.score),
                detector="combined",
            )
        else:
            merged.append(Onset(o.time_s, o.score, "combined"))
    return merged


def detect(x: np.ndarray, method: str = "combined", *, min_gap_s: float = 0.12) -> list[Onset]:
    if method == "combined":
        return detect_combined(x, min_gap_s=min_gap_s)
    if method in DETECTORS:
        return DETECTORS[method](x, min_gap_s=min_gap_s)
    raise ValueError(f"unknown detector {method!r}")
