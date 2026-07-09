"""Stage 2 — per-event acoustic feature vector.

Richer than the original ``audio_semantics.OnsetFeatures``: adds spectral bandwidth,
flatness (noisiness), zero-crossing rate, harmonic/percussive ratio and 4 MFCC summaries
on top of attack / decay / centroid / hf_ratio / rms. These feed both the rule classifier
and the unsupervised clusterer, and are written verbatim into the semantic sheet.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields

import numpy as np

SR = 16000


@dataclass
class EventFeatures:
    attack: float        # normalized energy rise into the onset (attack sharpness)
    decay_ms: float      # 1/e post-peak fall time (short = hard click)
    centroid: float      # spectral centroid Hz (brightness)
    bandwidth: float     # spectral bandwidth Hz (spread)
    flatness: float      # spectral flatness 0..1 (1 = noise-like / broadband friction)
    hf_ratio: float      # energy fraction >2 kHz (sharpness proxy)
    zcr: float           # zero-crossing rate (noisiness)
    rms: float           # peak short-time RMS (loudness)
    hpr: float           # harmonic/(harmonic+percussive) energy ratio 0..1
    mfcc1: float
    mfcc2: float
    mfcc3: float
    mfcc4: float

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

    @staticmethod
    def names() -> list[str]:
        return [f.name for f in fields(EventFeatures)]

    def as_dict(self) -> dict:
        return asdict(self)


_ZERO = EventFeatures(*([0.0] * len(EventFeatures.names())))


def compute_features(x: np.ndarray, onset_times: list[float], sr: int = SR) -> list[EventFeatures]:
    import librosa

    n = len(x)
    pre, post = int(0.060 * sr), int(0.200 * sr)
    win = int(0.050 * sr)
    hop = max(1, int(0.001 * sr))      # 1 ms envelope hop
    frame = max(hop, int(0.010 * sr))  # 10 ms envelope frame
    out: list[EventFeatures] = []

    for t in onset_times:
        c = int(round(t * sr))
        lo, hi = max(0, c - pre), min(n, c + post)
        seg = x[lo:hi]
        if seg.size < frame:
            out.append(_ZERO)
            continue
        env = np.array([np.sqrt(np.mean(seg[i:i + frame] ** 2))
                        for i in range(0, seg.size - frame, hop)])
        if env.size == 0:
            out.append(_ZERO)
            continue
        pk = int(np.argmax(env))
        rms = float(env[pk])
        a_lo = max(0, pk - int(0.030 * sr / hop))
        attack = float((env[pk] - env[a_lo]) / (rms + 1e-9))
        thr = rms / np.e
        decay_idx = next((j for j in range(pk, env.size) if env[j] < thr), env.size - 1)
        decay_ms = float((decay_idx - pk) * hop / sr * 1000.0)

        w_lo = max(0, c - win // 2)
        w = x[w_lo:w_lo + win].astype(np.float32)
        if w.size >= 64:
            mag = np.abs(np.fft.rfft(w * np.hanning(w.size)))
            freqs = np.fft.rfftfreq(w.size, 1.0 / sr)
            total = float(np.sum(mag) + 1e-9)
            centroid = float(np.sum(freqs * mag) / total)
            bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / total))
            hf_ratio = float(np.sum(mag[freqs >= 2000.0]) / total)
            flatness = float(np.exp(np.mean(np.log(mag + 1e-9))) / (np.mean(mag) + 1e-9))
            zcr = float(np.mean(np.abs(np.diff(np.sign(w)))) / 2.0)
            try:
                mfcc = librosa.feature.mfcc(y=w, sr=sr, n_mfcc=5, n_fft=min(512, w.size))
                m = mfcc[1:5].mean(axis=1)
            except Exception:
                m = np.zeros(4)
            try:
                har, per = librosa.effects.hpss(w)
                he, pe = float(np.sum(har ** 2)), float(np.sum(per ** 2))
                hpr = he / (he + pe + 1e-9)
            except Exception:
                hpr = 0.0
        else:
            centroid = bandwidth = hf_ratio = flatness = zcr = hpr = 0.0
            m = np.zeros(4)
        m = np.nan_to_num(np.asarray(m, dtype=float))
        vals = np.nan_to_num(np.array([attack, decay_ms, centroid, bandwidth, flatness,
                                       hf_ratio, zcr, rms, hpr]))
        out.append(EventFeatures(
            attack=vals[0], decay_ms=vals[1], centroid=vals[2], bandwidth=vals[3],
            flatness=vals[4], hf_ratio=vals[5], zcr=vals[6], rms=vals[7], hpr=vals[8],
            mfcc1=float(m[0]), mfcc2=float(m[1]), mfcc3=float(m[2]), mfcc4=float(m[3]),
        ))
    return out


def detect_periodic(onset_frames: list[int], tol: float = 0.25) -> bool:
    """True if onsets recur at a roughly constant cadence (dribble/bounce)."""
    if len(onset_frames) < 4:
        return False
    gaps = np.diff(sorted(onset_frames)).astype(np.float64)
    if gaps.size == 0 or np.median(gaps) <= 0:
        return False
    return bool(np.std(gaps) / (np.median(gaps) + 1e-9) <= tol)
