#!/usr/bin/env python3
"""Audio-event semantics for contact-aware pose optimization.

Audio onsets tell us *when* the object touches; this module adds *what kind* of touch
it is, so the optimizer can apply the right physics per event instead of one global
setting. The classification is deliberately **zero-shot and rule-based** on generic
acoustic features (attack sharpness, decay time, spectral brightness) - no per-object
training - matching the project's object-agnostic direction.

Self-contained: features are recomputed from ``audio.wav`` with ``scipy``/``numpy`` only
(no librosa), so it works uniformly regardless of which detector wrote
``audio_events.csv`` (basketball stores only peak/prominence, football stores more).

Taxonomy -> physical effect on the depth solver:

    impact     sharp bright transient, short decay   -> hard contact: anchor + full kink
    bounce     impact-like but against the ground     -> kink only, do NOT pin to a body part
    placement  medium attack, duller, set-down thunk  -> anchor to support, partial kink
    scrape     long, low-attack, broadband friction   -> no anchor, little relaxation
    sustained  long decay / resonant hold             -> anchor to part, almost no kink
    unknown    everything else                        -> the pre-semantics default

Each event maps to:
    gamma          0..1  local acceleration-relaxation strength (1 = allow full kink)
    w_audio_scale  0..   multiplier on the soft audio contact pull
    promote_anchor bool  promote this onset to a hard contact anchor
    contact_target 'part' | 'support' | 'either'   what the object touches

``promote_anchor`` is only honoured by the solver when ``contact_target == 'part'`` - a
floor bounce (target 'support') therefore stops being falsely pinned to a hand/foot.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.io import wavfile
except Exception:  # pragma: no cover - scipy always present in the gvhmr env
    wavfile = None


# ---------------------------------------------------------------------------
# Event taxonomy + physics table
# ---------------------------------------------------------------------------

EVENT_TYPES = ("impact", "bounce", "placement", "scrape", "sustained", "unknown")


@dataclass(frozen=True)
class EventPhysics:
    gamma: float           # local acceleration-relaxation (0..1)
    w_audio_scale: float   # multiplier on the soft audio contact pull
    promote_anchor: bool   # promote onset to a hard contact anchor
    contact_target: str    # 'part' | 'support' | 'either'


# w_audio_scale is the strength of the soft pull toward the contacting *body part*; it
# is only meaningful when a body part is the contact (target 'part'), so floor bounces
# and scrapes set it to 0 and rely on the acceleration relaxation instead.
EVENT_PHYSICS: dict[str, EventPhysics] = {
    "impact":    EventPhysics(gamma=1.0, w_audio_scale=1.0, promote_anchor=True,  contact_target="part"),
    "bounce":    EventPhysics(gamma=1.0, w_audio_scale=0.0, promote_anchor=False, contact_target="support"),
    "placement": EventPhysics(gamma=0.7, w_audio_scale=0.6, promote_anchor=False, contact_target="support"),
    "scrape":    EventPhysics(gamma=0.2, w_audio_scale=0.0, promote_anchor=False, contact_target="either"),
    "sustained": EventPhysics(gamma=0.1, w_audio_scale=0.6, promote_anchor=True,  contact_target="part"),
    "unknown":   EventPhysics(gamma=0.8, w_audio_scale=1.0, promote_anchor=True,  contact_target="part"),
}


# ---------------------------------------------------------------------------
# Acoustic features (scipy/numpy, no librosa)
# ---------------------------------------------------------------------------

@dataclass
class OnsetFeatures:
    attack: float       # normalized energy rise into the onset (sharpness of attack)
    decay_ms: float     # time for post-peak energy to fall to 1/e (short = hard click)
    centroid: float     # spectral centroid in Hz around the onset (brightness)
    hf_ratio: float     # fraction of energy above 2 kHz (sharpness proxy)
    rms: float          # peak short-time RMS at the onset (loudness)


def _load_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(str(path))
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x / peak
    return int(sr), x


def compute_onset_features(wav_path: Path, onset_times_s: list[float]) -> list[OnsetFeatures]:
    """Per-onset acoustic features from the raw waveform.

    Window: 60 ms before to 200 ms after each onset. Attack = energy slope in the 30 ms
    leading into the peak; decay = 1/e fall time after the peak; spectral features from a
    50 ms window at the peak.
    """
    sr, x = _load_wav(wav_path)
    n = len(x)
    feats: list[OnsetFeatures] = []
    pre, post = int(0.060 * sr), int(0.200 * sr)
    win = int(0.050 * sr)
    # short-time energy envelope (1 ms hop)
    hop = max(1, int(0.001 * sr))
    frame = max(hop, int(0.010 * sr))
    for t in onset_times_s:
        c = int(round(t * sr))
        lo, hi = max(0, c - pre), min(n, c + post)
        seg = x[lo:hi]
        if seg.size < frame:
            feats.append(OnsetFeatures(0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        env = np.array([np.sqrt(np.mean(seg[i:i + frame] ** 2))
                        for i in range(0, seg.size - frame, hop)])
        if env.size == 0:
            feats.append(OnsetFeatures(0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        pk = int(np.argmax(env))
        rms = float(env[pk])
        # attack: rise over ~30 ms before the peak, normalized by peak energy
        a_lo = max(0, pk - int(0.030 * sr / hop))
        attack = float((env[pk] - env[a_lo]) / (rms + 1e-9))
        # decay: samples until env drops below rms/e after the peak
        thr = rms / np.e
        decay_idx = next((j for j in range(pk, env.size) if env[j] < thr), env.size - 1)
        decay_ms = float((decay_idx - pk) * hop / sr * 1000.0)
        # spectral features on a 50 ms window at the peak
        w_lo = max(0, c - win // 2)
        w = x[w_lo:w_lo + win]
        if w.size >= 16:
            mag = np.abs(np.fft.rfft(w * np.hanning(w.size)))
            freqs = np.fft.rfftfreq(w.size, 1.0 / sr)
            total = float(np.sum(mag) + 1e-9)
            centroid = float(np.sum(freqs * mag) / total)
            hf_ratio = float(np.sum(mag[freqs >= 2000.0]) / total)
        else:
            centroid, hf_ratio = 0.0, 0.0
        feats.append(OnsetFeatures(attack=attack, decay_ms=decay_ms,
                                   centroid=centroid, hf_ratio=hf_ratio, rms=rms))
    return feats


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_event(f: OnsetFeatures, *, periodic: bool = False) -> str:
    """Rule-based event type from acoustic features.

    Thresholds are generic and tunable; they encode the qualitative signature of each
    contact type rather than any learned/per-object model. ``periodic`` flags onsets
    that recur at a regular cadence (e.g. a dribble) -> bounce rather than one-off impact.
    """
    sharp = (f.attack >= 0.5) and (f.hf_ratio >= 0.20 or f.centroid >= 1500.0)
    short = f.decay_ms <= 120.0
    long_tail = f.decay_ms >= 300.0
    quiet = f.rms < 0.03

    if quiet:
        return "unknown"
    if sharp and short:
        return "bounce" if periodic else "impact"
    if short and not sharp:
        return "placement"
    if long_tail and f.attack < 0.4:
        return "scrape" if f.hf_ratio >= 0.15 else "sustained"
    return "unknown"


def _detect_periodic(onset_frames: list[int], tol: float = 0.25) -> bool:
    """True if onsets recur at a roughly constant interval (a bounce/dribble cadence)."""
    if len(onset_frames) < 4:
        return False
    gaps = np.diff(sorted(onset_frames)).astype(np.float64)
    if gaps.size == 0 or np.median(gaps) <= 0:
        return False
    return bool(np.std(gaps) / (np.median(gaps) + 1e-9) <= tol)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedEvent:
    frame: int
    time: float
    score: float
    event_type: str
    features: OnsetFeatures
    physics: EventPhysics


def classify_audio_events(wav_path: Path, audio_rows: list[dict]) -> list[ClassifiedEvent]:
    """Classify each audio onset and attach its physics. ``audio_rows`` are the dicts
    from the solver's ``read_audio_events`` (frame/time/score)."""
    if not audio_rows or wavfile is None or not Path(wav_path).exists():
        return []
    times = [float(r["audio_time"]) for r in audio_rows]
    frames = [int(r["audio_frame"]) for r in audio_rows]
    feats = compute_onset_features(Path(wav_path), times)
    periodic = _detect_periodic(frames)
    out: list[ClassifiedEvent] = []
    for r, f in zip(audio_rows, feats):
        etype = classify_event(f, periodic=periodic)
        out.append(ClassifiedEvent(
            frame=int(r["audio_frame"]), time=float(r["audio_time"]),
            score=float(r["audio_score"]), event_type=etype, features=f,
            physics=EVENT_PHYSICS[etype],
        ))
    return out


def build_semantic_fields(frames: np.ndarray, classified: list[ClassifiedEvent],
                          radius: int = 2) -> dict[str, np.ndarray]:
    """Per-frame semantic arrays aligned to ``frames``.

    Each event is spread over +-``radius`` frames with a triangular falloff (same as the
    audio-support builder). ``support`` max-combines; ``gamma``/``w_audio_scale`` take the
    value of the strongest contributing event; ``promote``/``target`` are tagged only on
    the exact onset frame (anchors are 1-frame, the spread feeds pull + relaxation).
    """
    n = len(frames)
    frame_to_idx = {int(fr): i for i, fr in enumerate(np.asarray(frames).tolist())}
    support = np.zeros(n)
    gamma = np.full(n, EVENT_PHYSICS["unknown"].gamma)
    w_scale = np.ones(n)
    best = np.zeros(n)  # strongest contributing weight per frame, for param selection
    promote = np.zeros(n, dtype=bool)
    target = np.array(["none"] * n, dtype=object)
    etype = np.array(["none"] * n, dtype=object)

    for ev in classified:
        for fr in range(ev.frame - radius, ev.frame + radius + 1):
            idx = frame_to_idx.get(fr)
            if idx is None:
                continue
            w = max(0.0, 1.0 - 0.25 * abs(fr - ev.frame)) * ev.score
            support[idx] = max(support[idx], w)
            if w > best[idx]:
                best[idx] = w
                gamma[idx] = ev.physics.gamma
                w_scale[idx] = ev.physics.w_audio_scale
        # anchor decision tagged on the exact onset frame only
        idx = frame_to_idx.get(ev.frame)
        if idx is not None:
            promote[idx] = ev.physics.promote_anchor and ev.physics.contact_target == "part"
            target[idx] = ev.physics.contact_target
            etype[idx] = ev.event_type
    return {"support": support, "gamma": gamma, "w_audio_scale": w_scale,
            "promote": promote, "target": target, "event_type": etype}


def write_semantics_csv(path: Path, classified: list[ClassifiedEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "time", "audio_score", "event_type", "gamma", "w_audio_scale",
              "promote_anchor", "contact_target", "attack", "decay_ms", "centroid",
              "hf_ratio", "rms"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for ev in classified:
            w.writerow({
                "frame": ev.frame, "time": f"{ev.time:.6f}", "audio_score": f"{ev.score:.6f}",
                "event_type": ev.event_type, "gamma": f"{ev.physics.gamma:.3f}",
                "w_audio_scale": f"{ev.physics.w_audio_scale:.3f}",
                "promote_anchor": int(ev.physics.promote_anchor),
                "contact_target": ev.physics.contact_target,
                "attack": f"{ev.features.attack:.4f}", "decay_ms": f"{ev.features.decay_ms:.2f}",
                "centroid": f"{ev.features.centroid:.1f}", "hf_ratio": f"{ev.features.hf_ratio:.4f}",
                "rms": f"{ev.features.rms:.4f}",
            })


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Classify audio events and write semantics CSV.")
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--audio-events-csv", type=Path, default=None)
    ap.add_argument("--audio-wav", type=Path, default=None)
    args = ap.parse_args()
    events_csv = args.audio_events_csv or (args.sample_dir / "results" / "events" / "audio_events.csv")
    wav = args.audio_wav or (args.sample_dir / "audio.wav")
    rows = []
    if events_csv.exists():
        with events_csv.open() as f:
            recs = list(csv.DictReader(f))
        proms = [float(r.get("prominence", 0.0) or 0.0) for r in recs]
        pmax = max(proms) if proms else 0.0
        for r, prom in zip(recs, proms):
            if not r.get("audio_frame"):
                continue
            score = float(r["audio_score"]) if r.get("audio_score") else (prom / pmax if pmax > 0 else 0.0)
            rows.append({"audio_frame": int(float(r["audio_frame"])),
                         "audio_time": float(r.get("audio_time", 0.0) or 0.0),
                         "audio_score": float(np.clip(score, 0.0, 1.0))})
    classified = classify_audio_events(wav, rows)
    out = args.sample_dir / "results" / "events" / "audio_semantics.csv"
    write_semantics_csv(out, classified)
    from collections import Counter
    print(f"classified {len(classified)} events -> {out}")
    print("type counts:", dict(Counter(e.event_type for e in classified)))
