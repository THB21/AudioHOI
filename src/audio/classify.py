"""Stage 3 — audio event classification on object-agnostic physical axes.

Instead of object-specific event names, every HOI sound is described by two axes that
*any* object obeys, plus the entity attribution (added later in fusion):

  interaction_mode  (temporal structure → what the loss does in time)
      impulsive   single sharp transient            → anchor one instant
      repetitive  periodic transients               → series of anchors (dribble, hammering, steps)
      continuous  sustained broadband/tonal energy  → constrain an interval (slide/scrape/drag/roll)
      resonant    transient + long tonal decay      → contact instant, object free-rings after
      none        no significant event

  contact_quality   (surface character → constraint stiffness)
      hard        rigid impact (pin position)
      soft        cushioned/light touch
      friction    sliding/grinding (free tangential)
      air         non-contact whoosh/swing (no contact constraint, motion cue only)
      na          not applicable

The readable ``event_type`` is a *derived alias* of (mode × quality) — generic, so it covers
mug / chair / hammer / drawer / broom equally:
  strike(impulsive+hard) tap(impulsive+soft) bounce(repetitive+hard) rattle(repetitive+soft)
  slide(continuous+friction) roll(continuous+hard) ring(resonant) swish(air) none unknown

Approaches compared: C1 rule (thresholds on axes), C2 KMeans cluster (axes from centroid),
C3 pretrained AST (AudioSet tag → event_type → axes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import EventFeatures, detect_periodic

INTERACTION_MODES = ("impulsive", "repetitive", "continuous", "resonant", "none")
CONTACT_QUALITIES = ("hard", "soft", "friction", "air", "na")
TAXONOMY = ("strike", "tap", "bounce", "rattle", "slide", "roll", "ring", "swish", "none", "unknown")


@dataclass
class Label:
    event_type: str
    mode: str
    quality: str
    confidence: float
    source: str
    extra: dict = field(default_factory=dict)


# (mode, quality) → readable event_type
def event_from_axes(mode: str, quality: str) -> str:
    if mode == "none":
        return "none"
    if quality == "air":
        return "swish"
    if mode == "resonant":
        return "ring"
    if mode == "continuous":
        return "slide" if quality == "friction" else "roll"
    if mode == "repetitive":
        return "bounce" if quality == "hard" else "rattle"
    if mode == "impulsive":
        return "strike" if quality == "hard" else "tap"
    return "unknown"


# reverse map (for the pretrained arm, which yields a name → approx axes)
_EVENT_AXES = {
    "strike": ("impulsive", "hard"), "tap": ("impulsive", "soft"),
    "bounce": ("repetitive", "hard"), "rattle": ("repetitive", "soft"),
    "slide": ("continuous", "friction"), "roll": ("continuous", "hard"),
    "ring": ("resonant", "hard"), "swish": ("impulsive", "air"),
    "none": ("none", "na"), "unknown": ("impulsive", "soft"),
}


def axes_from_event(event_type: str) -> tuple[str, str]:
    return _EVENT_AXES.get(event_type, ("impulsive", "soft"))


# ---------------------------------------------------------------------------
# C1 — rule-based axes
# ---------------------------------------------------------------------------

def classify_axes(f: EventFeatures, *, periodic: bool) -> tuple[str, str, float]:
    short = f.decay_ms <= 140.0
    long_tail = f.decay_ms >= 300.0
    quiet = f.rms < 0.03
    noisy = (f.flatness >= 0.30) or (f.zcr >= 0.15)
    tonal = (f.hpr >= 0.55) and (f.flatness < 0.20)
    sharp = (f.attack >= 0.5) and (f.hf_ratio >= 0.20 or f.centroid >= 1500.0)
    airy = (f.attack < 0.30) and (f.hf_ratio < 0.05) and (f.centroid < 500.0) and not quiet

    # --- interaction mode ---
    if quiet and not (sharp or noisy):
        mode = "none"
    elif long_tail and tonal:
        mode = "resonant"
    elif long_tail and noisy:
        mode = "continuous"
    elif short:
        mode = "repetitive" if periodic else "impulsive"
    else:
        mode = "impulsive"

    # --- contact quality ---
    if mode == "none":
        quality = "na"
    elif airy:
        quality = "air"
    elif mode == "continuous":
        quality = "friction"
    elif sharp and not quiet:
        quality = "hard"
    else:
        quality = "soft"

    # confidence from how decisively the rules fired
    conf = 0.5
    if mode in ("resonant", "continuous"):
        conf = 0.7
    elif mode == "impulsive" and quality == "hard":
        conf = 0.85
    elif mode == "repetitive":
        conf = 0.8
    elif mode == "none":
        conf = 0.3
    return mode, quality, conf


def rule_classify(feats: list[EventFeatures], onset_frames: list[int]) -> list[Label]:
    periodic = detect_periodic(onset_frames)
    out = []
    for f in feats:
        mode, quality, conf = classify_axes(f, periodic=periodic)
        out.append(Label(event_from_axes(mode, quality), mode, quality, conf, "rule",
                         {"periodic": periodic}))
    return out


# ---------------------------------------------------------------------------
# C2 — unsupervised clustering (axes named from each cluster centroid)
# ---------------------------------------------------------------------------

def cluster_classify(feats: list[EventFeatures], onset_frames: list[int]) -> list[Label]:
    if len(feats) < 4:
        return [Label(l.event_type, l.mode, l.quality, l.confidence * 0.8, "cluster", l.extra)
                for l in rule_classify(feats, onset_frames)]
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = np.nan_to_num(np.vstack([f.vector() for f in feats]), nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(X)
    best = None
    for k in range(2, min(6, len(feats))):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        if len(set(km.labels_)) < 2:
            continue
        sil = silhouette_score(Xs, km.labels_)
        if best is None or sil > best[0]:
            best = (sil, km)
    if best is None:
        return rule_classify(feats, onset_frames)
    sil, km = best
    periodic = detect_periodic(onset_frames)
    centroids = []
    for cid in range(km.n_clusters):
        cf = EventFeatures(*X[km.labels_ == cid].mean(axis=0).tolist())
        centroids.append(classify_axes(cf, periodic=periodic))
    out = []
    for cid in km.labels_:
        mode, quality, _ = centroids[cid]
        out.append(Label(event_from_axes(mode, quality), mode, quality,
                         float(np.clip(sil, 0.0, 1.0)), "cluster",
                         {"cluster": int(cid), "silhouette": round(float(sil), 3), "k": int(km.n_clusters)}))
    return out


# ---------------------------------------------------------------------------
# C3 — pretrained AST (AudioSet) tag → event_type → axes
# ---------------------------------------------------------------------------

_AUDIOSET_MAP = [
    (("bounce", "bouncing", "boing"), "bounce"),
    (("slam", "bang", "knock", "thump", "thud", "hammer", "clap", "smash"), "strike"),
    (("tap", "tick", "click", "snap"), "tap"),
    (("scrape", "scratch", "friction", "rub", "sliding", "skid", "sweep", "saw"), "slide"),
    (("rattle", "rattling", "jingle", "keys", "chain", "clatter"), "rattle"),
    (("clink", "clang", "ding", "ring", "bell", "glass", "ceramic", "chime", "resonance"), "ring"),
    (("whoosh", "swish", "swoosh", "wind", "whip"), "swish"),
    (("roll", "rolling", "rumble"), "roll"),
]

_AST_STATE: dict = {}


def _load_ast():
    if "model" in _AST_STATE:
        return _AST_STATE["model"], _AST_STATE["fe"]
    try:
        import torch  # noqa: F401
        from transformers import AutoFeatureExtractor, ASTForAudioClassification
    except Exception as e:
        _AST_STATE.update(model=None, fe=None, err=str(e))
        return None, None
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    try:
        fe = AutoFeatureExtractor.from_pretrained(name)
        model = ASTForAudioClassification.from_pretrained(name).eval()
    except Exception as e:
        _AST_STATE.update(model=None, fe=None, err=str(e))
        return None, None
    _AST_STATE.update(model=model, fe=fe)
    return model, fe


def _map_tag(tag: str) -> str | None:
    t = tag.lower()
    for keys, label in _AUDIOSET_MAP:
        if any(k in t for k in keys):
            return label
    return None


def pretrained_classify(x: np.ndarray, onset_times: list[float], sr: int = 16000,
                        feats: list[EventFeatures] | None = None,
                        onset_frames: list[int] | None = None) -> list[Label] | None:
    model, fe = _load_ast()
    if model is None:
        return None
    import torch

    win = int(0.96 * sr)
    id2label = model.config.id2label
    out: list[Label] = []
    with torch.inference_mode():
        for t in onset_times:
            c = int(round(t * sr))
            lo = max(0, c - win // 2)
            seg = x[lo:lo + win].astype(np.float32)
            if seg.size < win:
                seg = np.pad(seg, (0, win - seg.size))
            inp = fe(seg, sampling_rate=sr, return_tensors="pt")
            probs = torch.softmax(model(**inp).logits[0], dim=-1)
            topk = torch.topk(probs, 8)
            tags = [(id2label[int(i)], float(p)) for i, p in zip(topk.indices, topk.values)]
            et, conf = "unknown", float(tags[0][1])
            for name, p in tags:
                m = _map_tag(name)
                if m is not None:
                    et, conf = m, p
                    break
            mode, quality = axes_from_event(et)
            out.append(Label(et, mode, quality, conf, "pretrained_ast",
                             {"top_tags": [t[0] for t in tags[:3]]}))
    return out


def ast_available() -> bool:
    m, _ = _load_ast()
    return m is not None


def ast_error() -> str:
    return _AST_STATE.get("err", "")


CLASSIFIERS = ("rule", "cluster", "pretrained")
