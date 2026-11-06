"""Turn clean phrases into noisy user sketches.

The simulator models what a user does when they "tap and draw" a melody:

* tapping jitter (Gaussian ms-level noise), occasional dropped/doubled taps
* a global tempo scale plus slow tempo drift over the phrase
* an approximate contour: ``y = (pitch - pitch[0]) * compression + noise``
  with a random compression factor and occasional missing heights
* per-note confidence that is correlated with the noise actually applied
* imperfect gesture tags: rule-based detection on the true pitches, then
  randomly dropped and boundary-shifted by +-1

Because taps can be dropped or doubled, the result carries the pitch
sequence *aligned to the observations* so that targets line up.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from core.representation import Constraint, GestureTag, Observation, Sketch
from ml.datasets import Phrase


@dataclass
class SimConfig:
    name: str = "medium"
    onset_jitter_ms: float = 25.0
    drop_prob: float = 0.03
    double_prob: float = 0.02
    base_tempo: float = 120.0
    tempo_scale_range: tuple[float, float] = (0.8, 1.2)
    tempo_drift: float = 0.05  # total relative drift (std) over the phrase
    height_noise: float = 1.0  # semitones (std of the per-note noise)
    height_noise_spread: float = 0.5  # log-normal spread of the per-note sigma
    compression_range: tuple[float, float] = (0.7, 1.3)
    missing_prob: float = 0.05
    tag_drop_prob: float = 0.3
    tag_shift_prob: float = 0.3
    tag_confidence: float = 0.8
    tempo_hint_prob: float = 0.5
    min_notes: int = 3
    # facts the user asserts (pinned pitch, interval direction, two notes equal): generated from
    # the true pitches so the model learns to condition on them; see SketchSimulator._facts
    facts_prob: float = 0.5  # share of sketches that carry any facts
    pin_prob: float = 0.15  # per note, given facts
    direction_prob: float = 0.15  # per interval, given facts
    same_prob: float = 0.3  # per exact-repeat pair (within 8 notes), given facts


PRESETS: dict[str, SimConfig] = {
    "easy": SimConfig(name="easy", onset_jitter_ms=10.0, drop_prob=0.0, double_prob=0.0,
                      tempo_drift=0.02, height_noise=0.5, missing_prob=0.0,
                      compression_range=(0.9, 1.1), tag_drop_prob=0.1, tag_shift_prob=0.1),
    "medium": SimConfig(),
    "hard": SimConfig(name="hard", onset_jitter_ms=45.0, drop_prob=0.08, double_prob=0.05,
                      tempo_drift=0.1, height_noise=1.8, missing_prob=0.15,
                      compression_range=(0.6, 1.4), tag_drop_prob=0.5, tag_shift_prob=0.5),
}


def get_config(level: str | SimConfig) -> SimConfig:
    if isinstance(level, SimConfig):
        return level
    if level not in PRESETS:
        raise KeyError(f"unknown sim level {level!r}; choose from {list(PRESETS)}")
    return replace(PRESETS[level])


@dataclass
class SimResult:
    sketch: Sketch
    pitches: list[int]  # true pitches aligned with sketch.observations
    onsets_beats: list[float]  # true onsets (beats) aligned with observations
    source_index: list[int]  # phrase note index behind each observation
    phrase: Phrase
    meta: dict = field(default_factory=dict)

    @property
    def intervals(self) -> list[int]:
        return [b - a for a, b in zip(self.pitches, self.pitches[1:])]


# --------------------------------------------------------------------------- #
# Rule-based gesture detection on TRUE pitches
# --------------------------------------------------------------------------- #


def detect_gestures(pitches: list[int], min_run: int = 3) -> list[GestureTag]:
    """Ground-truth contour gestures from an exact pitch sequence."""
    n = len(pitches)
    out: list[GestureTag] = []
    if n < 2:
        return out
    d = np.diff(np.asarray(pitches))
    sign = np.sign(d).astype(int)
    i = 0
    while i < n - 1:
        j = i
        while j < n - 1 and sign[j] == sign[i]:
            j += 1
        if j - i + 1 >= min_run:
            kind = {1: "ascending_run", -1: "descending_run", 0: "repeated_note"}[int(sign[i])]
            out.append(GestureTag(kind, i, j, 0.9))
        i = j if j > i else i + 1
    # repeated figure: smallest period p with pitches[i] == pitches[i+p] over >= 2p notes
    if n >= 6:
        for p in range(2, n // 2 + 1):
            if all(pitches[k] == pitches[k + p] for k in range(n - p)) and len(set(pitches[:p])) > 1:
                out.append(GestureTag("repeated_figure", 0, n - 1, 0.9, {"period": p}))
                break
    # big leaps
    for k, dk in enumerate(d):
        if abs(int(dk)) >= 10:
            out.append(GestureTag("big_leap", k, k, 0.9))
    return out


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #


class SketchSimulator:
    """Sample a noisy :class:`Sketch` from a :class:`Phrase`."""

    def __init__(self, rng: np.random.Generator | int | None = None,
                 cfg: SimConfig | str = "medium") -> None:
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.cfg = get_config(cfg)

    # -- individual noise sources ---------------------------------------- #

    def _tap_events(self, phrase: Phrase, rng: np.random.Generator) -> list[tuple[int, float]]:
        """(source note index, true onset in beats) per tap after drops/doubles."""
        cfg = self.cfg
        n = len(phrase)
        events: list[tuple[int, float]] = []
        n_dropped = 0
        max_drop = max(0, n - cfg.min_notes)
        for i in range(n):
            if i > 0 and n_dropped < max_drop and rng.random() < cfg.drop_prob:
                n_dropped += 1
                continue
            events.append((i, phrase.onsets_beats[i]))
            if rng.random() < cfg.double_prob:
                nxt = phrase.onsets_beats[i + 1] if i + 1 < n else phrase.onsets_beats[i] + 1.0
                gap = max(0.1, nxt - phrase.onsets_beats[i])
                events.append((i, phrase.onsets_beats[i] + gap * float(rng.uniform(0.2, 0.5))))
        return events

    def _onsets_seconds(self, beats: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        cfg = self.cfg
        tempo = cfg.base_tempo * float(rng.uniform(*cfg.tempo_scale_range))
        n = len(beats)
        step_std = cfg.tempo_drift / np.sqrt(max(n - 1, 1))
        drift = np.exp(np.cumsum(rng.normal(0.0, step_std, size=n)))
        t = np.zeros(n)
        for i in range(1, n):
            t[i] = t[i - 1] + (beats[i] - beats[i - 1]) * 60.0 / (tempo * drift[i])
        t = t + rng.normal(0.0, cfg.onset_jitter_ms / 1000.0, size=n)
        t = t - t[0]
        # keep strictly increasing with a minimal physical gap
        for i in range(1, n):
            t[i] = max(t[i], t[i - 1] + 0.03)
        return t, tempo

    def _heights(self, pitches: np.ndarray, rng: np.random.Generator):
        cfg = self.cfg
        n = len(pitches)
        comp = float(rng.uniform(*cfg.compression_range))
        sigma = cfg.height_noise * np.exp(rng.normal(0.0, cfg.height_noise_spread, size=n))
        noise = rng.normal(0.0, 1.0, size=n) * sigma
        y = (pitches - pitches[0]) * comp + noise
        missing = rng.random(n) < cfg.missing_prob
        missing[0] = False
        # confidence tracks the actual error (the user "feels" how sure they are)
        conf = np.exp(-np.abs(noise) / 1.5) + rng.normal(0.0, 0.1, size=n)
        conf = np.clip(conf, 0.05, 1.0)
        conf[missing] = np.clip(rng.uniform(0.0, 0.2, size=int(missing.sum())), 0.0, 1.0)
        return y, missing, conf, comp

    def _facts(self, pitches: list[int], rng: np.random.Generator) -> list[Constraint]:
        """Hard facts about the true pitches, as a user would assert them."""
        cfg = self.cfg
        n = len(pitches)
        if n < 2 or rng.random() >= cfg.facts_prob:
            return []
        out: list[Constraint] = []
        for i in range(n):
            if rng.random() < cfg.pin_prob:
                out.append(Constraint("pitch", i, value=float(pitches[i])))
        for i in range(n - 1):
            if rng.random() < cfg.direction_prob:
                d = pitches[i + 1] - pitches[i]
                out.append(Constraint("direction", i, value=float(np.sign(d))))
        for i in range(n):
            for j in range(i + 2, min(n, i + 9)):
                if pitches[i] == pitches[j] and rng.random() < cfg.same_prob:
                    out.append(Constraint("same", i, j=j))
                    break
        return out

    def _gestures(self, pitches: list[int], rng: np.random.Generator) -> list[GestureTag]:
        cfg = self.cfg
        n = len(pitches)
        out: list[GestureTag] = []
        for g in detect_gestures(pitches):
            if rng.random() < cfg.tag_drop_prob:
                continue
            start, end = g.start, g.end
            if rng.random() < cfg.tag_shift_prob:
                start = int(np.clip(start + rng.integers(-1, 2), 0, n - 1))
            if rng.random() < cfg.tag_shift_prob:
                end = int(np.clip(end + rng.integers(-1, 2), 0, n - 1))
            if start > end:
                start, end = end, start
            out.append(GestureTag(g.kind, start, end, cfg.tag_confidence, dict(g.params)))
        return out

    # -- public ----------------------------------------------------------- #

    def simulate(self, phrase: Phrase, rng: np.random.Generator | None = None) -> SimResult:
        rng = rng if rng is not None else self.rng
        events = self._tap_events(phrase, rng)
        src = [i for i, _ in events]
        beats = np.array([b for _, b in events], dtype=float)
        pitches = [int(phrase.pitches[i]) for i in src]
        t, tempo = self._onsets_seconds(beats, rng)
        y, missing, conf, comp = self._heights(np.array(pitches, dtype=float), rng)
        obs = [Observation(float(t[i]), None if missing[i] else float(y[i]), float(conf[i]))
               for i in range(len(events))]
        gestures = self._gestures(pitches, rng)
        facts = self._facts(pitches, rng)
        tempo_hint = float(tempo * rng.uniform(0.95, 1.05)) if rng.random() < self.cfg.tempo_hint_prob else None
        sketch = Sketch(obs, gestures, facts, tempo_hint,
                        meta={"song_id": phrase.song_id, "sim": self.cfg.name,
                              "compression": comp, "tempo": tempo})
        return SimResult(sketch, pitches, [float(b) for b in beats], src, phrase,
                         meta={"tempo": tempo, "compression": comp})

    def __call__(self, phrase: Phrase, rng: np.random.Generator | None = None) -> SimResult:
        return self.simulate(phrase, rng)
