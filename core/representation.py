"""Core data types shared by the app, the inference code and the ML pipeline.

Everything here is plain dataclasses with JSON (de)serialisation so the same
objects can flow through the HTTP API, scripts, notebooks and training code.

Vocabulary
----------
Sketch
    What the user gives us: a list of *observations*, one per tapped note
    attack.  Each observation has an onset time in seconds and an approximate
    pitch height ``y`` in an arbitrary unit (the vertical position the user
    dragged the point to, roughly in semitones but noisy).  Optional gesture
    tags and constraints carry what the user is confident about.
Riff
    A concrete symbolic melody: notes with MIDI pitch, onset (beats), duration
    (beats) and articulation flags.  This is what inference produces and what
    the synthesiser renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable
import json


# --------------------------------------------------------------------------- #
# Sketch side (user input)
# --------------------------------------------------------------------------- #


@dataclass
class Observation:
    """One tapped note attack with an approximate pitch height."""

    t: float  # onset time in seconds, relative to recording start
    y: float | None = None  # approximate pitch height; None = unknown
    confidence: float = 0.5  # 0..1, how sure the user is about y
    hold: float | None = None  # seconds the tap was held (key down to key up); None = not recorded

    def to_dict(self) -> dict[str, Any]:
        return {"t": self.t, "y": self.y, "confidence": self.confidence, "hold": self.hold}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        return cls(t=float(d["t"]), y=None if d.get("y") is None else float(d["y"]),
                   confidence=float(d.get("confidence", 0.5)),
                   hold=None if d.get("hold") is None else float(d["hold"]))


@dataclass
class GestureTag:
    """A musical gesture attached to a note index range ``[start, end]``.

    ``kind`` is a key in :mod:`core.motifs` (``"ascending_run"``,
    ``"palm_mute"``, ``"repeated_figure"`` ...).  ``params`` carries anything
    gesture-specific, e.g. ``{"period": 4}`` for a repeated figure.
    """

    kind: str
    start: int
    end: int  # inclusive
    confidence: float = 0.8
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GestureTag":
        return cls(kind=str(d["kind"]), start=int(d["start"]), end=int(d["end"]),
                   confidence=float(d.get("confidence", 0.8)),
                   params=dict(d.get("params", {})))

    def indices(self) -> range:
        return range(self.start, self.end + 1)


@dataclass
class Constraint:
    """A hard fact the user asserts about the melody.

    kinds
    -----
    ``pitch``      note ``i`` has exact MIDI pitch ``value``
    ``same``       notes ``i`` and ``j`` have the same pitch
    ``direction``  interval from ``i`` to ``i+1`` has sign ``value`` (-1, 0, +1)
    ``interval``   interval from ``i`` to ``i+1`` is exactly ``value`` semitones
    """

    kind: str
    i: int
    j: int | None = None
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Constraint":
        return cls(kind=str(d["kind"]), i=int(d["i"]),
                   j=None if d.get("j") is None else int(d["j"]),
                   value=None if d.get("value") is None else float(d["value"]))


@dataclass
class Sketch:
    observations: list[Observation] = field(default_factory=list)
    gestures: list[GestureTag] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    tempo_hint: float | None = None  # BPM, if the user tapped a tempo
    meta: dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.observations)

    @property
    def onsets(self) -> list[float]:
        return [o.t for o in self.observations]

    @property
    def heights(self) -> list[float | None]:
        return [o.y for o in self.observations]

    def sorted(self) -> "Sketch":
        """Return a copy with observations sorted by onset time."""
        order = sorted(range(len(self.observations)), key=lambda i: self.observations[i].t)
        remap = {old: new for new, old in enumerate(order)}
        obs = [self.observations[i] for i in order]
        gestures = [GestureTag(g.kind, remap[g.start], remap[g.end], g.confidence, dict(g.params))
                    for g in self.gestures]
        cons = [Constraint(c.kind, remap[c.i], None if c.j is None else remap[c.j], c.value)
                for c in self.constraints]
        return Sketch(obs, gestures, cons, self.tempo_hint, dict(self.meta))

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [o.to_dict() for o in self.observations],
            "gestures": [g.to_dict() for g in self.gestures],
            "constraints": [c.to_dict() for c in self.constraints],
            "tempo_hint": self.tempo_hint,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sketch":
        return cls(
            observations=[Observation.from_dict(o) for o in d.get("observations", [])],
            gestures=[GestureTag.from_dict(g) for g in d.get("gestures", [])],
            constraints=[Constraint.from_dict(c) for c in d.get("constraints", [])],
            tempo_hint=d.get("tempo_hint"),
            meta=dict(d.get("meta", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "Sketch":
        return cls.from_dict(json.loads(s))


# --------------------------------------------------------------------------- #
# Riff side (symbolic output)
# --------------------------------------------------------------------------- #


@dataclass
class Note:
    pitch: int  # MIDI note number
    onset: float  # in beats
    duration: float  # in beats
    velocity: int = 96
    articulation: dict[str, Any] = field(default_factory=dict)  # palm_mute, accent, bend, slide

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Note":
        return cls(pitch=int(d["pitch"]), onset=float(d["onset"]), duration=float(d["duration"]),
                   velocity=int(d.get("velocity", 96)), articulation=dict(d.get("articulation", {})))


@dataclass
class Riff:
    notes: list[Note] = field(default_factory=list)
    tempo: float = 120.0  # BPM
    key: str | None = None  # e.g. "E minor"
    score: float = 0.0  # inference score (higher = more plausible)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.notes)

    @property
    def pitches(self) -> list[int]:
        return [n.pitch for n in self.notes]

    @property
    def intervals(self) -> list[int]:
        p = self.pitches
        return [b - a for a, b in zip(p, p[1:])]

    def to_dict(self) -> dict[str, Any]:
        return {
            "notes": [n.to_dict() for n in self.notes],
            "tempo": self.tempo,
            "key": self.key,
            "score": self.score,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Riff":
        return cls(notes=[Note.from_dict(n) for n in d.get("notes", [])],
                   tempo=float(d.get("tempo", 120.0)), key=d.get("key"),
                   score=float(d.get("score", 0.0)), meta=dict(d.get("meta", {})))

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "Riff":
        return cls.from_dict(json.loads(s))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def name_to_midi(name: str) -> int:
    name = name.strip()
    i = 1
    while i < len(name) and name[i] in "#b":
        i += 1
    pc_name, octave = name[:i], int(name[i:])
    flat_map = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
    pc_name = flat_map.get(pc_name, pc_name)
    return 12 * (octave + 1) + NOTE_NAMES.index(pc_name)


def riff_from_pitches(pitches: Iterable[int], onsets_beats: Iterable[float] | None = None,
                      durations: Iterable[float] | None = None, tempo: float = 120.0) -> Riff:
    """Quick constructor used by tests and the ML pipeline."""
    pitches = list(pitches)
    onsets = list(onsets_beats) if onsets_beats is not None else [float(i) for i in range(len(pitches))]
    if durations is None:
        durations = [max(0.25, b - a) for a, b in zip(onsets, onsets[1:])] + [1.0]
    return Riff(notes=[Note(p, o, d) for p, o, d in zip(pitches, onsets, durations)], tempo=tempo)


# Interval classes shared by inference and the ML models: -24..+24 semitones.
INTERVAL_MIN = -24
INTERVAL_MAX = 24
N_INTERVAL_CLASSES = INTERVAL_MAX - INTERVAL_MIN + 1


def interval_to_class(d: int) -> int:
    return int(min(INTERVAL_MAX, max(INTERVAL_MIN, d)) - INTERVAL_MIN)


def class_to_interval(c: int) -> int:
    return int(c) + INTERVAL_MIN
