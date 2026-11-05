"""Turn explicit constraints and gesture tags into something inference can use.

Inference works over per-note pitch candidates; the :class:`ConstraintSet`
tells it which pitches are fixed, which consecutive intervals must have a
given sign or size, and which pairs of notes must be equal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .representation import Constraint, Riff, Sketch


@dataclass
class ConstraintSet:
    fixed_pitch: dict[int, int] = field(default_factory=dict)
    direction: dict[int, int] = field(default_factory=dict)  # i -> sign of interval i->i+1
    interval: dict[int, int] = field(default_factory=dict)  # i -> exact semitones i->i+1
    same: list[tuple[int, int]] = field(default_factory=list)  # (i, j) same pitch
    # soft hints consumed by inference's cost function
    accent: set[int] = field(default_factory=set)
    palm_mute: set[int] = field(default_factory=set)
    staccato: set[int] = field(default_factory=set)
    bend: dict[int, float] = field(default_factory=dict)
    slide: set[int] = field(default_factory=set)
    vibrato: set[int] = field(default_factory=set)
    big_leap: set[int] = field(default_factory=set)
    arpeggio: set[int] = field(default_factory=set)
    ending: set[int] = field(default_factory=set)

    def is_hard(self) -> bool:
        return bool(self.fixed_pitch or self.direction or self.interval or self.same)


def _sign(v: float) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


def build_constraints(sketch: Sketch, gesture_min_confidence: float = 0.75) -> ConstraintSet:
    """Expand explicit constraints and (confident) gestures into a ConstraintSet.

    Gestures below ``gesture_min_confidence`` are ignored for hard
    constraints; UI-inserted gestures default to 0.8 so they count, while
    rule-based suggestions (0.5-0.7) do not until the user accepts them.
    """
    cs = ConstraintSet()
    n = len(sketch.observations)

    for c in sketch.constraints:
        if c.kind == "pitch" and c.value is not None:
            cs.fixed_pitch[c.i] = int(round(c.value))
        elif c.kind == "same" and c.j is not None:
            cs.same.append((c.i, c.j))
        elif c.kind == "direction" and c.value is not None:
            cs.direction[c.i] = _sign(c.value)
        elif c.kind == "interval" and c.value is not None:
            cs.interval[c.i] = int(round(c.value))

    for g in sketch.gestures:
        idx = [i for i in g.indices() if 0 <= i < n]
        if not idx:
            continue
        hard = g.confidence >= gesture_min_confidence
        if g.kind == "ascending_run" and hard:
            for i in idx[:-1]:
                cs.direction.setdefault(i, 1)
        elif g.kind == "descending_run" and hard:
            for i in idx[:-1]:
                cs.direction.setdefault(i, -1)
        elif g.kind == "repeated_note" and hard:
            for i in idx[:-1]:
                cs.direction.setdefault(i, 0)
        elif g.kind in ("repeated_figure", "pedal_tone") and hard:
            period = int(g.params.get("period", 4))
            offset = int(g.params.get("offset", 0))
            for i in idx:
                j = i + period
                if j in idx and (g.kind == "repeated_figure" or (i - idx[0] - offset) % period == 0):
                    cs.same.append((i, j))
        elif g.kind == "accent":
            cs.accent.update(idx)
        elif g.kind == "palm_mute":
            cs.palm_mute.update(idx)
        elif g.kind == "staccato":
            cs.staccato.update(idx)
        elif g.kind == "bend":
            for i in idx:
                cs.bend[i] = float(g.params.get("semitones", 2))
        elif g.kind == "slide":
            cs.slide.update(idx)
        elif g.kind == "vibrato":
            cs.vibrato.update(idx)
        elif g.kind == "big_leap":
            cs.big_leap.update(idx)
        elif g.kind == "arpeggio":
            cs.arpeggio.update(idx)
        elif g.kind == "ending":
            cs.ending.update(idx)
    return cs


def violations(riff: Riff, cs: ConstraintSet) -> list[str]:
    """List human-readable hard-constraint violations for a riff."""
    p = riff.pitches
    out: list[str] = []
    for i, v in cs.fixed_pitch.items():
        if i < len(p) and p[i] != v:
            out.append(f"note {i} should be pitch {v}, got {p[i]}")
    for i, s in cs.direction.items():
        if i + 1 < len(p) and _sign(p[i + 1] - p[i]) != s:
            out.append(f"interval {i}->{i + 1} should have sign {s}")
    for i, v in cs.interval.items():
        if i + 1 < len(p) and p[i + 1] - p[i] != v:
            out.append(f"interval {i}->{i + 1} should be {v}")
    for i, j in cs.same:
        if i < len(p) and j < len(p) and p[i] != p[j]:
            out.append(f"notes {i} and {j} should match")
    return out


def apply_articulation(riff: Riff, cs: ConstraintSet) -> Riff:
    """Copy articulation flags from the constraint set onto the riff's notes."""
    for i, note in enumerate(riff.notes):
        if i in cs.accent:
            note.velocity = min(127, note.velocity + 24)
            note.articulation["accent"] = True
        if i in cs.palm_mute:
            note.articulation["palm_mute"] = True
        if i in cs.staccato:
            note.articulation["staccato"] = True
        if i in cs.bend:
            note.articulation["bend"] = cs.bend[i]
        if i in cs.slide:
            note.articulation["slide"] = True
        if i in cs.vibrato:
            note.articulation["vibrato"] = True
    return riff
