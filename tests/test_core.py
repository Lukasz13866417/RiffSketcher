import json

import numpy as np
import pytest

from core.constraints import build_constraints, violations
from core.inference import DeterministicReconstructor, InferenceConfig, filled_heights
from core.motifs import flat_library, suggest_gestures
from core.representation import (Constraint, GestureTag, Observation, Riff, Sketch,
                                 class_to_interval, interval_to_class, midi_to_name,
                                 name_to_midi, riff_from_pitches)
from core.rhythm import estimate_tempo, merge_passes, quantise
from core.synthesis import render, riff_to_wav


def make_sketch(pitches, beats, tempo=120, noise=0.0, conf=0.9, seed=0):
    rng = np.random.default_rng(seed)
    onsets = [b * 60 / tempo + rng.normal(0, 0.01) for b in beats]
    y = [p - pitches[0] + rng.normal(0, noise) for p in pitches]
    return Sketch([Observation(t, yy, conf) for t, yy in zip(onsets, y)])


# -- representation ------------------------------------------------------- #

def test_sketch_json_roundtrip():
    sk = Sketch([Observation(0.0, 0.0), Observation(0.5, None, 0.2)],
                [GestureTag("ascending_run", 0, 1, 0.8, {"x": 1})],
                [Constraint("pitch", 0, value=40)], tempo_hint=100, meta={"a": 1})
    back = Sketch.from_json(sk.to_json())
    assert back.to_dict() == sk.to_dict()
    assert back.observations[1].y is None


def test_sketch_sorted_remaps_indices():
    sk = Sketch([Observation(1.0, 1), Observation(0.0, 0)],
                [GestureTag("accent", 0, 0)], [Constraint("same", 0, 1)])
    s = sk.sorted()
    assert s.onsets == [0.0, 1.0]
    assert s.gestures[0].start == 1
    assert s.constraints[0].i == 1 and s.constraints[0].j == 0


def test_note_names():
    assert midi_to_name(40) == "E2"
    assert name_to_midi("E2") == 40
    assert name_to_midi("Bb3") == name_to_midi("A#3") == 58


def test_interval_classes():
    assert interval_to_class(0) == 24
    assert class_to_interval(interval_to_class(-24)) == -24
    assert interval_to_class(40) == 48


def test_riff_intervals():
    r = riff_from_pitches([40, 43, 41])
    assert r.intervals == [3, -2]
    assert Riff.from_json(r.to_json()).pitches == [40, 43, 41]


# -- rhythm ----------------------------------------------------------------- #

def test_estimate_tempo_recovers_even_pulse():
    tempo = 120
    onsets = [i * 60 / tempo for i in range(8)]
    t, _ = estimate_tempo(onsets)
    assert t in (60, 120) or abs(t - 120) <= 2


def test_quantise_snaps_to_grid():
    r = quantise([0.0, 0.26, 0.51, 0.74, 1.0], tempo=120)
    assert r.onsets_beats == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert len(r.durations_beats) == 5


def test_quantise_empty():
    r = quantise([])
    assert r.onsets_beats == []


def test_merge_passes_averages_and_keeps_extras():
    merged = merge_passes([[0, 0.5, 1.0], [10.02, 10.5, 11.0, 11.5]])
    assert len(merged) == 4
    assert merged[1] == pytest.approx(0.5, abs=0.02)


# -- motifs / constraints --------------------------------------------------- #

def test_library_flat_has_categories():
    lib = flat_library()
    assert lib["palm_mute"]["category"] == "articulation"
    assert "ascending_run" in lib


def test_suggest_ascending_run_and_repeat():
    sk = make_sketch([40, 42, 44, 45, 40, 42, 44, 45], [0, 1, 2, 3, 4, 5, 6, 7])
    kinds = {g.kind for g in suggest_gestures(sk)}
    assert "ascending_run" in kinds
    assert "repeated_figure" in kinds


def test_build_constraints_from_gestures():
    sk = Sketch([Observation(i * 0.5, 0) for i in range(4)],
                gestures=[GestureTag("ascending_run", 0, 2, 0.9), GestureTag("palm_mute", 0, 3)],
                constraints=[Constraint("pitch", 0, value=40)])
    cs = build_constraints(sk)
    assert cs.direction == {0: 1, 1: 1}
    assert cs.fixed_pitch == {0: 40}
    assert cs.palm_mute == {0, 1, 2, 3}
    bad = riff_from_pitches([40, 39, 41, 41])
    assert violations(bad, cs)


# -- inference ---------------------------------------------------------------- #

def test_reconstruct_recovers_clean_minor_scale():
    pitches = [45, 47, 48, 50, 52, 53, 55, 57]
    sk = make_sketch(pitches, [i * 0.5 for i in range(8)], noise=0.1)
    riffs = DeterministicReconstructor().reconstruct(sk, 3)
    assert riffs[0].intervals == [2, 1, 2, 2, 1, 2, 2]
    assert "minor" in riffs[0].key or "major" in riffs[0].key
    assert len(riffs[0].notes) == 8


def test_reconstruct_respects_hard_constraints():
    sk = make_sketch([40, 43, 45, 47], [0, 1, 2, 3], noise=0.3)
    sk.constraints = [Constraint("pitch", 0, value=40), Constraint("direction", 2, value=-1)]
    r = DeterministicReconstructor().reconstruct(sk, 1)[0]
    assert r.pitches[0] == 40
    assert r.pitches[3] < r.pitches[2]
    assert violations(r, build_constraints(sk)) == []


def test_reconstruct_same_constraint_non_adjacent():
    sk = make_sketch([40, 44, 47, 41], [0, 1, 2, 3], noise=0.2)
    sk.constraints = [Constraint("same", 0, 3)]
    r = DeterministicReconstructor().reconstruct(sk, 1)[0]
    assert r.pitches[0] == r.pitches[3]


def test_reconstruct_handles_missing_heights():
    sk = Sketch([Observation(0, 0, 0.9), Observation(0.5, None, 0.1), Observation(1.0, 4, 0.9)])
    assert filled_heights(sk).tolist() == [0, 2, 4]
    riffs = DeterministicReconstructor().reconstruct(sk, 2)
    assert len(riffs) >= 1 and len(riffs[0]) == 3


def test_reconstruct_candidates_are_distinct_and_ranked():
    sk = make_sketch([40, 43, 45, 47, 45, 43, 40], [0, 1, 2, 3, 4, 5, 6], noise=0.5)
    riffs = DeterministicReconstructor().reconstruct(sk, 5)
    seqs = [tuple(r.pitches) for r in riffs]
    assert len(set(seqs)) == len(seqs)
    assert all(a.score >= b.score for a, b in zip(riffs, riffs[1:]))


def test_interval_distribution_is_normalised():
    sk = make_sketch([40, 43, 45], [0, 1, 2], noise=0.3)
    d = DeterministicReconstructor().interval_distribution(sk)
    assert d.shape == (2, 49)
    assert np.allclose(d.sum(axis=1), 1.0)


def test_empty_sketch():
    assert DeterministicReconstructor().reconstruct(Sketch(), 3) == []


# -- synthesis ------------------------------------------------------------------ #

def test_render_and_wav():
    r = riff_from_pitches([40, 43, 45], tempo=120)
    r.notes[0].articulation["palm_mute"] = True
    r.notes[1].articulation["bend"] = 2
    r.notes[2].articulation["vibrato"] = True
    sig = render(r)
    assert sig.dtype == np.float32 and np.max(np.abs(sig)) <= 1.0 and len(sig) > 22050
    wav = riff_to_wav(r)
    assert wav[:4] == b"RIFF" and len(wav) > 1000


def test_reconstruct_extreme_heights_do_not_crash():
    # a height far outside the playable range must clamp to the edge, not raise
    sk = Sketch([Observation(0, 0, 0.9), Observation(0.5, -40, 0.9), Observation(1.0, 60, 0.9)])
    riffs = DeterministicReconstructor().reconstruct(sk, 2)
    assert riffs and len(riffs[0]) == 3
    assert all(36 <= p <= 84 for p in riffs[0].pitches)
