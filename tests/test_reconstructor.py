"""The model-backed reconstructor must behave like the deterministic one from the
API's point of view, even with an untrained model (no checkpoint needed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.constraints import build_constraints, violations  # noqa: E402
from core.representation import Constraint, GestureTag, Observation, Sketch  # noqa: E402
from ml.model import ModelConfig, SketchTransformer  # noqa: E402
from ml.reconstructor import TransformerReconstructor, guess_key  # noqa: E402


@pytest.fixture(scope="module")
def rec():
    torch.manual_seed(0)
    model = SketchTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=2)).eval()
    return TransformerReconstructor(model=model, device="cpu")


def make_sketch(heights, conf=0.8):
    return Sketch([Observation(i * 0.5, y, conf) for i, y in enumerate(heights)])


def test_guess_key():
    assert guess_key([45, 47, 48, 50, 52, 53, 55, 57]) == "A minor"
    assert guess_key([60, 64, 67, 72, 67, 64, 60]).startswith("C major")
    assert guess_key([]) == "unknown"


def test_reconstruct_shapes_and_ranking(rec):
    riffs = rec.reconstruct(make_sketch([0, 3, 5, 7, 5, 3, 0]), 4)
    assert 1 <= len(riffs) <= 4
    assert all(len(r) == 7 for r in riffs)
    assert all(r.key and r.tempo > 0 for r in riffs)
    assert len({tuple(r.pitches) for r in riffs}) == len(riffs)
    assert all(a.score >= b.score for a, b in zip(riffs, riffs[1:]) if a.meta["method"] == b.meta["method"])


def test_reconstruct_respects_constraints_or_falls_back(rec):
    sk = make_sketch([0, 3, 5, 7])
    sk.constraints = [Constraint("pitch", 0, value=40), Constraint("direction", 1, value=1)]
    sk.gestures = [GestureTag("palm_mute", 0, 3)]
    riffs = rec.reconstruct(sk, 3)
    cs = build_constraints(sk)
    assert riffs
    for r in riffs:
        assert violations(r, cs) == []
        assert r.pitches[0] == 40
        assert r.notes[0].articulation.get("palm_mute")


def test_single_note_and_empty(rec):
    assert rec.reconstruct(Sketch(), 3) == []
    r = rec.reconstruct(make_sketch([0]), 3)
    assert r and len(r[0]) == 1


def test_missing_checkpoint_is_not_available():
    assert not TransformerReconstructor(checkpoint="/nonexistent.pt").available


def test_untrained_genre_falls_back_to_unknown():
    torch.manual_seed(0)
    from ml.datasets import GENRES
    model = SketchTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=2, n_genres=len(GENRES))).eval()
    r = TransformerReconstructor(model=model, device="cpu", meta={"genres_seen": ["unknown", "pop"]})
    assert r.genres == ["unknown", "pop"]
    # an untrained label must behave exactly like "unknown"
    sk_metal = make_sketch([0, 2, 4]); sk_metal.meta["genre"] = "metal"
    sk_unknown = make_sketch([0, 2, 4]); sk_unknown.meta["genre"] = "unknown"
    seqs = lambda riffs: [tuple(x.pitches) for x in riffs]  # noqa: E731
    assert seqs(r.reconstruct(sk_metal, 3)) == seqs(r.reconstruct(sk_unknown, 3))
    # a trained label is passed through and changes the prediction
    sk_pop = make_sketch([0, 2, 4]); sk_pop.meta["genre"] = "pop"
    assert seqs(r.reconstruct(sk_pop, 3)) != seqs(r.reconstruct(sk_unknown, 3))


def test_rhythm_head_drives_candidate_timing():
    torch.manual_seed(0)
    from ml.datasets import GENRES, RHYTHM_CLASSES
    model = SketchTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=2, n_genres=len(GENRES),
                                          n_rhythm=len(RHYTHM_CLASSES))).eval()
    r = TransformerReconstructor(model=model, device="cpu", meta={"genres_seen": ["unknown"]})
    riffs = r.reconstruct(make_sketch([0, 2, 4, 5, 7]), 3)
    assert riffs
    for riff in riffs:
        if riff.meta.get("method") == "transformer":
            assert riff.meta.get("rhythm") == "model"
            onsets = [n.onset for n in riff.notes]
            assert onsets == sorted(onsets) and onsets[0] == 0.0
            assert 40 <= riff.tempo <= 240
            assert all(n.duration > 0 for n in riff.notes)
