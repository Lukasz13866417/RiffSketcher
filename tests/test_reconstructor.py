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


def _pitches_of(riffs):
    return [tuple(x.pitches) for x in riffs]


def test_constrained_decoding_masks_facts(rec):
    # contradicts the drawn contour on purpose: the model must still obey without the fallback engine
    sk = make_sketch([0, 3, 5, 7, 5, 3, 0])
    sk.constraints = [Constraint("pitch", 0, value=40), Constraint("direction", 3, value=1),
                      Constraint("same", 1, 5), Constraint("interval", 0, value=5), Constraint("pitch", 6, value=47)]
    riffs = rec.reconstruct(sk, 4)
    assert riffs and all(r.meta["method"] == "transformer" for r in riffs)  # no fallback needed
    cs = build_constraints(sk)
    for r in riffs:
        assert violations(r, cs) == []
        p = r.pitches
        assert p[0] == 40 and p[6] == 47 and p[1] - p[0] == 5 and p[4] > p[3] and p[1] == p[5]


def test_anchor_pin_is_the_earliest_pin():
    from ml.reconstructor import TransformerReconstructor
    from core.constraints import build_constraints
    sk = make_sketch([0, 4, 7])
    sk.constraints = [Constraint("pitch", 2, value=52), Constraint("pitch", 1, value=45)]
    assert TransformerReconstructor._anchor_pin(build_constraints(sk)) == (1, 45)
    assert TransformerReconstructor._anchor_pin(build_constraints(make_sketch([0, 1]))) is None


def test_allowed_mask_logic():
    from ml.reconstructor import TransformerReconstructor
    from core.representation import INTERVAL_MIN
    sk = make_sketch([0, 2, 4, 4])
    sk.constraints = [Constraint("direction", 0, value=-1), Constraint("same", 0, 3), Constraint("pitch", 2, value=50)]
    cs = build_constraints(sk)
    ok = TransformerReconstructor._allowed(cs, 1, [0], None)
    classes = list(range(INTERVAL_MIN, INTERVAL_MIN + len(ok)))
    assert all((c < 0) == ok[i] for i, c in enumerate(classes))  # only descending steps
    # a pin on note 2 at 50 relative to an anchor pin (note 0 at 48): rel[2] - rel[0] must be +2 -> step +4
    ok = TransformerReconstructor._allowed(cs, 2, [0, -2], (0, 48))
    assert [c for i, c in enumerate(classes) if ok[i]] == [4]
    ok = TransformerReconstructor._allowed(cs, 3, [0, -2, 2], (0, 48))  # same as note 0: rel must return to 0 -> -2
    assert [c for i, c in enumerate(classes) if ok[i]] == [-2]
    ok = TransformerReconstructor._allowed(cs, 2, [0, -2], None)  # no anchor pin: only the "same (0,3)" reachability prune
    assert ok is not None and ok.sum() > 40 and not ok[0]  # a -24 step would put note 3 out of reach of note 0
    ok = TransformerReconstructor._allowed(cs, 2, [0, -2], (2, 50))  # the anchor pin itself is free: only reachability remains
    assert ok is not None and ok.sum() > 40


def test_separate_timing_model_drives_gaps():
    torch.manual_seed(0)
    from ml.datasets import GENRES, RHYTHM_CLASSES
    pitch_model = SketchTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=2, n_genres=len(GENRES))).eval()
    timing_model = SketchTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=2, n_genres=len(GENRES),
                                                 n_rhythm=len(RHYTHM_CLASSES))).eval()
    r = TransformerReconstructor(model=pitch_model, rhythm_model=timing_model, rhythm_checkpoint=None,
                                 device="cpu", meta={"genres_seen": ["unknown"]})
    riffs = r.reconstruct(make_sketch([0, 2, 4, 5, 7]), 2)
    assert riffs and all(x.meta.get("rhythm") == "model" for x in riffs if x.meta["method"] == "transformer")
    plain = TransformerReconstructor(model=pitch_model, rhythm_checkpoint=None, device="cpu", meta={"genres_seen": ["unknown"]})
    assert all(x.meta.get("rhythm") == "quantised" for x in plain.reconstruct(make_sketch([0, 2, 4, 5, 7]), 2))


def test_mask_keeps_future_pins_reachable():
    from ml.reconstructor import TransformerReconstructor
    from core.representation import INTERVAL_MIN
    sk = make_sketch([0, 0, 0, 0])
    sk.constraints = [Constraint("pitch", 0, value=40), Constraint("pitch", 3, value=80)]  # +40 over three steps
    cs = build_constraints(sk)
    ok = TransformerReconstructor._allowed(cs, 1, [0], (0, 40))
    classes = [c for i, c in enumerate(range(INTERVAL_MIN, INTERVAL_MIN + len(ok))) if ok[i]]
    assert min(classes) == -8 and max(classes) == 24  # after step 1, two steps of +24 must still reach +40


def test_propagate_pins_through_same_chains():
    from ml.reconstructor import TransformerReconstructor
    sk = make_sketch([0, 0, 0, 0, 0])
    sk.constraints = [Constraint("same", 0, 2), Constraint("same", 2, 4), Constraint("pitch", 4, value=50)]
    cs = build_constraints(sk)
    TransformerReconstructor._propagate(cs)
    assert cs.fixed_pitch == {0: 50, 2: 50, 4: 50}
