"""Offline tests for the ML pipeline (synthetic data only, < 60 s total)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.representation import N_INTERVAL_CLASSES, Riff, Sketch
from ml.baseline import ContourBaseline, DeterministicBaseline, topk_sequences
from ml.datasets import (GESTURE_KINDS, PAD_TARGET, Phrase, PhraseDataset, collate, load_phrases,
                         save_phrases, sketch_to_features, split_by_song, synthetic_phrases)
from ml.evaluate import evaluate_predictor, expected_calibration_error, make_examples
from ml.model import ModelConfig, SketchTransformer, count_parameters, load_checkpoint, save_checkpoint
from ml.sketch_simulator import PRESETS, SimConfig, SketchSimulator, detect_gestures


@pytest.fixture(scope="module")
def phrases() -> list[Phrase]:
    return synthetic_phrases(40, seed=3)


# -- synthetic phrases ----------------------------------------------------- #

def test_synthetic_phrases_are_valid(phrases):
    assert len(phrases) == 40
    for p in phrases:
        assert 5 <= len(p) <= 16
        assert len(p.onsets_beats) == len(p.durations_beats) == len(p.pitches)
        assert p.onsets_beats[0] == 0.0
        assert all(b > a for a, b in zip(p.onsets_beats, p.onsets_beats[1:]))
        assert all(36 <= x <= 84 for x in p.pitches)
        assert all(abs(d) <= 24 for d in p.intervals)
    assert len({p.song_id for p in phrases}) == 10


def test_phrase_jsonl_roundtrip(tmp_path, phrases):
    path = tmp_path / "p.jsonl"
    save_phrases(phrases, path)
    back = load_phrases(path)
    assert [p.to_dict() for p in back] == [p.to_dict() for p in phrases]


# -- simulator ------------------------------------------------------------- #

@pytest.mark.parametrize("level", list(PRESETS))
def test_simulator_produces_valid_sketches(phrases, level):
    sim = SketchSimulator(0, level)
    for p in phrases[:20]:
        r = sim.simulate(p)
        sk = r.sketch
        assert isinstance(sk, Sketch)
        assert len(sk) == len(r.pitches) == len(r.source_index) >= 3
        assert len(r.intervals) == len(sk) - 1
        onsets = sk.onsets
        assert onsets[0] == 0.0 and all(b > a for a, b in zip(onsets, onsets[1:]))
        for o in sk.observations:
            assert 0.0 <= o.confidence <= 1.0
            assert o.y is None or np.isfinite(o.y)
        for g in sk.gestures:
            assert 0 <= g.start <= g.end < len(sk)
            assert g.kind in GESTURE_KINDS
        # aligned pitches are the phrase's pitches at the source indices
        assert r.pitches == [p.pitches[i] for i in r.source_index]


def test_simulator_no_drops_keeps_length(phrases):
    sim = SketchSimulator(0, SimConfig(drop_prob=0.0, double_prob=0.0))
    for p in phrases:
        assert len(sim.simulate(p).sketch) == len(p)


def test_simulator_respects_dropped_and_doubled_taps(phrases):
    sim = SketchSimulator(0, SimConfig(drop_prob=0.5, double_prob=0.0, min_notes=3))
    lens = [len(sim.simulate(p).sketch) for p in phrases]
    assert all(3 <= n <= len(p) for n, p in zip(lens, phrases))
    assert any(n < len(p) for n, p in zip(lens, phrases))
    sim2 = SketchSimulator(0, SimConfig(drop_prob=0.0, double_prob=0.5))
    r = sim2.simulate(phrases[0])
    assert len(r.sketch) > len(phrases[0])
    assert len(r.source_index) == len(set(r.source_index)) + (len(r.sketch) - len(phrases[0]))


def test_simulator_missing_heights_and_hint():
    p = synthetic_phrases(1, seed=5, min_notes=16, max_notes=16)[0]
    sim = SketchSimulator(1, SimConfig(missing_prob=0.5, tempo_hint_prob=1.0))
    r = sim.simulate(p)
    assert any(o.y is None for o in r.sketch.observations)
    assert r.sketch.observations[0].y is not None
    assert r.sketch.tempo_hint is not None and r.sketch.tempo_hint > 0


def test_simulator_confidence_tracks_noise():
    p = synthetic_phrases(1, seed=7, min_notes=16, max_notes=16)[0]
    sim = SketchSimulator(2, SimConfig(drop_prob=0, double_prob=0, missing_prob=0,
                                       compression_range=(1.0, 1.0), height_noise=1.5))
    errs, confs = [], []
    for _ in range(30):
        r = sim.simulate(p)
        y = np.array([o.y for o in r.sketch.observations])
        truth = np.array(r.pitches) - r.pitches[0]
        errs.extend(np.abs(y - truth).tolist())
        confs.extend(o.confidence for o in r.sketch.observations)
    assert np.corrcoef(errs, confs)[0, 1] < -0.3


def test_detect_gestures():
    tags = detect_gestures([60, 62, 64, 65, 64, 62, 60, 60, 60])
    kinds = {(g.kind, g.start, g.end) for g in tags}
    assert ("ascending_run", 0, 3) in kinds
    assert ("descending_run", 3, 6) in kinds
    assert ("repeated_note", 6, 8) in kinds
    rep = detect_gestures([60, 64, 67, 60, 64, 67, 60, 64])
    assert any(g.kind == "repeated_figure" and g.params["period"] == 3 for g in rep)


# -- splits ---------------------------------------------------------------- #

def test_split_by_song_has_no_overlap(phrases):
    s = split_by_song(phrases, 0.2, 0.2, seed=1)
    ids = {k: {p.song_id for p in v} for k, v in s.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert sum(len(v) for v in s.values()) == len(phrases)
    assert all(len(v) > 0 for v in s.values())


# -- features / dataset ---------------------------------------------------- #

def test_features_and_collate_shapes(phrases):
    sim = SketchSimulator(0, "medium")
    ds = PhraseDataset(phrases, sim, seed=0)
    items = [ds[i] for i in range(6)]
    for it, p in zip(items, phrases):
        n = it["cont"].shape[0]
        assert it["cont"].shape == (n, 6)
        assert it["gest"].shape == (n, len(GESTURE_KINDS))
        assert it["target"].shape == (n - 1,)
        assert int(it["target"].min()) >= 0 and int(it["target"].max()) < N_INTERVAL_CLASSES
    batch = collate(items)
    n_max = max(it["cont"].shape[0] for it in items)
    assert batch["cont"].shape == (6, n_max, 6)
    assert batch["gest"].shape == (6, n_max, len(GESTURE_KINDS))
    assert batch["mask"].shape == (6, n_max) and batch["mask"].dtype == torch.bool
    assert batch["target"].shape == (6, n_max - 1)
    assert int(batch["mask"].sum()) == sum(it["cont"].shape[0] for it in items)
    assert int((batch["target"] == PAD_TARGET).sum()) == 6 * (n_max - 1) - sum(it["target"].shape[0] for it in items)
    # resampling: a different epoch gives a different sketch of the same phrase
    ds.set_epoch(1)
    assert not torch.equal(ds[0]["cont"], items[0]["cont"])


def test_features_handle_missing_heights():
    sk = Sketch.from_dict({"observations": [{"t": 0.0, "y": 0.0}, {"t": 0.5, "y": None, "confidence": 0.1},
                                            {"t": 1.0, "y": 3.0}]})
    f = sketch_to_features(sk)
    assert f["cont"].shape == (3, 6)
    assert f["cont"][1, 5] == 1.0 and f["cont"][0, 5] == 0.0
    assert np.isfinite(f["cont"]).all()


# -- baselines ------------------------------------------------------------- #

def test_baselines_return_normalised_distributions(phrases):
    sim = SketchSimulator(0, "easy")
    for pred in (ContourBaseline(), ContourBaseline(scale_snap=False), DeterministicBaseline()):
        for p in phrases[:3]:
            sk = sim.simulate(p).sketch
            d = pred.predict(sk)
            assert d.shape == (len(sk) - 1, N_INTERVAL_CLASSES)
            assert np.all(d >= 0) and np.allclose(d.sum(axis=1), 1.0)
            cands = pred.candidates(sk, 3)
            assert 1 <= len(cands) <= 3 and all(len(c) == len(sk) - 1 for c in cands)


def test_contour_baseline_is_decent_on_easy_sketches(phrases):
    sim = SketchSimulator(0, SimConfig(height_noise=0.2, compression_range=(1, 1), missing_prob=0,
                                       drop_prob=0, double_prob=0))
    ex = make_examples(phrases[:10], sim)
    m = evaluate_predictor(ContourBaseline(scale_snap=False), ex, k=5)
    assert m["interval_acc"] > 0.8


def test_topk_sequences():
    d = np.full((2, N_INTERVAL_CLASSES), 1e-6)
    d[0, 24] = 0.9; d[0, 25] = 0.1
    d[1, 26] = 0.6; d[1, 22] = 0.4
    seqs = topk_sequences(d, 3)
    assert seqs[0] == [0, 2] and seqs[1] == [0, -2] and len(seqs) == 3


# -- model ----------------------------------------------------------------- #

def test_model_parameter_count_default_config():
    n = count_parameters(SketchTransformer())
    assert 500_000 <= n <= 6_000_000


def test_model_forward_and_generate_shapes(phrases):
    torch.manual_seed(0)
    sim = SketchSimulator(0, "medium")
    ds = PhraseDataset(phrases, sim)
    batch = collate([ds[i] for i in range(5)])
    model = SketchTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64))
    logits = model(batch)
    B, n_max = batch["mask"].shape
    assert logits.shape == (B, n_max - 1, N_INTERVAL_CLASSES)
    assert torch.isfinite(logits).all()
    for mode, s in (("sample", 3), ("greedy", 2), ("beam", 3)):
        cls, lp, pr = model.generate(batch, s, mode=mode)
        assert cls.shape == (B, s, n_max - 1) and lp.shape == (B, s)
        assert pr.shape == (B, s, n_max - 1, N_INTERVAL_CLASSES)
        assert int(cls.min()) >= 0 and int(cls.max()) < N_INTERVAL_CLASSES
    r = ds.simulate(0)
    d = model.predict(r.sketch, n_samples=4)
    assert d.shape == (len(r.sketch) - 1, N_INTERVAL_CLASSES) and np.allclose(d.sum(1), 1.0)
    cands = model.candidates(r.sketch, 4)
    assert len(cands) == 4 and all(len(c) == len(r.sketch) - 1 for c in cands)
    riffs = model.riffs_from_sketch(r.sketch, 3, start_pitch=52)
    assert 1 <= len(riffs) <= 3
    for riff in riffs:
        assert isinstance(riff, Riff) and len(riff) == len(r.sketch)
        assert riff.notes[0].pitch == 52 and riff.notes[0].onset == 0.0


def test_tiny_training_loop_decreases_loss(phrases):
    torch.manual_seed(0)
    sim = SketchSimulator(0, "easy")
    ds = PhraseDataset(phrases, sim)
    batch = collate([ds[i] for i in range(32)])
    model = SketchTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = model.loss(batch).item()
    for _ in range(40):
        loss = model.loss(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    last = model.loss(batch).item()
    assert last < 0.7 * first


def test_checkpoint_roundtrip(tmp_path):
    model = SketchTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64))
    path = tmp_path / "m.pt"
    save_checkpoint(model, str(path), epoch=1, val_acc=0.5)
    back, ckpt = load_checkpoint(str(path))
    assert ckpt["epoch"] == 1 and back.cfg == model.cfg
    for a, b in zip(model.parameters(), back.parameters()):
        assert torch.equal(a, b)


# -- evaluate -------------------------------------------------------------- #

def test_evaluate_metrics_are_sane(phrases):
    sim = SketchSimulator(0, "medium")
    ex = make_examples(phrases[:8], sim)
    model = SketchTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64))
    model.name = "model"
    for pred in (ContourBaseline(), model):
        m = evaluate_predictor(pred, ex, k=3)
        assert 0.0 <= m["interval_acc"] <= 1.0
        assert 0.0 <= m["top3_recon"] <= 1.0 and 0.0 <= m["seq_acc"] <= m["top3_recon"]
        assert m["nll"] >= 0.0 and 0.0 <= m["ece"] <= 1.0
        assert m["n"] == 8


def test_ece_edge_cases():
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([0.0, 0.0])) == pytest.approx(1.0)


def test_train_smoke_cli(tmp_path):
    from ml.train import main
    out = tmp_path / "smoke"
    main(["--smoke", "--out", str(out), "--epochs", "1", "--max-steps-per-epoch", "3"])
    assert (out / "best.pt").exists() and (out / "history.json").exists()
    hist = json.loads((out / "history.json").read_text())
    assert len(hist) == 1 and np.isfinite(hist[0]["val_loss"])


def test_deterministic_baseline_survives_extreme_heights():
    sk = Sketch.from_dict({"observations": [{"t": 0.25 * i, "y": y, "confidence": 0.9}
                                            for i, y in enumerate([0, 2, 4, 40, 42, -30, 1, 3])]})
    d = DeterministicBaseline().predict(sk)
    assert d.shape == (7, N_INTERVAL_CLASSES) and np.allclose(d.sum(1), 1.0)
    assert len(DeterministicBaseline().candidates(sk, 2)[0]) == 7
