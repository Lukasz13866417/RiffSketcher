# ml/ — Track B: learned sketch-to-intervals model

Predicts the exact semitone intervals of a melody from a noisy "tap and
draw" sketch (`core.representation.Sketch`). Everything runs offline on CPU
with a synthetic fallback; a GPU is used automatically when available.

## Modules

| file | what |
|---|---|
| `datasets.py` | `Phrase`, MIDI extraction (`phrases_from_midi`), `synthetic_phrases`, `split_by_song`, features, `PhraseDataset` + `collate` |
| `sketch_simulator.py` | `SketchSimulator` / `SimConfig` (`easy`/`medium`/`hard`): jitter, dropped/doubled taps, tempo drift, noisy compressed contour, missing heights, imperfect gesture tags |
| `baseline.py` | `ContourBaseline` (round drawn heights, optional scale snap) and `DeterministicBaseline` (wraps `core.inference`) |
| `model.py` | `SketchTransformer` (encoder over sketch tokens, autoregressive interval decoder), `predict`, `candidates`, `riffs_from_sketch` |
| `train.py`, `evaluate.py`, `prepare_dataset.py` | CLIs |

## Data: POP909

```bash
mkdir -p data/raw && git clone --depth 1 https://github.com/music-x-lab/POP909-Dataset data/raw/pop909
uv run python -m ml.prepare_dataset --midi-dir data/raw/pop909/POP909 --out data/processed/pop909.jsonl
```

`prepare_dataset` takes the `MELODY` track of each song, makes it
monophonic, converts times to beats through the file's tempo map (many
POP909 songs have tempo changes) and writes `pop909.jsonl` plus
`splits.json`. The split is **by song** (train/val/test = 80/10/10 of song
ids) so no phrase of a test song is ever seen in training.

Phrase splitting: the track is cut at rests longer than 2 beats, but pop
melodies rarely rest that long (1.3% of note transitions in POP909), so the
remaining runs of 50+ notes are cut recursively at their largest internal
gap (`split_segment`) until every phrase has 5–16 notes. Cutting every 16
notes instead would slice through musical phrases. On POP909 this gives
30,677 phrases from all 909 songs (mean 10 notes, pitches 42–98, intervals
within ±24): 24,520 train / 3,133 val / 3,024 test phrases over 727 / 91 /
91 songs. Interval histogram (|semitones|): 0: 22%, 1: 9%, 2: 38%, 3: 14%,
4: 4%, 5: 6%, 7: 4%, ≥8: 3% — mostly diatonic steps and repeats.

Licence caveat: POP909 is released for research use only; check the
dataset's licence/terms before redistributing the MIDI or anything derived
from it. The `data/raw/` and `data/processed/` folders are git-ignored.

No MIDI? `--synthetic 4000` writes generated phrases in the same format.

## Train / evaluate

```bash
uv run python -m ml.train --smoke                                   # < 1 min on CPU, writes runs/smoke/best.pt
uv run python -m ml.train --data data/processed/phrases.jsonl --epochs 20 --out runs/first
uv run python -m ml.evaluate --checkpoint runs/first/best.pt --data data/processed/phrases.jsonl --split test
uv run python -m ml.evaluate --checkpoint runs/smoke/best.pt --synthetic 200   # no data needed
```

Sketches are re-simulated every epoch (`--sim easy|medium|hard` picks the
noise level). `evaluate` prints interval accuracy, per-note NLL, calibration
(ECE), exact-sequence accuracy and top-5 reconstruction for the contour
baseline, the deterministic engine and the model.

Tests: `uv run pytest tests/test_ml.py -q` (offline; the MIDI test builds its
own POP909-style file with `pretty_midi`).

## Results

First real-data run (2026-11-05): default `SketchTransformer` (1.93M params,
d_model 128, 4 layers), 20 epochs, batch 64, lr 3e-4, one run per noise
preset, ~4 min each on an RTX 5060. `runs/pop909/best.pt` is the **medium**
run (the preset the app targets); the easy/hard runs are in
`runs/pop909_easy` and `runs/pop909_hard`. Each row below is the POP909
**test split** (3,024 sketches from 91 unseen songs), sketches simulated with
the preset the model was trained on, `k = 5`:

```
sim=easy   (jitter 10 ms, no dropped/doubled taps, mild contour noise)
predictor       interval_acc         nll         ece  top5_recon     seq_acc
contour                0.468       1.479       0.207       0.061       0.028
deterministic          0.640       1.505       0.027       0.301       0.129
transformer            0.748       0.677       0.011       0.653       0.360

sim=medium (default; runs/pop909/best.pt)
predictor       interval_acc         nll         ece  top5_recon     seq_acc
contour                0.248       1.990       0.070       0.004       0.001
deterministic          0.388       4.263       0.258       0.051       0.017
transformer            0.558       1.249       0.026       0.260       0.107

sim=hard   (jitter 45 ms, 8% dropped / 5% doubled taps, heavy contour noise)
predictor       interval_acc         nll         ece  top5_recon     seq_acc
contour                0.158       2.599       0.050       0.003       0.001
deterministic          0.264       6.987       0.375       0.011       0.004
transformer            0.443       1.677       0.027       0.097       0.033
```

Takeaways:

* The transformer beats both baselines on every preset and every metric.
  On medium noise it gets +17 points of interval accuracy over the
  deterministic engine and finds the exact phrase in its top-5 five times as
  often (26% vs 5%). It is also far better calibrated (ECE 0.03 vs 0.26):
  the deterministic engine is confidently wrong when the sketch is noisy.
* Noise-matched training matters but the model degrades gracefully: the
  medium model evaluated on hard sketches gets 0.429 interval accuracy
  (`runs/eval_medium_on_hard.log`), close to the hard-trained 0.443.
* The validation accuracy printed during training (0.874 / 0.730 / 0.602 for
  easy / medium / hard) is teacher-forced, i.e. the decoder sees the true
  previous intervals. `evaluate` runs the decoder freely (greedy path plus
  sampled paths), so its numbers are lower; compare models on `evaluate`.
* Real data changed the picture from the synthetic experiments: POP909
  melodies are mostly diatonic steps and repeats, so the deterministic engine
  scores much higher here (0.39 vs 0.31 on synthetic medium sketches) and the
  gap to the transformer is still larger.
* Exact-sequence accuracy on medium noise is only 11%, so the app still
  needs the top-k candidate list plus user correction; the next lever is more
  data (POP909 is 909 songs) and a smarter simulator fitted to real taps.

Reproduce:

```bash
uv run python -m ml.train --data data/processed/pop909.jsonl --epochs 20 --sim medium --out runs/pop909
uv run python -m ml.evaluate --checkpoint runs/pop909/best.pt --data data/processed/pop909.jsonl --split test --sim medium
```

The deterministic engine dominates evaluation time (~7.5 min for the test
split); pass `--no-deterministic` or `--limit 300` while iterating.
