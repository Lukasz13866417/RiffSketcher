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
mkdir -p data/raw && git clone https://github.com/music-x-lab/POP909-Dataset data/raw/POP909-Dataset
uv run python -m ml.prepare_dataset --midi-dir data/raw/POP909-Dataset/POP909 --out data/processed/phrases.jsonl
```

`prepare_dataset` takes the `MELODY` track of each song, makes it
monophonic, splits it into 5–16 note phrases at rests > 2 beats and writes
`phrases.jsonl` plus `splits.json`. The split is **by song** (train/val/test
= 80/10/10 of song ids) so no phrase of a test song is ever seen in training.

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

Tests: `uv run pytest tests/test_ml.py -q` (offline, synthetic only).
