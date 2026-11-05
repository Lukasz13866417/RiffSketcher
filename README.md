# Riff Sketcher

Reconstruct the riff you can hear in your head without knowing theory or playing an instrument.
You **tap** the rhythm, **draw** the rough pitch shape, **tag** what you know (palm mute, ascending run,
"these two notes are the same"), the engine proposes candidate riffs **silently**, and only then do you
**listen** once and refine. The product constraint is to extract as much information as possible before
playing anything, because hearing a wrong candidate can overwrite the melody in your head.

## Quick start

```bash
uv sync            # app + tests
uv sync --extra ml # also torch, pretty_midi (for the ML track)
uv run uvicorn api.main:app --reload --port 8765
```

Open <http://localhost:8765>. Keys: `R` start recording, `Space`/letters tap, `Enter` stop, drag points to
draw, `Shift`+click to select a range, `Delete` removes notes, double-click adds one, hold `P` and drag to
paint heights, arrow keys nudge the selection.

```bash
uv run pytest -q                       # all tests
uv run python -m ml.train --smoke      # tiny training run on synthetic phrases
uv run python -m ml.evaluate --synthetic 200 --checkpoint runs/smoke/best.pt
```

## Layout

```
frontend/   vanilla HTML/CSS/JS UI (tap recorder, pitch graph, gesture tagging, candidates, preview)
api/        FastAPI: JSON in, candidates / WAV out; serves frontend/ at /
core/       music logic, usable without the web app
  representation.py   Sketch (observations, gestures, constraints) and Riff (notes) + JSON
  rhythm.py           tempo estimation, grid quantisation, multi-pass merging
  motifs.py           hierarchical gesture library + rule-based suggestions
  constraints.py      gestures/facts -> hard and soft constraints, violation checks
  inference.py        deterministic reconstruction (Viterbi over scale tones), Reconstructor protocol
  synthesis.py        numpy pluck synth -> WAV (palm mute, bend, slide, vibrato, staccato)
ml/         data pipeline, sketch simulator, baselines, small Transformer, train/evaluate (see ml/README.md)
tests/      pytest for core, api and ml
experiments/notebooks/
data/raw/, data/processed/   datasets (git-ignored)
```

## The sketch format

Everything speaks the same JSON (`core/representation.py`):

```json
{
  "observations": [{"t": 0.0, "y": 0, "confidence": 0.8}, {"t": 0.52, "y": 3.0, "confidence": 0.5}],
  "gestures":     [{"kind": "ascending_run", "start": 0, "end": 3, "confidence": 0.9, "params": {}}],
  "constraints":  [{"kind": "pitch", "i": 0, "value": 40}, {"kind": "same", "i": 0, "j": 7}],
  "tempo_hint": null
}
```

`t` is the tapped onset in seconds, `y` the drawn height (roughly semitones, may be `null`), and
`confidence` how sure the user is. Gestures and constraints are what the user asserts; confident ones
become hard constraints in inference.

## How reconstruction works today (no ML yet)

1. `rhythm.quantise` estimates a tempo (60–200 BPM) and snaps onsets to a sixteenth-note grid.
2. `constraints.build_constraints` expands gestures and facts into fixed pitches, interval signs and
   "same pitch" pairs.
3. `inference.DeterministicReconstructor` runs a Viterbi search for every key x scale x height scaling:
   emission cost = distance from the drawn height, scale penalty for non-scale tones, mild leap penalty,
   hard constraints as infinite cost, small bonus for starting/ending on the tonic. Non-adjacent "same"
   pairs are enforced by fixing and re-solving. Paths are de-duplicated and ranked; the top N become
   candidates with a key label and score. It also exposes `interval_distribution()` so it can be
   scored with the same metrics as the learned model.
4. `synthesis.render` turns a candidate into audio only when the user explicitly presses play.

On a bank of eight hand-written rock riffs with simulated drawing noise of 0.15 semitones the engine
gets 88% of intervals right in the top candidate and 98% within the top five; at 0.4 semitones noise
that drops to 77% / 86%. That is the baseline the Transformer has to beat (`ml/evaluate.py` compares them).

## API

| Endpoint | Body | Returns |
| --- | --- | --- |
| `GET /api/gestures` | – | hierarchical gesture library |
| `POST /api/merge-passes` | `{passes: [[s...], ...]}` | merged onsets |
| `POST /api/quantise` | `{onsets, tempo_hint}` | tempo, onsets in beats |
| `POST /api/suggest-gestures` | sketch | rule-based gesture suggestions |
| `POST /api/reconstruct` | `{sketch, n_candidates}` | ranked candidate riffs with note names and violations |
| `POST /api/synthesize` | riff | `audio/wav` |

## Roadmap (from the plan)

1. Interactive prototype: tapping, timeline, editable pitch graph — **done**
2. Gesture system: region tags, inserted tropes, rule-based suggestions — **done (first version)**
3. First playback: deterministic inference + synthesis — **done**
4. Data pipeline: MIDI extraction, sketch simulator, song-level splits — **scaffolded in `ml/`**
5. First trained model: 1–5M-parameter Transformer over interval classes — **scaffolded, needs real data**
6. Model integration into the UI (swap `DeterministicReconstructor` for the model behind `Reconstructor`)
7. Active feedback: choose the most informative silent/audible question
8. Guitar specialisation: muting, slides, bends, motifs

Datasets to fetch into `data/raw/`: POP909 first (labelled melody tracks), then Lakh MIDI / PDMX, later
DadaGP and GuitarSet. Split by song before cutting phrases. Check composition rights before any
commercial use.
