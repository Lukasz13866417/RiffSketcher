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
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8765
```

Open <http://localhost:8765>. With `--host 0.0.0.0` the app is also reachable from other devices on your
network at `http://<this machine's LAN IP>:8765` (find it with `hostname -I`); the desktop app's launch
config in `.claude/launch.json` does the same. There is no authentication, so only do this on a network
you trust. Each device keeps its own saved sketch, since it lives in the browser's local storage. The UI is a five-step wizard: `→` / `←` move between steps (or click the
step pills). Keys: `R` start recording, `Space`/letters tap, `Enter` stop, drag points to draw,
`Shift`+click to select a range, `↑`/`↓` nudge the selection, `Delete` removes notes, double-click adds
one, hold `P` and drag to paint heights. The graph toolbar zooms the time and pitch axes (pitch zoom
centres on the selection) and `Fit` frames the drawn heights.

If `runs/pop909/best.pt` exists (or `RIFF_MODEL` points at a checkpoint) the API loads the trained
Transformer and makes it the default engine; otherwise it uses the deterministic search. Step 4 has an
engine switch. Train a model with the commands in `ml/README.md`.

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
| `GET /api/engines` | – | available reconstruction engines and the default |
| `POST /api/reconstruct` | `{sketch, n_candidates, engine}` | ranked candidate riffs with note names and violations; `engine` is `auto`, `model` or `deterministic` |
| `POST /api/synthesize` | riff | `audio/wav` |

## Roadmap (from the plan)

1. Interactive prototype: tapping, timeline, editable pitch graph — **done**
2. Gesture system: region tags, inserted tropes, rule-based suggestions — **done (first version)**
3. First playback: deterministic inference + synthesis — **done**
4. Data pipeline: MIDI extraction, sketch simulator, song-level splits — **done (`ml/`)**
5. First trained model: 1.9M-parameter Transformer over interval classes — **done on POP909**, see `ml/README.md` Results
6. Model integration: `ml/reconstructor.py` wraps the trained model behind the same `Reconstructor`
   protocol (beam + sampled candidates, anchored to pinned pitches, hard-constraint filtering with the
   search engine as fallback) — **done**; a constrained decoder that honours facts during generation
   instead of filtering afterwards is the next improvement
7. Active feedback: choose the most informative silent/audible question
8. Guitar specialisation: muting, slides, bends, motifs

Datasets to fetch into `data/raw/`: POP909 first (labelled melody tracks), then Lakh MIDI / PDMX, later
DadaGP and GuitarSet. Split by song before cutting phrases. Check composition rights before any
commercial use.
