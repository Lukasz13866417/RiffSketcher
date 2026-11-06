# ml/ — Track B: learned sketch-to-intervals model

Predicts the exact semitone intervals of a melody from a noisy "tap and
draw" sketch (`core.representation.Sketch`). Everything runs offline on CPU
with a synthetic fallback; a GPU is used automatically when available.

## Modules

| file | what |
|---|---|
| `datasets.py` | `Phrase`, melody extraction (`phrases_from_midi`), guitar-riff extraction (`phrases_from_guitar_midi`, `phrases_from_dadagp`, shared `riff_line_phrases`), `synthetic_phrases`, `split_by_song`, features, `PhraseDataset` + `collate` |
| `sketch_simulator.py` | `SketchSimulator` / `SimConfig` (`easy`/`medium`/`hard`): jitter, dropped/doubled taps, tempo drift, noisy compressed contour, missing heights, imperfect gesture tags |
| `baseline.py` | `ContourBaseline` (round drawn heights, optional scale snap) and `DeterministicBaseline` (wraps `core.inference`) |
| `model.py` | `SketchTransformer` (encoder over sketch tokens, autoregressive interval decoder), `predict`, `candidates`, `riffs_from_sketch` |
| `train.py`, `evaluate.py`, `prepare_dataset.py` | CLIs (`evaluate` compares several checkpoints and reports per data source) |
| `../experiments/chug_probe.py` | the 16-note chug-riff probe from the decoding experiments, run against any checkpoint |

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

## Data: guitar riffs (Lakh guitar tracks; DadaGP when you have it)

POP909 is pop vocal melody, so a model trained on it alone treats tritones,
chromatic moves and chug patterns as noise (see "Decoding experiments"
below). The second source is guitar riff lines.

**DadaGP** (26k rock/metal GuitarPro tabs as tokens, the ideal source) is not
a public download: the [repo](https://github.com/dada-bots/dadaGP) asks you to
email the authors for research access. The loader is written and tested
against the repo's example token file (`phrases_from_dadagp`: standard tuning
plus the `downtune` token, `wait` ticks at 960 per beat, tied notes sustain,
dead notes are dropped, `distorted*`/`clean*` tracks only, repeats not
unrolled). Once you have the files, point `--dadagp-dir` at them; they are
tagged `source = "dadagp"`.

**Fallback used here: Lakh MIDI `clean_midi`** (17,256 general MIDI files
named `Artist/Title.mid`, 234 MB; licence: the MIDI files were scraped from
the web and are *not* cleared for redistribution, so treat them as
research-only and keep `data/raw/` out of git):

```bash
mkdir -p data/raw/lakh_clean && cd data/raw/lakh_clean
curl -L -o clean_midi.tar.gz http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz
tar -xzf clean_midi.tar.gz && rm clean_midi.tar.gz && cd -
cp data/processed/splits.json data/processed/pop909_splits.json   # keep the POP909 song split
uv run python -m ml.prepare_dataset --midi-dir data/raw/pop909/POP909 \
    --guitar-midi-dir data/raw/lakh_clean/clean_midi --max-per-song 12 \
    --base-splits data/processed/pop909_splits.json --out data/processed/mixed.jsonl
```

`phrases_from_guitar_midi` takes every track whose General MIDI program is
a guitar (25–32, including overdrive and distortion) or whose name says
"guitar"/"gtr" (bass tracks excluded by name). The riff line is the
**lowest** note of each onset: the root of a power chord is what you hum,
and lead lines are monophonic anyway. Onsets with more than three
simultaneous notes are strummed chords, not riff notes, and end the phrase
(dropping them silently would fake a rest mid-riff); in a 600-file sample
92% of guitar onsets had at most three notes. Then the same rest / largest-gap
splitting as the melody loader, phrases with an interval beyond ±24 semitones
dropped, exact repeats within a song dropped (riffs repeat, and doubled
guitar tracks copy each other). `Title.1.mid`, `Title.2.mid` versions of the
same song share a song id so they never straddle a split.

`--max-per-song 12` keeps a random 12 phrases per guitar song (Lakh songs
have several guitar tracks and would otherwise outweigh POP909 twenty to
one); `--base-splits` keeps every POP909 song in the split it already had, so
the earlier POP909 model is never evaluated on songs it trained on. Result:
`data/processed/mixed.jsonl`, 119,956 phrases (89,279 `lakh_guitar` from
7,970 songs, 30,677 `pop909` from 909 songs; 279 unreadable MIDI files
skipped), 95,779 / 12,040 / 12,137 train / val / test phrases over 7,103 /
888 / 888 songs. The guitar phrases are a different distribution: interval
histogram 0: 33%, 1: 7%, 2: 16%, 3: 10%, 4: 6%, 5: 9%, 6: 1%, 7: 6%, ≥8: 12%
(POP909: 0: 22%, 2: 38%, ≥8: 3%) — more repeats, fourths, fifths and octave
leaps, far fewer whole steps. Tritones are still rare (1%), so Lakh is a
partial substitute for DadaGP, not a metal corpus.

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

### Results: mixed data (2026-11-05)

Same model and recipe (1.93M params, 20 epochs, batch 64, lr 3e-4, `--sim
medium`), trained on `data/processed/mixed.jsonl` (POP909 melodies + Lakh
guitar riff lines, see "Data: guitar riffs"); 28 s per epoch on the RTX 5060.
Teacher-forced validation accuracy 0.762 (POP909-only run: 0.730), still
creeping up at epoch 20. Evaluated on the mixed **test split** (12,137
sketches from 888 unseen songs, medium noise, `k = 5`); `pop909` is the
POP909-only model (`runs/pop909/best.pt`, evaluated on the same POP909 test
songs as before thanks to `--base-splits`), `mixed` is `runs/mixed/best.pt`.
The deterministic rows come from a separate run (`runs/eval_mixed_baselines.log`,
~40 min, with the same seed) because the search engine dominates evaluation time:

```
all sources (n = 12137)
predictor       interval_acc         nll         ece  top5_recon     seq_acc
contour                0.242       2.092       0.059       0.007       0.002
deterministic          0.417       4.195       0.242       0.122       0.066
pop909                 0.526       1.570       0.075       0.261       0.141
mixed                  0.595       1.179       0.025       0.377       0.209

source = lakh_guitar (n = 9113)
predictor       interval_acc         nll         ece  top5_recon     seq_acc
contour                0.240       2.129       0.055       0.008       0.003
deterministic          0.427       4.170       0.236       0.146       0.082
pop909                 0.515       1.685       0.093       0.261       0.152
mixed                  0.599       1.178       0.031       0.397       0.231

source = pop909 (n = 3024)
predictor       interval_acc         nll         ece  top5_recon     seq_acc
contour                0.248       1.990       0.070       0.004       0.001
deterministic          0.388       4.263       0.258       0.051       0.017
pop909                 0.558       1.250       0.026       0.260       0.107
mixed                  0.582       1.181       0.010       0.315       0.144
```

Chug-riff probe (`uv run python -m experiments.chug_probe runs/pop909/best.pt
runs/mixed/best.pt`; the 16-note riff from the decoding experiments, drawn
exactly at confidence 0.8, no gesture tags):

| model | top-1 interval acc | best candidate | exact riff in top-5 |
| --- | --- | --- | --- |
| pop909 | 0.47 | `0 0 0 0 0 +2 -2 0 0 +5 -5 0 +2 -2 0` | no |
| mixed | 0.53 | same | no |

Takeaways:

* The mixed model is better on **both** sources, so the API default
  checkpoint (`DEFAULT_CHECKPOINT` in `ml/reconstructor.py`) is now
  `runs/mixed/best.pt`. On guitar phrases it gains +8 points of interval
  accuracy and finds the exact phrase in its top-5 half again as often (40%
  vs 26%); on POP909 it still gains +2 points and +5 points of top-5
  reconstruction, i.e. three times more data did not cost the pop prior
  anything. It is also much better calibrated (ECE 0.025 vs 0.075).
* The POP909 model was confidently wrong on guitar data (NLL 1.69, ECE
  0.09): exactly the "pop-like candidates" the user saw.
* The probe shows what changed and what did not. Per step, the mixed model
  now puts the 3-semitone moves first (0.54 / 0.63 / 0.51 vs 0.40 / 0.36 /
  0.48 for a whole step) where the POP909 model had 0.17 / 0.33 / 0.35, but
  the two readings are close enough that beam search still returns the
  whole-step version as the top candidate. The semitone moves are *more*
  firmly read as repeats (0.95) and the tritone still becomes a fourth
  (0.71): Lakh guitar tracks have 7% semitone and 1% tritone intervals, so
  this is still the prior talking. The fix is genuinely metal data
  (DadaGP: the loader is ready) or a riff-specific augmentation (transpose
  moves of 2 to 1/3, 5 to 6) if that data cannot be had.
* The deterministic engine does better on guitar riffs (0.427) than on pop
  melodies (0.388) — repeated notes suit its search — but its calibration is
  as poor as before (ECE 0.24) and the mixed model beats it by 17 points on
  both sources, so the fallback stays a fallback.

Reproduce:

```bash
uv run python -m ml.train --data data/processed/mixed.jsonl --epochs 20 --sim medium --out runs/mixed
uv run python -m ml.evaluate --checkpoint runs/pop909/best.pt runs/mixed/best.pt --data data/processed/mixed.jsonl --split test --sim medium
uv run python -m experiments.chug_probe runs/pop909/best.pt runs/mixed/best.pt
```

### Trope catalogue (2026-11-06)

`ml/tropes.py` counts recurring patterns in the real phrases, per genre, and
writes `data/tropes.json` (113 KB, committed). Everything is a document
frequency (once per song), so a song that repeats one riff does not dominate.
It holds, per genre and overall: interval unigrams; interval, rhythm and
combined "figure" n-grams (2-5) with the share of songs containing each;
named tropes from rule detectors with example phrases; the smallest scale
family each phrase fits; and compressed contour words.

Named tropes (share of songs), POP909 melodies vs Lakh guitar riff lines:

| trope | pop | rock |
| --- | --- | --- |
| chug run (3+ repeated notes) | 58% | 71% |
| pedal tone | 76% | 52% |
| repeated figure (exact interval period) | 10% | 53% |
| tritone move | 5% | 25% |
| octave leap | 43% | 55% |
| gallop (eighth + two sixteenths, twice) | 20% | 11% |
| pentatonic lick (4+ pitch classes, two minor-third hops) | 70% | 28% |
| descending run | 54% | 27% |
| ascending run | 35% | 19% |
| ends on its lowest note | 92% | 79% |

Interval unigrams: repeats are 22% of pop intervals and 33% of rock; whole
steps are 38% of pop and 16% of rock. Top rock trigram `[0, 0, 0]` appears in
71% of songs; top pop trigrams are `[-2, -2, 2]` and `[-2, 2, -2]` (62% / 61%).
Rhythm trigram `[0.5, 0.5, 0.5]` leads both.

Used in the app via `core.motifs.label_tropes` (chips on every candidate, plus
a familiarity score = share of the candidate's interval trigrams that are
common in the genre) and `GET /api/tropes`. Rebuild after changing the data:

```bash
uv run python -m ml.tropes --data data/processed/mixed.jsonl --out data/tropes.json
```

### Rhythm head: learned timing vs the rule-based quantiser (2026-11-06)

Motivation: on fast riffs tap jitter (30-50 ms) is comparable to a sixteenth
note, so the rule-based grid snap (`core.rhythm.quantise`) guesses. The model
now has an optional **rhythm head** (`ModelConfig.n_rhythm`, on by default in
`ml.train`, `--no-rhythm` to disable): one class per gap between consecutive
notes from `ml.datasets.RHYTHM_CLASSES` (sixteenth multiples plus triplets),
trained from the simulator's true onsets in beats alongside the interval loss.
`TransformerReconstructor` uses the predicted gaps for candidate timing when
the checkpoint has the head (`riff.meta["rhythm"] == "model"`), with the tempo
fitted by least squares to the tapped seconds; otherwise it quantises.

Two designs were tried on the mixed data (20 epochs, medium noise):

1. *Decoder-side*: the previous gap class fed back as a decoder input, gap
   logits from the decoder state. Teacher-forced validation accuracy 0.851,
   but free decoding 0.600 and interval accuracy down 2 points: the
   teacher-forced number leaked the beat unit through the true previous gap,
   and wrong gaps compounded.
2. *Encoder-side* (kept): gap k from the encoder states of notes k and k+1,
   non-autoregressive, a 2-layer MLP head. Interval accuracy 0.581 vs 0.590
   for the same model without the head.

Rhythm accuracy on 2,000 mixed test sketches (`runs/eval_rhythm.log` and a
scratch analysis), where *tempo-inv* allows a global factor of 2 in the beat
label (same grid, different tempo, harmless for playback), *fast gaps* are
true gaps of at most 0.3 beats, and *whole riff* means every gap right:

```
                      exact   tempo-inv   fast gaps   whole riff
medium  quantise      0.674     0.783       0.798       0.364
medium  rhythm head   0.647     0.781       0.805       0.344
hard    quantise      0.497     0.570       0.640       0.059
hard    rhythm head   0.460     0.622       0.608       0.104
```

Takeaways:

* On medium noise the learned head and the rule tie. On hard noise (45 ms
  jitter, dropped and doubled taps) the head is better tempo-invariantly and
  gets the whole riff right almost twice as often, but is slightly worse on
  the fastest gaps, which were the motivation.
* The head costs about a point of interval accuracy, so the API default stays
  `runs/genre/best.pt`; set `RIFF_MODEL=runs/rhythm/best.pt` to serve the
  rhythm-head checkpoint. The code path is tested and ready for a better run.
* Absolute beat labels are inherently ambiguous from tempo-normalised inputs
  (is the common gap a quarter or an eighth?), which caps both methods; the
  tempo-invariant numbers are the ones to watch.
* What would move this: real tapping data to replace the simulator's jitter
  model (the hard preset is a guess), and joint beam search over gaps and
  intervals instead of greedy gaps.

Reproduce:

```bash
uv run python -m ml.train --data data/processed/mixed.jsonl --epochs 20 --sim medium --genre-dropout 0.2 --out runs/rhythm
uv run python -m ml.evaluate --checkpoint runs/genre/best.pt runs/rhythm/best.pt --data data/processed/mixed.jsonl --split test --sim medium --no-deterministic --limit 2000
```

### Genre conditioning: one model vs separate models (2026-11-05)

Question: should each genre get its own model? Setup: the data source is the
genre label (`ml.datasets.GENRES`: unknown / pop / rock / metal; POP909 ->
pop, Lakh guitar -> rock, DadaGP -> metal). `SketchTransformer` gets a learned
genre embedding added to every encoder token (`ModelConfig.n_genres`; old
checkpoints have 0 and load unchanged). Training shows the label as "unknown"
for 20% of sketches (`--genre-dropout`), so the model also works without a
choice. Four models, same recipe (20 epochs, medium noise):

| run | data | genre token |
| --- | --- | --- |
| `runs/mixed` | POP909 + Lakh guitar | no |
| `runs/genre` | POP909 + Lakh guitar | yes, dropout 0.2 |
| `runs/pop909` | POP909 only | no (separate pop model) |
| `runs/guitar_only` | Lakh guitar only (`--source lakh_guitar`) | no (separate rock model) |

Mixed test split, medium noise, `k = 5`, conditioned model given the true
label (`--genre auto`; `runs/eval_genre_auto.log`):

```
source = pop909 (n = 3024)        interval_acc   nll    ece   top5_recon
mixed        (shared, no token)        0.581    1.186  0.016    0.315
genre        (shared, token=pop)       0.590    1.181  0.022    0.327
pop909       (separate pop model)      0.559    1.249  0.024    0.260
guitar_only  (separate rock model)     0.542    1.271  0.020    0.237

source = lakh_guitar (n = 9113)   interval_acc   nll    ece   top5_recon
mixed        (shared, no token)        0.600    1.177  0.030    0.397
genre        (shared, token=rock)      0.597    1.173  0.027    0.398
pop909       (separate pop model)      0.516    1.682  0.091    0.261
guitar_only  (separate rock model)     0.592    1.188  0.027    0.385
```

With no label (`--genre unknown`, `runs/eval_genre_unknown.log`) the
conditioned model scores 0.578 on pop and 0.598 on guitar, i.e. the same as
the unconditioned shared model.

Takeaways:

* **Separate models lose.** The pop-only model is 3 points behind the shared
  model on pop, and the guitar-only model is behind on guitar too (0.592 vs
  0.597-0.600) despite seeing the same guitar phrases: the other genre's data
  helps rather than hurts. Keep one model.
* **The genre token is nearly free and slightly positive**: +0.9 points and
  +1.2 points of top-5 reconstruction on pop, a wash on rock, and no cost when
  the label is left at "any". Pop and rock guitar lines are simply not that
  different in interval statistics; the token should matter more once a
  genuinely different genre (metal from DadaGP) is in the mix.
* The API default is now `runs/genre/best.pt`, which turns on the Genre
  dropdown in the reconstruct step (sent as `sketch.meta["genre"]` or the
  `genre` field of `/api/reconstruct`).
* **Only trained genres may be offered.** The checkpoint records
  `genres_seen` and the reconstructor maps anything else to "unknown". The
  reason: with no metal data yet, asking the model for "metal" hits an
  untrained embedding and the chug probe turns into noise (`+1 +1 +1 +5 -5 ...`,
  0.00 interval accuracy), while unknown / pop / rock all give the same 0.73
  as the unconditioned model. On this probe the pop/rock token changes
  nothing; metal data is what would.

Reproduce:

```bash
uv run python -m ml.train --data data/processed/mixed.jsonl --epochs 20 --sim medium --genre-dropout 0.2 --out runs/genre
uv run python -m ml.train --data data/processed/mixed.jsonl --epochs 20 --sim medium --source lakh_guitar --no-genre --out runs/guitar_only
uv run python -m ml.evaluate --checkpoint runs/mixed/best.pt runs/genre/best.pt runs/pop909/best.pt runs/guitar_only/best.pt \
    --data data/processed/mixed.jsonl --split test --sim medium --no-deterministic --genre auto
```

### Decoding experiments (2026-11-05)

A user reported a flat "all the same note" top candidate for a chug-heavy riff
with clearly drawn moves. Reproduced with a 16-note riff (mostly repeated E2,
moves of 1, 3 and 6 semitones) at the UI's then-default confidence 0.5:

| setting | top-1 interval accuracy | top candidate |
| --- | --- | --- |
| exact heights, confidence 0.5 | 0.60 | only the 6-semitone leap survives |
| exact heights, confidence 0.8 | 0.73 | 3-semitone moves recovered |
| heights drawn at half scale, confidence 0.8 | 0.53 | moves shrunk to 2 semitones |

The simulator ties confidence to the noise it applied (`conf = exp(-|noise|/1.5)`),
so to the model confidence 0.5 means "about one semitone of error", and it
then prefers its prior (repeats dominate POP909 melodies) over the drawing.
The UI default is now 0.8.

Adding a contour-fit term to the beam score (`TransformerReconstructor.guide_weight`,
Gaussian penalty on realised pitch vs drawn height, sigma from confidence) did
**not** help. On 300 POP909 test sketches:

```
preset  guide  top1_acc  best_of_5_acc  top5_exact
medium  0.0    0.527     0.691          0.217
medium  0.5    0.520     0.674          0.191
medium  1.0    0.502     0.661          0.157
medium  2.0    0.479     0.641          0.147
hard    0.0    0.415     0.534          0.084
hard    1.0    0.367     0.492          0.050
```

So the option stays at 0. The remaining gap for metal riffs is training data:
POP909 is pop vocal melody, so tritones, chromatic moves and chug patterns are
out of distribution. The next lever is adding guitar-riff data (DadaGP, Lakh
guitar tracks) to the training set.
