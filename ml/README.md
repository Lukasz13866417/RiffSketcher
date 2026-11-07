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
| `db.py` | the song database: ingest raw folders, split by song, export training phrases (see "Song database") |
| `train.py`, `evaluate.py` | CLIs (`evaluate` compares several checkpoints and reports per data source) |
| `prepare_dataset.py` | synthetic phrases for smoke tests only |
| `../experiments/chug_probe.py` | the 16-note chug-riff probe from the decoding experiments, run against any checkpoint |

## Data: POP909

```bash
mkdir -p data/raw && git clone --depth 1 https://github.com/music-x-lab/POP909-Dataset data/raw/pop909
uv run python -m ml.db ingest --pop909-dir data/raw/pop909/POP909
uv run python -m ml.db split && uv run python -m ml.db export --out data/processed/pop909.jsonl --sources pop909
```

Ingest takes the `MELODY` track of each song, makes it monophonic, converts
times to beats through the file's tempo map (many POP909 songs have tempo
changes) and stores the phrases; export writes `pop909.jsonl` plus
`splits.json`. The split is **by song** (train/val/test = 80/10/10) so no phrase of a test song is ever seen in training.

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
uv run python -m ml.db ingest --guitar-midi-dir data/raw/lakh_clean/clean_midi   # incremental: POP909 stays
uv run python -m ml.db split            # only the new songs are drawn; earlier assignments stay
uv run python -m ml.db export --out data/processed/mixed.jsonl --max-per-song 12
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
same song are renditions of one song row, so they never straddle a split.

`--max-per-song 12` keeps a random 12 phrases per guitar song (Lakh songs
have several guitar tracks and transcriptions and would otherwise outweigh
POP909 twenty to one); splits already assigned stay put when new songs are
ingested, so an earlier model is never evaluated on songs it trained on.
The first mixed file (built before the database existed, on which the result
tables below were measured) had 119,956 phrases (89,279 `lakh_guitar` from
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
songs as before, since split assignments are kept when sources are added), `mixed` is `runs/mixed/best.pt`.
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

### Song database (2026-11-06)

`ml/db.py` builds `data/riffs.db`, a SQLite file that holds everything we know
about every song, so new sources and extractions add rows instead of new
files. Tables:

| table | one row per | notes |
| --- | --- | --- |
| `source` | dataset | licence and provenance text |
| `song` | composition | `id` is the song id; artist/title keys (see below) match renditions across sources |
| `rendition` | file of one source | `id` is the rendition id; `original_id` is the source's own identifier; path, content hash, tempo, resolution; several per song (Lakh versions) |
| `track` | instrument track | role: melody / guitar / bass / drums / other; notes stored for melody, guitar, bass and drums (of drums only kick and snare hits) |
| `note` | note event | onset and duration in beats, seconds, pitch, velocity, string, fret, palm mute, bend, slide, hammer, vibrato, dead, tie |
| `extractor` | name + version + params | every derived row points at one |
| `phrase` | training phrase | `id`; pitches / onsets / durations as JSON; the track and the **ids of the note rows it covers**; the extractor that cut it |
| `split` | song | train / val / test, kept across runs and seedable |
| `song_feature`, `rendition_feature`, `phrase_feature`, `note_feature` | key/value | any extractor output at the right granularity (see "Features from stored data") |
| `frame_feature` | time-indexed value | signals not aligned to notes (audio curves) |

Ingest parses files in worker processes (`pretty_midi` for MIDI, the DadaGP
token reader for tabs) and stores all tracks but only the notes of melody
(POP909) and guitar (Lakh, DadaGP) tracks. Phrases are then **cut from the
stored note rows** by `cut_phrases` (the melody line of a melody track, the
riff line of a guitar track: lowest note per onset, strummed chords end a
phrase, rests and largest gaps split, exact repeats within a rendition kept
once — for melodies too, where the earlier file loader kept every repeated
chorus as a separate phrase), and each phrase stores the ids of the notes it
covers, so anything recorded per note
(articulation from tabs, a classifier's palm-mute probability) reaches
training through the phrase. The raw files are not needed after ingest;
`cut --recut` re-derives every phrase, e.g. after changing the rules. (Cut
from the stored rows, 99.85% of the phrases are identical to what the old
file loader cut; the rest are the melody repeats, now kept once, and runs
with tied largest gaps that split at a different tie, because a 32nd-note
grid sits on the four-decimal rounding boundary: 1.46875 beats.) Files
already present are skipped, so ingest is incremental. Exported phrases
carry `song_id`, `rendition_id` and `phrase_id`; `--max-per-song` caps
phrases per song for riff sources (`--cap-exempt pop909` by default, since
a melody source has one track per song).

```bash
uv run python -m ml.db ingest --pop909-dir data/raw/pop909/POP909 --guitar-midi-dir data/raw/lakh_clean/clean_midi
uv run python -m ml.db cut --recut             # phrases from the note rows (ingest does this for new files)
uv run python -m ml.db split --seed 0          # seeded draw over song ids; re-runs only assign new songs
uv run python -m ml.db export --out data/processed/mixed.jsonl --max-per-song 12
uv run python -m ml.db stats
uv run python -m ml.db sql "select role, count(*) from track group by role"
uv run python -m ml.tropes --db data/riffs.db --out data/tropes.json
```

**Matching songs across files and sources.** `artist_key` and `title_key`
(stored as `artist_norm` / `title_norm`) decide which song a file belongs
to: accents stripped, punctuation to spaces, "&" / "n" / "'n" / "and" all
read as "and", a leading article dropped ("The Beatles" = "Beatles"), and
parenthetical or " - Live / Remastered" title suffixes ignored, since they
mark a rendition rather than a different composition. So "Guns N' Roses",
"Guns 'n Roses", "Guns & Roses" and "Guns And Roses" are one artist and
"Sweet Child O' Mine (Live)" is the same song as "Sweet Child O'Mine". As a
backstop for typos, a file whose title key matches an existing song and
whose artist key is at least 0.85 similar (difflib ratio), or vice versa
with 0.92 on the title, joins that song. `uv run python -m ml.db dedupe
--dry-run` lists what the current rules would merge in an existing
database, `dedupe` applies it (the lowest id survives and keeps its split),
and `merge KEEP DROP` folds two songs by hand for the cases no rule
catches. Files without names (POP909 ids) are never matched.

Why it matters for the model: transitions and articulation become per-note
columns the moment any source fills them (DadaGP, scraped tabs, an audio
technique classifier), and a new output head reads them through the same
phrase query; palm mute as a model input is the same column.

Full ingest of POP909 and Lakh guitar: 909 + 16,977 renditions (279
unreadable MIDI files skipped) into 10,923 songs, 173,201 tracks, 22.0M notes
(melody and guitar tracks only) and 552,757 phrases cut from the note rows
(104,085 exported after the 12-per-song cap); 1.7 GB; 3,425 songs have more
than one rendition. Matching by normalised artist and title merges
Lakh's case variants of one title ("Stairway to Heaven" / "Stairway To
Heaven", "Y.M.C.A" / "Y.M.C.A."); a path-based song id, as the first mixed
file used, treated those as different songs and put six of them on both
sides of its split (e.g. "Sweet Child O' Mine" in train and test): too few
to move the result tables above, and impossible now that splits are per
song. Superseded by the joint-token decoder, see below; at the time, the
served pair (`runs/genre_facts/best.pt` for pitches,
`runs/genre_facts_rh/best.pt` for timing; see "Constrained decoding") is
trained on the note-linked export: teacher-forced validation accuracy 0.762
and 0.750; on 2,000 test sketches of the database split (medium noise, true
genre label) interval accuracy 0.594 with facts shown and 0.579 with facts
hidden, NLL 1.151, ECE 0.014, top-5 reconstruction 0.343; the rhythm head
scores 0.748 on gap classes against the quantiser's 0.655. These are a
point or two under the numbers in the "Constrained decoding" tables, which
were measured on the previous export: that export kept every repeated
chorus of a POP909 melody as its own test phrase, so its test set was
easier, and the two are not comparable. Earlier checkpoints were trained
on a different split and are not comparable on this test set either.

### Features from stored data, and genres from the artist (2026-11-07)

Two enrichments that need no new licence, because they read what is on disk.

**Bass and drum notes.** Ingest stores the notes of bass tracks and the kick
and snare hits of drum tracks next to the melody and guitar notes (hi-hats
and cymbals would triple the note table and carry no downbeat). `ml.db notes
--roles bass,drums` backfills a database ingested without them by re-reading
the files; it added 17.5M notes over 36,381 tracks of 16,097 renditions and
took the database from 1.7 to 3.2 GB. The file's first time signature is kept
in the rendition's `meta_json`.

**`ml/features.py`** writes the feature tables with extractor provenance, so
every value can be recomputed or replaced (`run --recompute`):

| level | extractor | keys |
| --- | --- | --- |
| rendition | `rendition_key v1` | key, mode, tonic, key_confidence (margin over the runner-up), key_corr: Krumhansl-Schmuckler fit of the duration-weighted pitch-class histogram of all stored pitched notes |
| rendition | `rendition_metre v1` | time_signature, beats_per_bar, swing (median within-beat position of off-beat lead notes: 0.5 straight, 0.67 swung), n_offbeat, doubled_tracks (guitar tracks copying an earlier one) |
| song | `song_key v1` | key, mode, tonic: the rendition keys weighted by confidence |
| phrase | `phrase_context v1` | bar_position (where the phrase starts in the bar), beats_per_bar, bass_hits / kick_hits / snare_hits (one 0/1 per note: does an accompaniment onset fall within 0.05 beats of it; value_num = the share) |
| phrase | `phrase_tropes v1` | tropes (the catalogue's labels), familiarity |

Context lives on the phrase, not the note: a phrase knows its start and its
notes, and four keys for 22M note rows would double the database again.
`note_feature` stays for things that are really per note (a palm-mute
classifier's probability). Over the current database: 4/4 in 16,733 of
17,886 renditions, then 3/4 (247), 2/4 (215), 6/8 and 12/8 (157 each); the
most common song keys are C, G, D, A and E major; mean swing 0.534 over
13,395 renditions with enough off-beat notes (most MIDI is straight); a riff
note falls with a bass note half the time, with the kick 31% and with the
snare 20% of the time. A tapped sketch has none of this context; it is what
a downbeat-aware model or a trope-by-position analysis would read.

**Genres from MusicBrainz** (`ml/genres.py`). The genre token used to come
from the data source alone, so every Lakh song was "rock". `genres lookup`
asks MusicBrainz for each of the 2,178 artists (one request per second, the
rate it asks for; the search itself takes 2-5 s, so a full pass is about
three hours) and caches the match, its community tags and the folded genre in
`data/artist_genres.json` (committed). Tags vote by keyword into the
vocabulary in `ml.datasets.GENRES` (unknown, pop, rock, metal, punk,
electronic, hiphop, jazz, blues, country, folk, classical, soul, reggae; the
first four keep their indices so older checkpoints still load), a downvoted
tag counts zero, and ties go to the more specific genre (pop punk is punk).
`genres apply` writes `song.genre` (POP909 songs are pop) and the exported
phrases carry it; training, evaluation and the catalogue use the song's genre
where there is one (`phrase_genre`) and the source default elsewhere.
Genre counts after the lookup (songs): rock 3,733, pop 2,890 (909 of them
POP909), unknown 1,611 (no MusicBrainz match above score 90, or tags that
name no genre), soul 571, jazz 423, electronic 389, folk 306, classical 225,
metal 199, country 168, punk 151, hiphop 128, blues 86, reggae 43. So the
Lakh guitar tracks are far from all rock: a third of the labelled songs are
something else, which the source-only label hid. `data/processed/
mixed_genres.jsonl` is the export with these labels; `mixed.jsonl` keeps
the source labels so earlier results stay comparable.

### Timing branch and note durations: one model (2026-11-07)

Until now timing came from a second checkpoint, because a gap head sitting
directly on the shared encoder cost two points of interval accuracy when
trained jointly. The model is now one network with one trunk and branches
that diverge where the tasks do:

* **trunk**: input projection, genre embedding, `n_layers` encoder layers,
  shared by everything;
* **pitch branch**: the decoder and the interval classifier, unchanged;
* **timing branch** (`ModelConfig.n_branch` encoder layers of its own on top
  of the trunk, `ml.train --rhythm --branch-layers 2`): the gap head (one
  class per gap, as before) and a new **duration head** (one class per note).

Training is one supervised run on one scalar: interval cross-entropy plus
`--rhythm-weight` times the gap loss plus `--duration-weight` times the
duration loss, one optimiser over all parameters. `--branch-detach` is a
control: the timing branch reads the trunk but its gradient stops there, so
the trunk is shaped by pitch alone and timing has only its own layers.

**Durations.** The rhythm head predicts only the gap to the next onset, and
every note was held for that gap, so a riff of choked notes and rests came
out legato. Two additions: the tapping step records how long each tap was
held (key down to key up, `Observation.hold`), and the model gets it as a
share of the gap to the next tap plus a known flag (feature columns 14 and
15; the simulator draws holds from the true lengths with log-normal noise,
for 70% of sketches at medium noise). The duration head predicts, per note,
`DURATION_CLASSES`: short (0.25 of the gap), half, most (0.8), full, over
(1.3, ringing into the next note); the last note is judged against the
median gap. `TransformerReconstructor` scales each gap by the predicted
class (`riff.meta["durations"] == "model"`); without the head it is legato as
before. `ml.evaluate` reports duration accuracy against two baselines, legato
(every note "full") and the hold time read off directly.

Results so far (same export, 20 epochs, medium noise; test = 2,000 sketches
with facts, all four re-simulated with hold times, so the quantiser baseline
reads 0.648 here against 0.655 in the tables above; gap-only models, no
duration head yet):

```
model                                   val intervals  val gaps   test intervals  test gaps
genre_facts (pitch only, served)            0.762         -           0.588          -
genre_facts_rh (gap head on the trunk)      0.750       0.716         0.584        0.747
branch, joint (2 timing layers)             0.753       0.726         0.588        0.755
branch, trunk detached                      0.762       0.656         0.594        0.690
```

The trunk matters to timing: letting the gap loss shape it is worth six
points of gap accuracy over the detached control. The joint branch model
matches the pitch-only model on test intervals and beats the old head on
gaps, in one network, so it replaces the pair. Two runs with the duration head and gap/duration
loss weights 1.0 and 0.5 (`runs/branch_dur_w1`, `runs/branch_dur_w05`) were
started to pick the final weights; they were interrupted by a shutdown and
need rerunning, after which the one chosen model replaces the pair, is
retrained once more on the export with real genres, and the duration
accuracy table goes here.

### Joint-token decoder (2026-11-08)

The branches above keep timing as a per-note denoising task on the encoder.
The more general design, and the one that can later emit notes that are not
in the input at all (fill-ins, transitions), is a decoder whose every step
emits a whole note: `ModelConfig.joint` (`ml.train --joint`). Step k emits
the interval k→k+1, the gap k→k+1 and the duration of note k+1 with
factorised heads on one decoder state, in a fixed order within the step
(interval; gap given the interval; duration given both), and takes the
previous step's three tokens as input next to the local encoder evidence.
Note 0's duration, which no step covers, comes from its encoder state. So
timing conditions on pitch and pitch on the previous timing.

Two things that broke the first decoder-side attempt are handled:

* **Compounding.** Training feeds back the true previous tokens; free
  decoding feeds back the model's own. `--ss-max` (default 0.25) is
  scheduled sampling: a share of the fed-back tokens, ramping from 0 to that
  value over the run, is replaced by the model's own greedy prediction from
  a no-grad pass, so the decoder has seen its own mistakes before it is
  judged on them.
* **The vocabulary.** 49 × 17 × 5 joint classes would be 4,165 outputs;
  factorised heads keep three small ones and still condition each on the
  earlier choices in the step.

Decoding (`TransformerReconstructor._decode_joint`): the beam branches on
intervals under the same fact masks as before, and each hypothesis carries
the gaps and durations the decoder chose for its intervals, chosen greedily
within the step. Candidates therefore differ in timing as well as pitch.

The inputs also grew from 16 to 22 columns: position in the sketch, sketch
length, the ratio of the next IOI to the previous (accelerating or slowing),
the quantised gap to the *next* note, and the tempo hint with a flag.

Results on the same 2,000 test sketches (medium noise, facts shown, free
decoding; source genre labels, 20 epochs, the large model 16). Interval
accuracy comes from sampled candidates and moves by about half a point
with the random state, so the branch model reads 0.584 here and 0.588 in
the table above:

```
model                                    inputs  params  val intervals (TF)  test intervals  top-5   gaps    durations
branch, gap head only                      16     2.4M        0.753              0.584       0.301   0.755    -  (legato)
branch, gap + duration heads               22     2.4M        0.750              0.581       0.284   0.803   0.661
joint, d_model 128, 4+4 layers             22     1.9M        0.726              0.580       0.241   0.774   0.641
joint, d_model 192, 6+6 layers             22     6.6M        0.741              0.591       0.281   0.787   0.647
baselines                                                                                           0.648 q  0.419 legato / 0.426 hold
```

Reading it honestly:

* The six extra input columns did most of the timing work: the branch model
  went from 0.755 to 0.803 on gaps when it got them (the gap to the *next*
  note and the IOI ratio are exactly what a gap head wants).
* With equal inputs the two designs tie within noise on intervals and top-5
  reconstruction (0.581/0.284 vs 0.591/0.281), and the branch model is a
  point or two *ahead* on gaps and durations at a third of the parameters
  and with cheaper decoding (one encoder pass for all timing, no per-step
  timing heads in the beam).
* Teacher-forced validation accuracy is lower for the joint decoder yet its
  free decoding holds up: it is judged on its own previous timing tokens in
  validation too, and scheduled sampling trades teacher-forced fit for
  robustness. Capacity matters for it: the small joint model loses six
  points of top-5 reconstruction, the large one recovers it.
* Both learned duration heads are right for about two notes in three,
  against 42% for the legato rule or for reading the tap hold time alone.

So the joint decoder is not a better *denoiser* than the branch; its case is
what it can emit that the branch cannot, notes that are not in the input
(fill-ins, transitions), and one search over pitch and timing. It is the
served model for that reason, retrained on the genre-labelled export
(`runs/joint_large_genres`: on the genre-labelled test sketches, intervals
0.588 / gaps 0.783 / durations 0.642 with the song's genre shown, and
0.578 / 0.777 / 0.634 with it hidden, so the fourteen-way token is worth
about a point on each axis, roughly what the two-way one was). The branch model retrained on
the same export is `runs/branch_dur_genres`: intervals 0.571, top-5 0.279,
gaps 0.804, durations 0.653 with the genre shown, so on equal data the
joint model leads by 1.7 points on intervals and trails by 3.4 on top-5
reconstruction, 2 on gaps and 1 on durations; a draw, decided by what the
joint decoder can grow into. The server takes either through `RIFF_MODEL`.

### Constrained decoding and facts as inputs (2026-11-06)

Two things used to happen *after* the model: timing came from the rule-based
quantiser, and user facts were checked by dropping candidates that broke
them, with the deterministic engine filling the gap. Both now happen in or
before the model.

**Facts inside decoding.** `TransformerReconstructor._decode` runs the beam
search and the diversity samples itself. At every step the interval classes
that would break a fact are masked out and the remaining probabilities
renormalise: a direction fact keeps one sign, an exact-interval fact or a
"same as note i" fact keeps one class, and a pinned pitch keeps the one
class that lands on it, which is possible because the absolute anchor is
fixed *before* decoding from the first pinned note and the drawn offset to
it (`_pre_anchor`). A candidate therefore cannot violate a satisfiable fact;
only mutually contradictory facts leave the mask empty, in which case that
step is decoded unmasked and the old filter plus fallback still apply. Scores
shown in the UI are the model's log-probabilities under the facts.

**Facts and the grid snap as inputs.** `sketch_to_features` grew from 6 to
14 continuous columns per note (documented in its docstring): the
quantiser's snapped gap from the previous note (log2 beats), pinned flag and
pinned pitch relative to the first pin, the known direction of the incoming
interval, and "same as an earlier note" with the distance back. The
simulator (`SimConfig.facts_prob`, `pin_prob`, `direction_prob`,
`same_prob`) generates true facts from the phrase so the model learns what a
pin implies for its neighbours and where the quantiser errs. Checkpoints
trained on the 6-column features still load: the encoder slices the prefix
it was trained on (`ModelConfig.n_cont`).

Measured on 2,000 test sketches (medium noise, half of them carrying
simulated facts; `runs/eval_facts_on.log`, `runs/eval_facts_off.log`; the
evaluate script decodes without masks, so these numbers isolate the input
conditioning):

```
model                              intervals, facts shown   facts hidden   gap accuracy
genre        (6 inputs, no facts)         0.602                0.598        quantiser 0.669
genre_facts  (14 inputs)                  0.612                0.596        quantiser 0.669
genre_facts_rh (14 inputs + rhythm head)  0.589                0.582        model     0.774
```

Takeaways:

* Facts as inputs give +1 point when facts are present and cost nothing
  when they are not. The masks add more on top: on the 560 test sketches
  that carry facts, top-1 interval accuracy is 0.584 with an unmasked beam
  and 0.609 with the constrained beam, and the fallback engine is needed for
  9 of them (facts whose implications span several steps; a first version
  that derived note 0's pitch from the drawn offset to the first pin scored
  0.544 and fell back 73 times, which is why pins are enforced relative to
  the earliest pin, pins propagate through "same" chains, and the mask looks
  one step ahead). Decoding costs about 80 ms per sketch on the GPU.
* The rhythm head with the quantiser's gap as input beats the rule by ten
  points (0.774 vs 0.669), where without that input it only tied. Timing is
  now the model's. But sharing the encoder with the rhythm loss costs two
  points of interval accuracy, so the API serves **two checkpoints**:
  `runs/genre_facts/best.pt` for pitches (`RIFF_MODEL`) and
  `runs/genre_facts_rh/best.pt` for gaps (`RIFF_RHYTHM_MODEL`); each is
  used where it is best, and `GET /api/engines` reports both.
* A joint model that matches both numbers would need a smaller rhythm loss
  weight or separate encoder layers; not tried yet.

Reproduce:

```bash
uv run python -m ml.train --data data/processed/mixed.jsonl --epochs 20 --sim medium --genre-dropout 0.2 --out runs/genre_facts
uv run python -m ml.train --data data/processed/mixed.jsonl --epochs 20 --sim medium --genre-dropout 0.2 --rhythm --out runs/genre_facts_rh
uv run python -m ml.evaluate --checkpoint runs/genre/best.pt runs/genre_facts/best.pt runs/genre_facts_rh/best.pt --data data/processed/mixed.jsonl --split test --sim medium --no-deterministic --limit 2000
```

### Trope catalogue (2026-11-06)

`ml/tropes.py` counts recurring patterns in every phrase of the song database,
per genre, and writes `data/tropes.json` (113 KB, committed). Everything is a document
frequency (once per song), so a song that repeats one riff does not dominate.
It holds, per genre and overall: interval unigrams; interval, rhythm and
combined "figure" n-grams (2-5) with the share of songs containing each;
named tropes from rule detectors with example phrases; the smallest scale
family each phrase fits; and compressed contour words.

Named tropes (share of songs), POP909 melodies vs Lakh guitar riff lines (all
538k guitar phrases, not the per-song training sample):

| trope | pop | rock |
| --- | --- | --- |
| chug run (3+ repeated notes) | 58% | 79% |
| pedal tone | 76% | 69% |
| repeated figure (exact interval period) | 17% | 66% |
| tritone move | 5% | 44% |
| octave leap | 43% | 69% |
| gallop (eighth + two sixteenths, twice) | 20% | 22% |
| pentatonic lick (4+ pitch classes, two minor-third hops) | 70% | 48% |
| descending run | 54% | 44% |
| ascending run | 35% | 35% |
| ends on its lowest note | 92% | 89% |

Interval unigrams: repeats are 22% of pop intervals and 33% of rock; whole
steps are 38% of pop and 15% of rock. Top rock trigram `[0, 0, 0]` appears in
79% of songs; top pop trigrams are `[-2, -2, 2]` and `[-2, 2, -2]` (62% / 61%).
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
now has an optional **rhythm head** (`ModelConfig.n_rhythm`, off by default,
`ml.train --rhythm` turns it on): one class per gap between consecutive
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
* The head costs about a point of interval accuracy, so at this point the
  served default kept rule-based timing. Superseded: with the quantiser's
  gap as an input the head beats the rule by ten points, see "Constrained
  decoding and facts as inputs" below, and timing is now served from a
  second checkpoint.
* Absolute beat labels are inherently ambiguous from tempo-normalised inputs
  (is the common gap a quarter or an eighth?), which caps both methods; the
  tempo-invariant numbers are the ones to watch.
* What would move this: real tapping data to replace the simulator's jitter
  model (the hard preset is a guess), and joint beam search over gaps and
  intervals instead of greedy gaps.

Reproduce:

```bash
uv run python -m ml.train --data data/processed/mixed.jsonl --epochs 20 --sim medium --genre-dropout 0.2 --rhythm --out runs/rhythm
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
