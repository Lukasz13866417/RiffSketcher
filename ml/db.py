"""The song database: everything we know about every song, in one SQLite file.

Design
------
``song``       the composition (title, artist, genre). One row per song however
               many files we hold of it. Splits attach here, so a song is never
               in training through one file and in test through another.
``rendition``  one file of one source (a Lakh MIDI, a DadaGP tab, an audio
               recording ...) pointing at its song; carries the file's own
               facts (path, hash, tempo, resolution).
``track``      instrument tracks of a rendition, with a role (melody, guitar,
               bass, drums, other).
``note``       the per-moment table: one row per note event of the tracks we
               keep, with pitch, timing, velocity, string/fret and typed
               articulation columns (palm mute, bend, slide, hammer-on,
               vibrato, dead, tie) that tabs and audio extraction fill in.
``extractor``  name + version + params of anything that derives data; every
               derived row points at one so it can be re-run and replaced.
``phrase``     a training phrase cut from a track's note rows by ``cut_phrases``
               (the melody line or the riff line, see ``ml.datasets``), stored
               with its pitches/onsets/durations and the ids of the note rows it
               covers, so per-note data reaches training through the phrase.
``split``      song -> train/val/test.
``*_feature``  key/value rows per song, rendition or note from any extractor,
               and ``frame_feature`` for time-indexed signals not aligned to
               notes (loudness curves and the like).

Identifiers: ``song.id`` is the song id and ``rendition.id`` the rendition id,
both integers assigned at ingest. A source's own identifier (POP909's folder
number, Lakh's relative path, a DadaGP file name) is kept as
``rendition.original_id`` for provenance only. Songs are matched across
sources by normalised artist and title; sources that carry no names get one
song per rendition. Exported phrases carry ``song_id = song.id`` and
``rendition_id``.

CLI::

    uv run python -m ml.db init
    uv run python -m ml.db ingest --pop909-dir data/raw/pop909/POP909 --guitar-midi-dir data/raw/lakh_clean/clean_midi
    uv run python -m ml.db cut --recut      # ingest cuts phrases for new files; this recuts everything
    uv run python -m ml.db split --seed 0
    uv run python -m ml.db export --out data/processed/mixed.jsonl --max-per-song 12
    uv run python -m ml.db stats
    uv run python -m ml.db sql "select role, count(*) from track group by role"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ml.datasets import (MAX_RIFF_CHORD, Phrase, _is_guitar_track, _monophonic_notes,
                         _onsets_from_dadagp_events, _pick_melody_instrument, dadagp_track_events,
                         melody_line_phrases, riff_line_phrases_indexed, save_phrases)

DEFAULT_DB = Path("data/riffs.db")

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS source (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, licence TEXT, notes TEXT);

CREATE TABLE IF NOT EXISTS song (
    id INTEGER PRIMARY KEY,
    title TEXT, artist TEXT, title_norm TEXT, artist_norm TEXT, genre TEXT, notes TEXT);
CREATE INDEX IF NOT EXISTS song_norm ON song(artist_norm, title_norm);

CREATE TABLE IF NOT EXISTS rendition (
    id INTEGER PRIMARY KEY, song_id INTEGER NOT NULL REFERENCES song(id),
    source_id INTEGER NOT NULL REFERENCES source(id), original_id TEXT NOT NULL,
    path TEXT, content_hash TEXT, format TEXT, tempo REAL, resolution INTEGER, meta_json TEXT,
    UNIQUE(source_id, original_id));
CREATE INDEX IF NOT EXISTS rendition_work ON rendition(song_id);

CREATE TABLE IF NOT EXISTS track (
    id INTEGER PRIMARY KEY, rendition_id INTEGER NOT NULL REFERENCES rendition(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL, name TEXT, program INTEGER, role TEXT, is_drum INTEGER DEFAULT 0,
    tuning_json TEXT, n_notes INTEGER DEFAULT 0, notes_stored INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS track_rendition ON track(rendition_id);

CREATE TABLE IF NOT EXISTS note (
    id INTEGER PRIMARY KEY, track_id INTEGER NOT NULL REFERENCES track(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL, onset_beats REAL NOT NULL, duration_beats REAL NOT NULL, onset_s REAL,
    pitch INTEGER NOT NULL, velocity INTEGER, string INTEGER, fret INTEGER,
    palm_mute INTEGER, bend REAL, slide INTEGER, hammer INTEGER, vibrato INTEGER, dead INTEGER, tie INTEGER);
CREATE INDEX IF NOT EXISTS note_track ON note(track_id, idx);

CREATE TABLE IF NOT EXISTS extractor (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL, params_json TEXT,
    UNIQUE(name, version, params_json));

CREATE TABLE IF NOT EXISTS phrase (
    id INTEGER PRIMARY KEY, rendition_id INTEGER NOT NULL REFERENCES rendition(id) ON DELETE CASCADE,
    track_id INTEGER REFERENCES track(id) ON DELETE SET NULL,
    extractor_id INTEGER NOT NULL REFERENCES extractor(id), idx INTEGER NOT NULL,
    start_beats REAL, n_notes INTEGER NOT NULL,
    pitches_json TEXT NOT NULL, onsets_json TEXT NOT NULL, durations_json TEXT NOT NULL, note_ids_json TEXT);
CREATE INDEX IF NOT EXISTS phrase_rendition ON phrase(rendition_id);

CREATE TABLE IF NOT EXISTS split (
    song_id INTEGER PRIMARY KEY REFERENCES song(id) ON DELETE CASCADE, name TEXT NOT NULL, seed INTEGER);

CREATE TABLE IF NOT EXISTS song_feature (
    song_id INTEGER NOT NULL REFERENCES song(id) ON DELETE CASCADE,
    extractor_id INTEGER NOT NULL REFERENCES extractor(id), key TEXT NOT NULL,
    value_num REAL, value_text TEXT, PRIMARY KEY (song_id, extractor_id, key));
CREATE TABLE IF NOT EXISTS rendition_feature (
    rendition_id INTEGER NOT NULL REFERENCES rendition(id) ON DELETE CASCADE,
    extractor_id INTEGER NOT NULL REFERENCES extractor(id), key TEXT NOT NULL,
    value_num REAL, value_text TEXT, PRIMARY KEY (rendition_id, extractor_id, key));
CREATE TABLE IF NOT EXISTS phrase_feature (
    phrase_id INTEGER NOT NULL REFERENCES phrase(id) ON DELETE CASCADE,
    extractor_id INTEGER NOT NULL REFERENCES extractor(id), key TEXT NOT NULL,
    value_num REAL, value_text TEXT, PRIMARY KEY (phrase_id, extractor_id, key));

CREATE TABLE IF NOT EXISTS note_feature (
    note_id INTEGER NOT NULL REFERENCES note(id) ON DELETE CASCADE,
    extractor_id INTEGER NOT NULL REFERENCES extractor(id), key TEXT NOT NULL,
    value_num REAL, value_text TEXT, PRIMARY KEY (note_id, extractor_id, key));
CREATE TABLE IF NOT EXISTS frame_feature (
    rendition_id INTEGER NOT NULL REFERENCES rendition(id) ON DELETE CASCADE,
    extractor_id INTEGER NOT NULL REFERENCES extractor(id), key TEXT NOT NULL,
    t_beats REAL NOT NULL, value_num REAL NOT NULL);
CREATE INDEX IF NOT EXISTS frame_feature_idx ON frame_feature(rendition_id, key, t_beats);
"""

SOURCES = {
    "pop909": ("POP909 (Wang et al. 2020): 909 Chinese pop songs with labelled melody tracks; "
               "research use, see data/raw/pop909/LICENSE"),
    "lakh_guitar": ("Lakh MIDI Dataset clean_midi subset (Raffel 2016), guitar tracks; "
                    "CC-BY 4.0 for the collection, underlying compositions are copyrighted"),
    "dadagp": "DadaGP (Sarmento et al. 2021): GuitarPro tabs as tokens; research-only, by request",
    "guitarset": ("GuitarSet (Xi et al. 2018): 360 real guitar excerpts with per-string note transcriptions, "
                  "beats, tempo and key; CC BY 4.0 (annotations from Zenodo record 3371780)"),
    "lakh_full": ("Lakh MIDI Dataset full set (Raffel 2016), files named by MD5 with no artist or title, guitar "
                  "tracks only; files already present from clean_midi are skipped by content hash; CC-BY 4.0 for "
                  "the collection, underlying compositions are copyrighted"),
}

# GuitarSet file names: <player>_<style><progression>-<bpm>-<key>_<comp|solo>.jams
GUITARSET_STYLE_GENRE = {"BN": "jazz", "Funk": "soul", "Jazz": "jazz", "Rock": "rock", "SS": "folk"}
GUITARSET_OPEN_STRINGS = (40, 45, 50, 55, 59, 64)  # string index 0 = low E

_NORM_RE = re.compile(r"[^a-z0-9]+")
_PAREN_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")  # "(Live)", "[Remastered]"
_SUFFIX_RE = re.compile(r"\s+-\s+(live|remaster(ed)?( \d{4})?|single version|album version|radio edit|demo|acoustic|mono|stereo)\b.*$")
_AND_WORDS = {"&", "n", "and", "'n", "'n'", "+"}
_LEAD_ARTICLES = ("the ", "a ", "an ", "los ", "las ", "le ", "la ", "les ", "die ", "der ", "das ")


def norm_text(s: str | None) -> str:
    """Lowercase ASCII words: accents stripped, punctuation to spaces."""
    import unicodedata

    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return _NORM_RE.sub(" ", s.lower()).strip()


def artist_key(artist: str | None) -> str:
    """Matching key for an artist name: "Guns N' Roses", "Guns 'n Roses",
    "Guns & Roses" and "Guns And Roses" all give "guns and roses"; a leading
    article is dropped ("The Beatles" -> "beatles")."""
    raw = (artist or "").lower().strip()
    for art in _LEAD_ARTICLES:
        if raw.startswith(art):
            raw = raw[len(art):]
            break
    words = [("and" if w.strip("'") in {"n", "and"} or w in _AND_WORDS else w) for w in raw.replace("&", " & ").split()]
    return norm_text(" ".join(words))


def title_key(title: str | None) -> str:
    """Matching key for a title: parenthetical and " - live/remastered" suffixes
    dropped (they mark a rendition, not a different composition)."""
    raw = (title or "").lower().strip()
    raw = _PAREN_RE.sub("", raw)
    raw = _SUFFIX_RE.sub("", raw)
    return norm_text(raw) or norm_text(title)


def similar(a: str, b: str) -> float:
    """0..1 string similarity (difflib ratio), used as a typo backstop."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


ARTIST_FUZZY = 0.85  # same title key, artist keys this similar -> same song
TITLE_FUZZY = 0.92  # same artist key, title keys this similar -> same song


# --------------------------------------------------------------------------- #
# connection helpers
# --------------------------------------------------------------------------- #


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    for name, notes in SOURCES.items():
        con.execute("INSERT OR IGNORE INTO source(name, notes) VALUES (?, ?)", (name, notes))
    con.commit()


def source_id(con: sqlite3.Connection, name: str) -> int:
    row = con.execute("SELECT id FROM source WHERE name = ?", (name,)).fetchone()
    if row is None:
        con.execute("INSERT INTO source(name) VALUES (?)", (name,))
        return source_id(con, name)
    return int(row["id"])


def extractor_id(con: sqlite3.Connection, name: str, version: str = "1", params: dict | None = None) -> int:
    pj = json.dumps(params or {}, sort_keys=True)
    row = con.execute("SELECT id FROM extractor WHERE name=? AND version=? AND params_json=?",
                      (name, version, pj)).fetchone()
    if row is None:
        con.execute("INSERT INTO extractor(name, version, params_json) VALUES (?,?,?)", (name, version, pj))
        return extractor_id(con, name, version, params)
    return int(row["id"])


def find_song(con: sqlite3.Connection, title_k: str, artist_k: str) -> int | None:
    """The song whose keys match exactly, else the best fuzzy match
    (same title key and artist similarity >= ARTIST_FUZZY, or same artist key
    and title similarity >= TITLE_FUZZY); None when nothing qualifies."""
    if not title_k or not artist_k:
        return None
    row = con.execute("SELECT id FROM song WHERE title_norm=? AND artist_norm=?", (title_k, artist_k)).fetchone()
    if row is not None:
        return int(row["id"])
    best, best_score = None, 0.0
    for r in con.execute("SELECT id, artist_norm FROM song WHERE title_norm=?", (title_k,)):
        sc = similar(artist_k, r["artist_norm"] or "")
        if sc >= ARTIST_FUZZY and sc > best_score:
            best, best_score = int(r["id"]), sc
    for r in con.execute("SELECT id, title_norm FROM song WHERE artist_norm=?", (artist_k,)):
        sc = similar(title_k, r["title_norm"] or "")
        if sc >= TITLE_FUZZY and sc > best_score:
            best, best_score = int(r["id"]), sc
    return best


def get_or_create_song(con: sqlite3.Connection, title: str | None, artist: str | None,
                       genre: str | None = None) -> int:
    """The song this artist+title refers to, created if absent.

    Matching uses :func:`artist_key` / :func:`title_key` (stored as
    ``artist_norm`` / ``title_norm``) with :func:`find_song`'s fuzzy backstop.
    Without both names there is nothing to match on, so a new song is created.
    """
    tk, ak = title_key(title), artist_key(artist)
    found = find_song(con, tk, ak)
    if found is not None:
        return found
    cur = con.execute("INSERT INTO song(title, artist, title_norm, artist_norm, genre) VALUES (?,?,?,?,?)",
                      (title, artist, tk or None, ak or None, genre))
    return int(cur.lastrowid)


def merge_songs(con: sqlite3.Connection, keep: int, drop: int) -> None:
    """Fold song ``drop`` into ``keep``: renditions move over, features move
    unless the key already exists on ``keep``, the split of ``keep`` wins."""
    if keep == drop:
        return
    con.execute("UPDATE rendition SET song_id=? WHERE song_id=?", (keep, drop))
    con.execute("UPDATE OR IGNORE song_feature SET song_id=? WHERE song_id=?", (keep, drop))
    con.execute("DELETE FROM song_feature WHERE song_id=?", (drop,))
    if con.execute("SELECT 1 FROM split WHERE song_id=?", (keep,)).fetchone() is None:
        con.execute("UPDATE OR IGNORE split SET song_id=? WHERE song_id=?", (keep, drop))
    con.execute("DELETE FROM split WHERE song_id=?", (drop,))
    con.execute("DELETE FROM song WHERE id=?", (drop,))


def dedupe_songs(con: sqlite3.Connection, dry_run: bool = False, log=print) -> list[tuple[int, int, str]]:
    """Merge songs that the current matching rules consider the same.

    Re-keys every song with the current :func:`artist_key` / :func:`title_key`
    (so a database built with older rules catches up), then merges fuzzy
    duplicates into the lowest id. Returns ``(keep, drop, reason)`` triples.
    """
    rows = con.execute("SELECT id, title, artist FROM song ORDER BY id").fetchall()
    for r in rows:
        con.execute("UPDATE song SET title_norm=?, artist_norm=? WHERE id=?",
                    (title_key(r["title"]) or None, artist_key(r["artist"]) or None, r["id"]))
    merges: list[tuple[int, int, str]] = []
    seen_drop: set[int] = set()
    for r in con.execute("SELECT id, title_norm, artist_norm FROM song WHERE title_norm IS NOT NULL AND artist_norm IS NOT NULL ORDER BY id").fetchall():
        if r["id"] in seen_drop:
            continue
        cands = con.execute(
            "SELECT id, title_norm, artist_norm FROM song WHERE id > ? AND (title_norm=? OR artist_norm=?)",
            (r["id"], r["title_norm"], r["artist_norm"])).fetchall()
        for c in cands:
            if c["id"] in seen_drop:
                continue
            same = (c["title_norm"] == r["title_norm"] and c["artist_norm"] == r["artist_norm"])
            reason = "exact keys" if same else None
            if not same and c["title_norm"] == r["title_norm"]:
                sc = similar(r["artist_norm"], c["artist_norm"])
                if sc >= ARTIST_FUZZY:
                    reason = f"artist similarity {sc:.2f}"
            if not same and reason is None and c["artist_norm"] == r["artist_norm"]:
                sc = similar(r["title_norm"], c["title_norm"])
                if sc >= TITLE_FUZZY:
                    reason = f"title similarity {sc:.2f}"
            if reason:
                merges.append((int(r["id"]), int(c["id"]), reason))
                seen_drop.add(int(c["id"]))
                if not dry_run:
                    merge_songs(con, int(r["id"]), int(c["id"]))
    if dry_run:
        con.rollback()
    else:
        con.commit()
    log(f"{len(merges)} merges{' (dry run)' if dry_run else ''}")
    return merges


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# parsing one file into plain dicts (runs in worker processes)
# --------------------------------------------------------------------------- #

Parsed = dict[str, Any]  # {"tracks": [...], "phrases": [...], "tempo":..., "resolution":..., "meta":...}


# Roles whose notes are stored. Melody and guitar notes are what phrases are cut from; bass and
# drum notes are the metric context (ml/features.py: which riff notes land with the bass, the
# kick or the snare). Of a drum track only kick and snare hits are kept: hi-hats and cymbals
# would triple the note table for little, and the two carry the downbeats and backbeats.
STORED_ROLES: tuple[str, ...] = ("melody", "guitar", "bass", "drums")
DRUM_KEEP: frozenset[int] = frozenset({35, 36, 38, 40})  # GM kick (35, 36) and snare (38, 40)


def _midi_role(ins: Any, melody_ins: Any) -> str:
    if ins.is_drum:
        return "drums"
    if melody_ins is not None and ins is melody_ins:
        return "melody"
    if "bass" in ins.name.lower() or 32 <= ins.program < 40:
        return "bass"
    if _is_guitar_track(ins):
        return "guitar"
    return "other"


def parse_midi(path: Path, source: str, original_id: str, keep_roles: tuple[str, ...], min_notes: int,
               max_notes: int) -> Parsed:
    """Tracks (+ notes for ``keep_roles``) of one MIDI file; phrases are cut later from the note rows."""
    import pretty_midi

    warnings.simplefilter("ignore")
    pm = pretty_midi.PrettyMIDI(str(path))
    res = float(pm.resolution)

    def to_beats(t: float) -> float:
        return float(pm.time_to_tick(t)) / res

    melody_ins = _pick_melody_instrument(pm) if source == "pop909" else None
    tracks = []
    for idx, ins in enumerate(pm.instruments):
        role = _midi_role(ins, melody_ins)
        notes = []
        if role in keep_roles and ins.notes:
            src_notes = _monophonic_notes(ins.notes) if role == "melody" else sorted(ins.notes, key=lambda n: (n.start, n.pitch))
            if role == "drums":
                src_notes = [n for n in src_notes if int(n.pitch) in DRUM_KEEP]
            for k, n in enumerate(src_notes):
                s, e = to_beats(n.start), to_beats(n.end)
                # six decimals: a 480-tick quarter note is 1/480 = 0.002083 beats, and phrase onsets are
                # differences of these values rounded to four, so the store must be finer than that
                notes.append((k, round(s, 6), round(max(e - s, 0.0), 6), round(float(n.start), 6),
                              int(n.pitch), int(n.velocity), None, None))
        tracks.append({"idx": idx, "name": ins.name, "program": int(ins.program), "role": role,
                       "is_drum": int(ins.is_drum), "n_notes": len(ins.notes), "notes": notes})
    tempos = pm.get_tempo_changes()[1]
    ts = pm.time_signature_changes
    meta = {"n_instruments": len(pm.instruments), "n_tempos": int(len(tempos)), "n_time_signatures": len(ts),
            "time_signature": f"{ts[0].numerator}/{ts[0].denominator}" if ts else None}
    return {"tracks": tracks,
            "tempo": float(tempos[0]) if len(tempos) else None, "resolution": int(pm.resolution),
            "format": path.suffix.lower().lstrip("."), "meta": meta}


def parse_dadagp(path: Path, source: str, original_id: str, min_notes: int, max_notes: int) -> Parsed:
    """Guitar tracks of a DadaGP token file; the riff line (lowest note per onset) is stored."""
    with path.open(encoding="utf-8", errors="replace") as f:
        tracks_ev = dadagp_track_events(f)
    tracks = []
    for idx, (name, events) in enumerate(tracks_ev.items()):
        notes = []
        for k, (s, e, pitches) in enumerate(_onsets_from_dadagp_events(events)):
            if pitches:
                notes.append((k, round(s, 6), round(max(e - s, 0.0), 6), None, int(min(pitches)), None, None, None))
        tracks.append({"idx": idx, "name": name, "program": None, "role": "guitar", "is_drum": 0,
                       "n_notes": len(notes), "notes": notes})
    return {"tracks": tracks, "tempo": None, "resolution": 960,
            "format": "dadagp", "meta": {}}


def guitarset_name(stem: str) -> tuple[str, str, str, str]:
    """(player, title, style, mode) from a GuitarSet stem such as ``00_BN1-129-Eb_comp``;
    the title ``BN1-129-Eb`` names the progression, so comp and solo takes by every
    player are renditions of one song."""
    m = re.match(r"(\d+)_([A-Za-z]+)(\d+-\d+-[A-G][b#]?)_(comp|solo)$", stem)
    if not m:
        return "", stem, "", ""
    return m.group(1), m.group(2) + m.group(3), m.group(2), m.group(4)


def parse_guitarset(path: Path, source: str, original_id: str) -> Parsed:
    """One GuitarSet JAMS file: one guitar track with every note's string and fret,
    seconds mapped to beats through the annotated beat grid."""
    j = json.loads(Path(path).read_text())
    anns = j["annotations"]
    tempo_ann = next((a for a in anns if a["namespace"] == "tempo"), None)
    tempo = float(tempo_ann["data"][0]["value"]) if tempo_ann and tempo_ann["data"] else 120.0
    beats = next((a for a in anns if a["namespace"] == "beat_position"), None)
    beat_times = np.array(sorted(d["time"] for d in beats["data"]), dtype=float) if beats and beats["data"] else np.zeros(0)
    bps = tempo / 60.0

    def to_beats(t: float) -> float:
        if len(beat_times) >= 2:
            if t < beat_times[0]:
                return float((t - beat_times[0]) * bps)
            if t > beat_times[-1]:
                return float(len(beat_times) - 1 + (t - beat_times[-1]) * bps)
            return float(np.interp(t, beat_times, np.arange(len(beat_times))))
        return float(t * bps)

    notes_raw = []
    for a in anns:
        if a["namespace"] != "note_midi":
            continue
        try:
            string = int(a["annotation_metadata"].get("data_source"))
        except (TypeError, ValueError):
            string = None
        for d in a["data"]:
            pitch = int(round(float(d["value"])))
            fret = pitch - GUITARSET_OPEN_STRINGS[string] if string is not None and 0 <= string < 6 else None
            notes_raw.append((float(d["time"]), float(d["duration"]), pitch, string, fret))
    notes_raw.sort(key=lambda x: (x[0], x[2]))
    notes = []
    for k, (t, dur, pitch, string, fret) in enumerate(notes_raw):
        s = to_beats(t)
        e = to_beats(t + dur)
        notes.append((k, round(s, 6), round(max(e - s, 0.0), 6), round(t, 6), pitch, None, string, fret))
    first = beats["data"][0]["value"] if beats and beats["data"] else {}
    key_ann = next((a for a in anns if a["namespace"] == "key_mode"), None)
    player, title, style, mode = guitarset_name(Path(path).stem)
    meta = {"n_instruments": 1, "n_tempos": 1, "n_time_signatures": 1,
            "time_signature": f"{first.get('num_beats', 4)}/{first.get('beat_units', 4)}",
            "key": key_ann["data"][0]["value"] if key_ann and key_ann["data"] else None,
            "player": player, "style": style, "mode": mode}
    tracks = [{"idx": 0, "name": f"guitar ({mode})", "program": 25, "role": "guitar", "is_drum": 0,
               "n_notes": len(notes), "notes": notes}]
    return {"tracks": tracks, "tempo": tempo, "resolution": None, "format": "jams", "meta": meta}


def _run(job: tuple) -> tuple[tuple, Parsed | None, str | None]:
    kind, path, source, original_id, keep_roles, min_notes, max_notes = job
    try:
        if kind == "midi":
            return job, parse_midi(Path(path), source, original_id, keep_roles, min_notes, max_notes), None
        if kind == "guitarset":
            return job, parse_guitarset(Path(path), source, original_id), None
        return job, parse_dadagp(Path(path), source, original_id, min_notes, max_notes), None
    except Exception as e:  # corrupt files happen
        return job, None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #


def _midi_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in (".mid", ".midi"))


def lakh_artist_title(path: Path, root: Path) -> tuple[str | None, str]:
    """Lakh names files ``Artist/Title.mid`` with ``.1``, ``.2`` suffixes for further
    transcriptions of the same song; returns (artist, title) without the suffix."""
    title = re.sub(r"\.\d+$", "", path.stem)
    artist = path.parent.name if path.parent != root else None
    return artist, title


def dadagp_artist_title(path: Path) -> tuple[str | None, str]:
    """DadaGP token files are named ``Artist - Title.<ext>.txt`` in the common layout."""
    stem = path.name.split(".")[0]
    artist, sep, title = stem.partition(" - ")
    return (artist.strip(), title.strip()) if sep else (None, stem)


def jobs_for(pop909_dir: Path | None, guitar_dir: Path | None, dadagp_dir: Path | None,
             min_notes: int, max_notes: int, limit: int | None = None,
             guitarset_dir: Path | None = None, lakh_full_dir: Path | None = None) -> list[tuple]:
    """(kind, path, source, original_id, keep_roles, min, max, artist, title[, genre])."""
    jobs: list[tuple] = []
    if guitarset_dir:
        files = sorted(Path(guitarset_dir).rglob("*.jams"))
        for f in files[:limit] if limit else files:
            player, title, style, mode = guitarset_name(f.stem)
            jobs.append(("guitarset", str(f), "guitarset", f.stem, ("guitar",), min_notes, max_notes,
                         "GuitarSet", title, GUITARSET_STYLE_GENRE.get(style)))
    if pop909_dir:
        files = [f for f in _midi_files(pop909_dir) if "versions" not in f.parts]
        for f in files[:limit] if limit else files:
            oid = f.parent.name if f.parent != pop909_dir else f.stem
            jobs.append(("midi", str(f), "pop909", oid, STORED_ROLES, min_notes, max_notes, None, None))
    if guitar_dir:
        files = _midi_files(guitar_dir)
        for f in files[:limit] if limit else files:
            artist, title = lakh_artist_title(f, guitar_dir)
            jobs.append(("midi", str(f), "lakh_guitar", str(f.relative_to(guitar_dir)), STORED_ROLES,
                         min_notes, max_notes, artist, title))
    if lakh_full_dir:  # no metadata: one song per file, guitar notes only (volume, not context)
        files = _midi_files(lakh_full_dir)
        for f in files[:limit] if limit else files:
            jobs.append(("midi", str(f), "lakh_full", str(f.relative_to(lakh_full_dir)), ("guitar",),
                         min_notes, max_notes, None, None))
    if dadagp_dir:
        files = sorted(dadagp_dir.rglob("*.txt"))
        for f in files[:limit] if limit else files:
            artist, title = dadagp_artist_title(f)
            jobs.append(("dadagp", str(f), "dadagp", f.name, ("guitar",), min_notes, max_notes, artist, title))
    return jobs


def ingest(con: sqlite3.Connection, jobs: list[tuple], workers: int = 1, min_notes: int = 5,
           max_notes: int = 16, log=print) -> dict[str, int]:
    """Parse files in worker processes and write songs, renditions, tracks, notes and phrases."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(it, **kw):
            return it
    counts: dict[str, int] = defaultdict(int)
    new_renditions: list[int] = []
    parse_jobs = [j[:7] for j in jobs]
    names = {j[1]: j[7:] for j in jobs}  # path -> (artist, title[, genre])
    done = set(r["path"] for r in con.execute("SELECT path FROM rendition"))
    todo = [j for j in parse_jobs if j[1] not in done]
    log(f"{len(jobs)} files, {len(todo)} not yet ingested")
    # the same file under another path (clean_midi inside the full Lakh set) is one rendition
    known_hashes = set(r["content_hash"] for r in con.execute("SELECT content_hash FROM rendition") if r["content_hash"])
    if known_hashes and todo:
        kept = []
        for j in todo:
            try:
                h = file_hash(Path(j[1]))
            except OSError:
                kept.append(j)
                continue
            if h in known_hashes:
                counts["duplicate"] += 1
            else:
                known_hashes.add(h)
                kept.append(j)
        if counts["duplicate"]:
            log(f"{counts['duplicate']} files skipped: same content as a rendition already stored")
        todo = kept

    def results():
        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                yield from pool.map(_run, todo, chunksize=4)
        else:
            for j in todo:
                yield _run(j)

    for job, parsed, err in tqdm(results(), total=len(todo), desc="ingest"):
        kind, path, source, original_id, _, _, _ = job
        artist, title, *rest = names[path]
        genre = rest[0] if rest else None
        if parsed is None:
            counts["failed"] += 1
            log(f"  skip {path}: {err}")
            continue
        sid = source_id(con, source)
        wid = get_or_create_song(con, title, artist, genre)
        cur = con.execute(
            "INSERT INTO rendition(song_id, source_id, original_id, path, content_hash, format, tempo, resolution, meta_json)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (wid, sid, original_id, path, file_hash(Path(path)), parsed["format"], parsed["tempo"],
             parsed["resolution"], json.dumps(parsed["meta"])))
        rid = int(cur.lastrowid)
        track_ids: dict[int, int] = {}
        for t in parsed["tracks"]:
            cur = con.execute(
                "INSERT INTO track(rendition_id, idx, name, program, role, is_drum, n_notes, notes_stored) VALUES (?,?,?,?,?,?,?,?)",
                (rid, t["idx"], t["name"], t["program"], t["role"], t["is_drum"], t["n_notes"], int(bool(t["notes"]))))
            tid = int(cur.lastrowid)
            track_ids[t["idx"]] = tid
            if t["notes"]:
                con.executemany(
                    "INSERT INTO note(track_id, idx, onset_beats, duration_beats, onset_s, pitch, velocity, string, fret)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    [(tid, k, s, d, os_, p, v, st, fr) for k, s, d, os_, p, v, st, fr in t["notes"]])
                counts["notes"] += len(t["notes"])
            counts["tracks"] += 1
        new_renditions.append(rid)
        counts["renditions"] += 1
        if counts["renditions"] % 200 == 0:
            con.commit()
    con.commit()
    if new_renditions:
        counts["phrases"] = cut_phrases(con, new_renditions, min_notes, max_notes, log=log)
    return dict(counts)


def add_notes(con: sqlite3.Connection, roles: tuple[str, ...], workers: int = 1, log=print) -> dict[str, int]:
    """Store the notes of tracks with one of ``roles`` that were ingested without them
    (the files are re-read; nothing else changes). Also refreshes the rendition's
    ``meta_json`` with the time signature. Returns counts."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(it, **kw):
            return it
    marks = ",".join("?" * len(roles))
    rows = con.execute(
        f"""SELECT DISTINCT r.id, r.path, s.name AS source, r.original_id FROM rendition r
            JOIN source s ON s.id = r.source_id JOIN track t ON t.rendition_id = r.id
            WHERE r.format = 'mid' AND t.role IN ({marks}) AND t.notes_stored = 0 AND t.n_notes > 0
            ORDER BY r.id""", roles).fetchall()
    jobs = [("midi", r["path"], r["source"], r["original_id"], tuple(roles), 5, 16) for r in rows
            if r["path"] and os.path.exists(r["path"])]
    rid_of = {r["path"]: int(r["id"]) for r in rows}
    log(f"{len(rows)} renditions have {'/'.join(roles)} tracks without notes; {len(jobs)} files present")
    counts: dict[str, int] = defaultdict(int)

    def results():
        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                yield from pool.map(_run, jobs, chunksize=4)
        else:
            for j in jobs:
                yield _run(j)

    for job, parsed, err in tqdm(results(), total=len(jobs), desc="notes"):
        rid = rid_of[job[1]]
        if parsed is None:
            counts["failed"] += 1
            log(f"  skip {job[1]}: {err}")
            continue
        tracks = {int(t["idx"]): int(t["id"]) for t in
                  con.execute("SELECT id, idx FROM track WHERE rendition_id=? AND notes_stored=0", (rid,))}
        for t in parsed["tracks"]:
            tid = tracks.get(t["idx"])
            if tid is None or not t["notes"]:
                continue
            con.executemany(
                "INSERT INTO note(track_id, idx, onset_beats, duration_beats, onset_s, pitch, velocity, string, fret)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                [(tid, k, s, d, os_, p, v, st, fr) for k, s, d, os_, p, v, st, fr in t["notes"]])
            con.execute("UPDATE track SET notes_stored=1 WHERE id=?", (tid,))
            counts["notes"] += len(t["notes"])
            counts["tracks"] += 1
        old = con.execute("SELECT meta_json FROM rendition WHERE id=?", (rid,)).fetchone()["meta_json"]
        meta = {**(json.loads(old) if old else {}), **parsed["meta"]}
        con.execute("UPDATE rendition SET meta_json=? WHERE id=?", (json.dumps(meta), rid))
        counts["renditions"] += 1
        if counts["renditions"] % 200 == 0:
            con.commit()
    con.commit()
    return dict(counts)


# --------------------------------------------------------------------------- #
# phrases from the note rows
# --------------------------------------------------------------------------- #

CUT_EXTRACTOR = "cut_phrases"
CUT_VERSION = "1"


def _onsets_with_ids(rows: list[sqlite3.Row], tol_beats: float = 0.05) -> tuple[list, list[int]]:
    """Group a guitar track's note rows into onsets ``(start, end, pitches)``
    plus, per onset, the id of its lowest note (the riff note)."""
    onsets: list[list] = []  # [start, end, pitches, lowest id, lowest pitch]
    for r in sorted(rows, key=lambda r: (r["onset_beats"], r["pitch"])):
        start, end, pitch = float(r["onset_beats"]), float(r["onset_beats"] + r["duration_beats"]), int(r["pitch"])
        if onsets and start - onsets[-1][0] <= tol_beats:
            o = onsets[-1]
            if pitch < o[4]:
                o[1], o[3], o[4] = end, int(r["id"]), pitch
            o[2].append(pitch)
        else:
            onsets.append([start, end, [pitch], int(r["id"]), pitch])
    return [(o[0], o[1], o[2]) for o in onsets], [o[3] for o in onsets]


def cut_phrases(con: sqlite3.Connection, rendition_ids: Iterable[int] | None = None, min_notes: int = 5,
                max_notes: int = 16, rest_split_beats: float = 2.0, log=print) -> int:
    """Cut training phrases from the stored note rows and link them to their notes.

    Melody tracks give the melody line (:func:`melody_line_phrases`); guitar
    tracks give the riff line (:func:`riff_line_phrases_indexed`: lowest note
    per onset, strummed chords end a phrase). Exact repeats within a rendition
    are kept once. Existing phrases of the renditions are replaced. Returns the
    number of phrases written.
    """
    ext = extractor_id(con, CUT_EXTRACTOR, CUT_VERSION,
                       {"min_notes": min_notes, "max_notes": max_notes, "rest_split_beats": rest_split_beats,
                        "max_chord": MAX_RIFF_CHORD})
    if rendition_ids is None:
        rids = [int(r["id"]) for r in con.execute("SELECT id FROM rendition ORDER BY id")]
    else:
        rids = list(rendition_ids)
    total = 0
    for n_done, rid in enumerate(rids, 1):
        con.execute("DELETE FROM phrase WHERE rendition_id=?", (rid,))
        src = con.execute("SELECT s.name FROM rendition r JOIN source s ON s.id=r.source_id WHERE r.id=?", (rid,)).fetchone()["name"]
        seen: set[tuple] = set()
        idx = 0
        # only melody and guitar tracks are phrase sources; bass and drum notes are context (ml/features.py)
        for t in con.execute("SELECT id, role FROM track WHERE rendition_id=? AND notes_stored=1 "
                             "AND role IN ('melody', 'guitar') ORDER BY idx", (rid,)):
            rows = con.execute("SELECT id, onset_beats, duration_beats, pitch FROM note WHERE track_id=? ORDER BY idx",
                               (t["id"],)).fetchall()
            if not rows:
                continue
            if t["role"] == "melody":
                starts = [float(r["onset_beats"]) for r in rows]
                ends = [float(r["onset_beats"] + r["duration_beats"]) for r in rows]
                found = [(p, [int(rows[i]["id"]) for i in ix]) for p, ix in melody_line_phrases(
                    starts, ends, [int(r["pitch"]) for r in rows], str(rid), src, min_notes, max_notes, rest_split_beats)]
            else:
                onsets, lowest = _onsets_with_ids(rows)
                found = [(p, [lowest[i] for i in ix]) for p, ix in riff_line_phrases_indexed(
                    onsets, str(rid), src, min_notes, max_notes, rest_split_beats)]
            for p, note_ids in found:
                key = (tuple(p.pitches), tuple(p.onsets_beats))
                if key in seen:
                    continue
                seen.add(key)
                start = con.execute("SELECT onset_beats FROM note WHERE id=?", (note_ids[0],)).fetchone()["onset_beats"]
                con.execute(
                    "INSERT INTO phrase(rendition_id, track_id, extractor_id, idx, start_beats, n_notes, pitches_json, onsets_json, durations_json, note_ids_json)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (rid, t["id"], ext, idx, float(start), len(p.pitches), json.dumps(p.pitches),
                     json.dumps(p.onsets_beats), json.dumps(p.durations_beats), json.dumps(note_ids)))
                idx += 1
                total += 1
        if n_done % 500 == 0:
            con.commit()
            log(f"  cut {n_done}/{len(rids)} renditions, {total} phrases")
    con.commit()
    return total


# --------------------------------------------------------------------------- #
# splits, export, stats
# --------------------------------------------------------------------------- #


def assign_splits(con: sqlite3.Connection, val_frac: float = 0.1, test_frac: float = 0.1,
                  seed: int = 0) -> dict[str, int]:
    """Song-level split: a seeded random draw over song ids.

    Songs already in the table keep their split, so re-running after a new
    ingest only draws the new songs; evaluation sets stay stable.
    """
    existing = {int(r["song_id"]): r["name"] for r in con.execute("SELECT song_id, name FROM split")}
    ids = [int(r["id"]) for r in con.execute(
        "SELECT id FROM song WHERE id IN (SELECT DISTINCT song_id FROM rendition) ORDER BY id")]
    new = [i for i in ids if i not in existing]
    rng = np.random.default_rng([seed, len(existing)])
    order = [new[i] for i in rng.permutation(len(new))]
    n_test, n_val = int(round(len(order) * test_frac)), int(round(len(order) * val_frac))
    drawn = {wid: "test" for wid in order[:n_test]}
    drawn.update({wid: "val" for wid in order[n_test:n_test + n_val]})
    drawn.update({wid: "train" for wid in order[n_test + n_val:]})
    counts: dict[str, int] = defaultdict(int)
    for wid in ids:
        name = existing.get(wid) or drawn[wid]
        con.execute("INSERT OR REPLACE INTO split(song_id, name, seed) VALUES (?,?,?)", (wid, name, seed))
        counts[name] += 1
    con.commit()
    return dict(counts)


def load_db_phrases(con: sqlite3.Connection, sources: Iterable[str] | None = None) -> list[Phrase]:
    q = ("SELECT r.song_id, r.id AS rendition_id, p.id AS phrase_id, s.name AS source, w.genre AS genre, "
         "p.pitches_json, p.onsets_json, p.durations_json "
         "FROM phrase p JOIN rendition r ON r.id = p.rendition_id "
         "JOIN source s ON s.id = r.source_id JOIN song w ON w.id = r.song_id")
    args: tuple = ()
    if sources:
        names = list(sources)
        q += " WHERE s.name IN (%s)" % ",".join("?" * len(names))
        args = tuple(names)
    q += " ORDER BY r.id, p.idx"
    return [Phrase(str(r["song_id"]), json.loads(r["pitches_json"]), json.loads(r["onsets_json"]),
                   json.loads(r["durations_json"]), r["source"], rendition_id=int(r["rendition_id"]),
                   phrase_id=int(r["phrase_id"]), genre=r["genre"])
            for r in con.execute(q, args)]


def cap_per_song(phrases: list[Phrase], max_per_song: int | None, seed: int = 0) -> list[Phrase]:
    """Keep at most ``max_per_song`` phrases per song (a seeded random subset, in
    time order), so a song with many renditions or guitar tracks cannot
    outweigh one with a single melody track."""
    if not max_per_song:
        return phrases
    by_song: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(phrases):
        by_song[p.song_id].append(i)
    keep: set[int] = set()
    for sid, idx in by_song.items():
        rng = np.random.default_rng([seed, int(sid) if sid.isdigit() else abs(hash(sid)) % (2 ** 31)])
        keep.update(idx if len(idx) <= max_per_song else sorted(rng.choice(idx, max_per_song, replace=False).tolist()))
    return [p for i, p in enumerate(phrases) if i in keep]


def export(con: sqlite3.Connection, out: Path, sources: Iterable[str] | None = None,
           max_per_song: int | None = None, cap_exempt: Iterable[str] = ("pop909",),
           seed: int = 0) -> tuple[int, dict[str, int]]:
    """Write phrases as JSONL (``song_id`` = song id) plus ``splits.json`` next to it.

    ``max_per_song`` caps phrases per song for every source except those in
    ``cap_exempt`` (melody sources have one track per song and need no cap).
    """
    exempt = set(cap_exempt or ())
    phrases = load_db_phrases(con, sources)
    capped = cap_per_song([p for p in phrases if p.source not in exempt], max_per_song, seed)
    phrases = [p for p in phrases if p.source in exempt] + capped
    save_phrases(phrases, out)
    splits = defaultdict(list)
    for r in con.execute("SELECT song_id, name FROM split"):
        splits[r["name"]].append(str(r["song_id"]))
    present = {p.song_id for p in phrases}
    ids = {name: sorted((k for k in v if k in present), key=int) for name, v in splits.items()}
    Path(out).with_name("splits.json").write_text(json.dumps(ids, indent=1))
    return len(phrases), {k: len(v) for k, v in ids.items()}


def stats(con: sqlite3.Connection) -> str:
    lines = []
    for r in con.execute("""SELECT s.name, COUNT(DISTINCT r.song_id) songs, COUNT(r.id) renditions,
                                   (SELECT COUNT(*) FROM track t JOIN rendition r2 ON r2.id=t.rendition_id WHERE r2.source_id=s.id) tracks,
                                   (SELECT COUNT(*) FROM note n JOIN track t ON t.id=n.track_id JOIN rendition r3 ON r3.id=t.rendition_id WHERE r3.source_id=s.id) notes,
                                   (SELECT COUNT(*) FROM phrase p JOIN rendition r4 ON r4.id=p.rendition_id WHERE r4.source_id=s.id) phrases
                            FROM source s JOIN rendition r ON r.source_id = s.id GROUP BY s.id"""):
        lines.append(f"{r['name']:12s} songs {r['songs']:6d}  renditions {r['renditions']:6d}  tracks {r['tracks']:7d}  "
                     f"notes {r['notes']:9d}  phrases {r['phrases']:7d}")
    multi = con.execute("SELECT COUNT(*) c FROM (SELECT song_id FROM rendition GROUP BY song_id HAVING COUNT(*) > 1)").fetchone()["c"]
    lines.append(f"songs with more than one rendition: {multi}")
    sp = {r["name"]: r["c"] for r in con.execute("SELECT name, COUNT(*) c FROM split GROUP BY name")}
    lines.append(f"splits (songs): {sp}")
    roles = {r["role"]: r["c"] for r in con.execute("SELECT role, COUNT(*) c FROM track GROUP BY role")}
    lines.append(f"track roles: {roles}")
    ex = [f"{r['name']} v{r['version']}" for r in con.execute("SELECT name, version FROM extractor")]
    lines.append(f"extractors: {ex}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("ingest")
    p.add_argument("--pop909-dir", type=Path)
    p.add_argument("--guitar-midi-dir", type=Path)
    p.add_argument("--dadagp-dir", type=Path)
    p.add_argument("--guitarset-dir", type=Path, help="folder of GuitarSet .jams annotations")
    p.add_argument("--lakh-full-dir", type=Path, help="Lakh MIDI full set (lmd_full): guitar notes only, no metadata")
    p.add_argument("--min-notes", type=int, default=5)
    p.add_argument("--max-notes", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="max files per source (debug)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p = sub.add_parser("notes", help="store the notes of bass/drum tracks ingested without them (re-reads the files)")
    p.add_argument("--roles", default="bass,drums")
    p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("cut", help="cut phrases from the stored notes (ingest does this for new files)")
    p.add_argument("--recut", action="store_true", help="replace the phrases of every rendition")
    p.add_argument("--min-notes", type=int, default=5)
    p.add_argument("--max-notes", type=int, default=16)
    p = sub.add_parser("split")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("export")
    p.add_argument("--out", type=Path, default=Path("data/processed/phrases.jsonl"))
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--max-per-song", type=int, default=None)
    p.add_argument("--cap-exempt", nargs="*", default=["pop909"], help="sources the cap does not apply to")
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("stats")
    p = sub.add_parser("dedupe", help="merge songs the matching rules consider the same")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--show", type=int, default=20, help="print this many merges")
    p = sub.add_parser("merge", help="fold one song id into another by hand")
    p.add_argument("keep", type=int)
    p.add_argument("drop", type=int)
    p = sub.add_parser("sql")
    p.add_argument("query")
    args = ap.parse_args(argv)

    con = connect(args.db)
    init(con)
    if args.cmd == "init":
        print(f"initialised {args.db}")
    elif args.cmd == "ingest":
        jobs = jobs_for(args.pop909_dir, args.guitar_midi_dir, args.dadagp_dir, args.min_notes, args.max_notes, args.limit,
                        guitarset_dir=args.guitarset_dir, lakh_full_dir=args.lakh_full_dir)
        print(ingest(con, jobs, args.workers, args.min_notes, args.max_notes))
    elif args.cmd == "notes":
        print(add_notes(con, tuple(r.strip() for r in args.roles.split(",") if r.strip()), args.workers))
    elif args.cmd == "cut":
        if args.recut:
            n = cut_phrases(con, None, args.min_notes, args.max_notes)
        else:
            missing = [int(r["id"]) for r in con.execute(
                "SELECT id FROM rendition WHERE id NOT IN (SELECT DISTINCT rendition_id FROM phrase)")]
            n = cut_phrases(con, missing, args.min_notes, args.max_notes)
        print(f"{n} phrases written")
    elif args.cmd == "split":
        print(assign_splits(con, args.val_frac, args.test_frac, args.seed))
    elif args.cmd == "export":
        n, sp = export(con, args.out, args.sources, args.max_per_song, args.cap_exempt, args.seed)
        print(f"wrote {n} phrases to {args.out}; splits {sp}")
    elif args.cmd == "stats":
        print(stats(con))
    elif args.cmd == "dedupe":
        merges = dedupe_songs(con, args.dry_run)
        names = {r["id"]: f"{r['artist']} / {r['title']}" for r in con.execute("SELECT id, artist, title FROM song")} if args.dry_run else {}
        for keep, drop, reason in merges[:args.show]:
            print(f"  {drop} -> {keep}  ({reason})" + (f": {names.get(drop)} => {names.get(keep)}" if names else ""))
    elif args.cmd == "merge":
        merge_songs(con, args.keep, args.drop)
        con.commit()
        print(f"merged song {args.drop} into {args.keep}")
    elif args.cmd == "sql":
        for row in con.execute(args.query):
            print(dict(row))


if __name__ == "__main__":
    main()
