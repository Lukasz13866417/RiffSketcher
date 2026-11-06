"""The song database: everything we know about every song, in one SQLite file.

Design
------
``work``       the composition (title, artist, genre). One row per song however
               many files we hold of it. Splits attach here, so a song is never
               in training through one file and in test through another.
``rendition``  one file of one source (a Lakh MIDI, a DadaGP tab, an audio
               recording ...) pointing at its work; carries the file's own
               facts (path, hash, tempo, resolution).
``track``      instrument tracks of a rendition, with a role (melody, guitar,
               bass, drums, other).
``note``       the per-moment table: one row per note event of the tracks we
               keep, with pitch, timing, velocity, string/fret and typed
               articulation columns (palm mute, bend, slide, hammer-on,
               vibrato, dead, tie) that tabs and audio extraction fill in.
``extractor``  name + version + params of anything that derives data; every
               derived row points at one so it can be re-run and replaced.
``phrase``     a training phrase cut from a track by an extractor, stored with
               its pitches/onsets/durations and the note ids it covers.
``split``      work -> train/val/test.
``*_feature``  key/value rows per work, rendition or note from any extractor,
               and ``frame_feature`` for time-indexed signals not aligned to
               notes (loudness curves and the like).

``work.key`` is the legacy ``song_id`` used by the JSONL pipeline (POP909
``"003"``, Lakh ``"Artist/Title"``), so existing ``splits.json`` files and
trained models carry over unchanged. ``export`` reproduces the JSONL that
``ml.prepare_dataset`` wrote, so the rest of ``ml/`` keeps working.

CLI::

    uv run python -m ml.db init
    uv run python -m ml.db ingest --pop909-dir data/raw/pop909/POP909 --guitar-midi-dir data/raw/lakh_clean/clean_midi
    uv run python -m ml.db split --base-splits data/processed/splits.json
    uv run python -m ml.db export --out data/processed/mixed.jsonl --max-per-song 40
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

from ml.datasets import (Phrase, _is_guitar_track, _monophonic_notes, _onsets_from_dadagp_events,
                         _onsets_from_notes, _pick_melody_instrument, dadagp_track_events,
                         phrases_from_dadagp, phrases_from_guitar_midi, phrases_from_midi,
                         save_phrases, split_by_song, split_ids)

DEFAULT_DB = Path("data/riffs.db")

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS source (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, licence TEXT, notes TEXT);

CREATE TABLE IF NOT EXISTS work (
    id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL,
    title TEXT, artist TEXT, title_norm TEXT, artist_norm TEXT, genre TEXT, notes TEXT);
CREATE INDEX IF NOT EXISTS work_norm ON work(artist_norm, title_norm);

CREATE TABLE IF NOT EXISTS rendition (
    id INTEGER PRIMARY KEY, work_id INTEGER NOT NULL REFERENCES work(id),
    source_id INTEGER NOT NULL REFERENCES source(id), original_id TEXT NOT NULL,
    path TEXT, content_hash TEXT, format TEXT, tempo REAL, resolution INTEGER, meta_json TEXT,
    UNIQUE(source_id, original_id));
CREATE INDEX IF NOT EXISTS rendition_work ON rendition(work_id);

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
    work_id INTEGER PRIMARY KEY REFERENCES work(id) ON DELETE CASCADE, name TEXT NOT NULL, seed INTEGER);

CREATE TABLE IF NOT EXISTS work_feature (
    work_id INTEGER NOT NULL REFERENCES work(id) ON DELETE CASCADE,
    extractor_id INTEGER NOT NULL REFERENCES extractor(id), key TEXT NOT NULL,
    value_num REAL, value_text TEXT, PRIMARY KEY (work_id, extractor_id, key));
CREATE TABLE IF NOT EXISTS rendition_feature (
    rendition_id INTEGER NOT NULL REFERENCES rendition(id) ON DELETE CASCADE,
    extractor_id INTEGER NOT NULL REFERENCES extractor(id), key TEXT NOT NULL,
    value_num REAL, value_text TEXT, PRIMARY KEY (rendition_id, extractor_id, key));
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
}

_NORM_RE = re.compile(r"[^a-z0-9]+")


def norm_text(s: str | None) -> str:
    return _NORM_RE.sub(" ", (s or "").lower()).strip()


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


def get_or_create_work(con: sqlite3.Connection, key: str, title: str | None = None,
                       artist: str | None = None, genre: str | None = None) -> int:
    """Find a work by its key, else by normalised artist+title, else create it."""
    row = con.execute("SELECT id FROM work WHERE key = ?", (key,)).fetchone()
    if row is not None:
        return int(row["id"])
    tn, an = norm_text(title), norm_text(artist)
    if tn and an:
        row = con.execute("SELECT id FROM work WHERE title_norm=? AND artist_norm=?", (tn, an)).fetchone()
        if row is not None:
            return int(row["id"])
    cur = con.execute("INSERT INTO work(key, title, artist, title_norm, artist_norm, genre) VALUES (?,?,?,?,?,?)",
                      (key, title, artist, tn or None, an or None, genre))
    return int(cur.lastrowid)


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


def parse_midi(path: Path, source: str, song_key: str, keep_roles: tuple[str, ...], min_notes: int,
               max_notes: int) -> Parsed:
    """Tracks (+ notes for ``keep_roles``) and the legacy phrases of one MIDI file."""
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
            for k, n in enumerate(src_notes):
                s, e = to_beats(n.start), to_beats(n.end)
                notes.append((k, round(s, 4), round(max(e - s, 0.0), 4), round(float(n.start), 4),
                              int(n.pitch), int(n.velocity)))
        tracks.append({"idx": idx, "name": ins.name, "program": int(ins.program), "role": role,
                       "is_drum": int(ins.is_drum), "n_notes": len(ins.notes), "notes": notes})
    if source == "pop909":
        phrases = phrases_from_midi(path, min_notes, max_notes, song_id=song_key, source=source)
    else:
        phrases = phrases_from_guitar_midi(path, min_notes, max_notes, song_id=song_key, source=source)
    tempos = pm.get_tempo_changes()[1]
    return {"tracks": tracks, "phrases": [p.to_dict() for p in phrases],
            "tempo": float(tempos[0]) if len(tempos) else None, "resolution": int(pm.resolution),
            "format": path.suffix.lower().lstrip("."), "meta": {"n_instruments": len(pm.instruments)}}


def parse_dadagp(path: Path, source: str, song_key: str, min_notes: int, max_notes: int) -> Parsed:
    """Guitar tracks of a DadaGP token file; the riff line (lowest note per onset) is stored."""
    with path.open(encoding="utf-8", errors="replace") as f:
        tracks_ev = dadagp_track_events(f)
    tracks = []
    for idx, (name, events) in enumerate(tracks_ev.items()):
        notes = []
        for k, (s, e, pitches) in enumerate(_onsets_from_dadagp_events(events)):
            if pitches:
                notes.append((k, round(s, 4), round(max(e - s, 0.0), 4), None, int(min(pitches)), None))
        tracks.append({"idx": idx, "name": name, "program": None, "role": "guitar", "is_drum": 0,
                       "n_notes": len(notes), "notes": notes})
    phrases = phrases_from_dadagp(path, min_notes, max_notes, song_id=song_key, source=source)
    return {"tracks": tracks, "phrases": [p.to_dict() for p in phrases], "tempo": None, "resolution": 960,
            "format": "dadagp", "meta": {}}


def _run(job: tuple) -> tuple[tuple, Parsed | None, str | None]:
    kind, path, source, song_key, keep_roles, min_notes, max_notes = job
    try:
        if kind == "midi":
            return job, parse_midi(Path(path), source, song_key, keep_roles, min_notes, max_notes), None
        return job, parse_dadagp(Path(path), source, song_key, min_notes, max_notes), None
    except Exception as e:  # corrupt files happen
        return job, None, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #


def _midi_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in (".mid", ".midi"))


def lakh_song_key(path: Path, root: Path) -> tuple[str, str, str]:
    """(work key 'Artist/Title', artist, title) with Lakh's '.1' version suffix removed."""
    title = re.sub(r"\.\d+$", "", path.stem)
    artist = path.parent.name if path.parent != root else ""
    return (f"{artist}/{title}" if artist else title), artist, title


def jobs_for(pop909_dir: Path | None, guitar_dir: Path | None, dadagp_dir: Path | None,
             min_notes: int, max_notes: int, limit: int | None = None) -> list[tuple]:
    """(kind, path, source, work key, keep_roles, min, max, artist, title, original_id)."""
    jobs: list[tuple] = []
    if pop909_dir:
        files = [f for f in _midi_files(pop909_dir) if "versions" not in f.parts]
        for f in files[:limit] if limit else files:
            key = f.parent.name if f.parent != pop909_dir else f.stem
            jobs.append(("midi", str(f), "pop909", key, ("melody",), min_notes, max_notes, None, None, key))
    if guitar_dir:
        files = _midi_files(guitar_dir)
        for f in files[:limit] if limit else files:
            key, artist, title = lakh_song_key(f, guitar_dir)
            jobs.append(("midi", str(f), "lakh_guitar", key, ("guitar",), min_notes, max_notes, artist, title,
                         str(f.relative_to(guitar_dir))))
    if dadagp_dir:
        files = sorted(dadagp_dir.rglob("*.txt"))
        for f in files[:limit] if limit else files:
            key = f.name.split(".")[0]
            jobs.append(("dadagp", str(f), "dadagp", key, ("guitar",), min_notes, max_notes, None, None, f.name))
    return jobs


def ingest(con: sqlite3.Connection, jobs: list[tuple], workers: int = 1, min_notes: int = 5,
           max_notes: int = 16, log=print) -> dict[str, int]:
    """Parse files in worker processes and write works, renditions, tracks, notes and phrases."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(it, **kw):
            return it
    counts: dict[str, int] = defaultdict(int)
    ext = {
        "pop909": extractor_id(con, "phrases_from_midi", "1", {"min_notes": min_notes, "max_notes": max_notes}),
        "lakh_guitar": extractor_id(con, "phrases_from_guitar_midi", "1", {"min_notes": min_notes, "max_notes": max_notes}),
        "dadagp": extractor_id(con, "phrases_from_dadagp", "1", {"min_notes": min_notes, "max_notes": max_notes}),
    }
    work_jobs = [j[:7] for j in jobs]
    extra = {j[1]: j[7:] for j in jobs}  # path -> (artist, title, original_id)
    done = set(r["path"] for r in con.execute("SELECT path FROM rendition"))
    todo = [j for j in work_jobs if j[1] not in done]
    log(f"{len(jobs)} files, {len(todo)} not yet ingested")

    def results():
        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                yield from pool.map(_run, todo, chunksize=4)
        else:
            for j in todo:
                yield _run(j)

    for job, parsed, err in tqdm(results(), total=len(todo), desc="ingest"):
        kind, path, source, key, _, _, _ = job
        artist, title, original_id = extra[path]
        if parsed is None:
            counts["failed"] += 1
            log(f"  skip {path}: {err}")
            continue
        sid = source_id(con, source)
        wid = get_or_create_work(con, key, title, artist)
        cur = con.execute(
            "INSERT INTO rendition(work_id, source_id, original_id, path, content_hash, format, tempo, resolution, meta_json)"
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
                    "INSERT INTO note(track_id, idx, onset_beats, duration_beats, onset_s, pitch, velocity) VALUES (?,?,?,?,?,?,?)",
                    [(tid, k, s, d, os_, p, v) for k, s, d, os_, p, v in t["notes"]])
                counts["notes"] += len(t["notes"])
            counts["tracks"] += 1
        for i, p in enumerate(parsed["phrases"]):
            con.execute(
                "INSERT INTO phrase(rendition_id, track_id, extractor_id, idx, start_beats, n_notes, pitches_json, onsets_json, durations_json)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (rid, None, ext[source], i, None, len(p["pitches"]), json.dumps(p["pitches"]),
                 json.dumps(p["onsets_beats"]), json.dumps(p["durations_beats"])))
        counts["phrases"] += len(parsed["phrases"])
        counts["renditions"] += 1
        if counts["renditions"] % 200 == 0:
            con.commit()
    con.commit()
    return dict(counts)


# --------------------------------------------------------------------------- #
# splits, export, stats
# --------------------------------------------------------------------------- #


def assign_splits(con: sqlite3.Connection, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0,
                  base_ids: dict[str, list[str]] | None = None) -> dict[str, int]:
    """Song-level split over works. Works named in ``base_ids`` keep their split;
    works already in the table keep theirs; only new works are drawn."""
    fixed = {k: name for name, keys in (base_ids or {}).items() for k in keys}
    existing = {r["key"]: r["name"] for r in con.execute(
        "SELECT w.key, s.name FROM split s JOIN work w ON w.id = s.work_id")}
    rows = con.execute("SELECT id, key FROM work WHERE id IN (SELECT DISTINCT rendition.work_id FROM rendition)").fetchall()
    new = [r for r in rows if r["key"] not in fixed and r["key"] not in existing]
    # reuse split_by_song's logic on dummy phrases so the rule is identical to the JSONL pipeline
    dummies = [Phrase(r["key"], [60] * 5, [0, 1, 2, 3, 4], [1] * 5, "db") for r in new]
    drawn = {p.song_id: name for name, ps in split_by_song(dummies, val_frac, test_frac, seed).items() for p in ps}
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        name = existing.get(r["key"]) or fixed.get(r["key"]) or drawn.get(r["key"])
        if name is None:
            continue
        con.execute("INSERT OR REPLACE INTO split(work_id, name, seed) VALUES (?,?,?)", (r["id"], name, seed))
        counts[name] += 1
    con.commit()
    return dict(counts)


def load_db_phrases(con: sqlite3.Connection, sources: Iterable[str] | None = None) -> list[Phrase]:
    q = ("SELECT w.key AS song_id, s.name AS source, p.pitches_json, p.onsets_json, p.durations_json "
         "FROM phrase p JOIN rendition r ON r.id = p.rendition_id JOIN work w ON w.id = r.work_id "
         "JOIN source s ON s.id = r.source_id")
    args: tuple = ()
    if sources:
        names = list(sources)
        q += " WHERE s.name IN (%s)" % ",".join("?" * len(names))
        args = tuple(names)
    q += " ORDER BY r.id, p.idx"
    return [Phrase(r["song_id"], json.loads(r["pitches_json"]), json.loads(r["onsets_json"]),
                   json.loads(r["durations_json"]), r["source"]) for r in con.execute(q, args)]


def export(con: sqlite3.Connection, out: Path, sources: Iterable[str] | None = None,
           max_per_song: int | None = None, seed: int = 0) -> tuple[int, dict[str, int]]:
    """Write phrases as JSONL plus splits.json next to it, like ml.prepare_dataset did.

    ``max_per_song`` caps non-POP909 phrases per work with the same random rule.
    """
    from ml.prepare_dataset import cap_per_song

    phrases = load_db_phrases(con, sources)
    guitar = cap_per_song([p for p in phrases if p.source != "pop909"], max_per_song, seed)
    phrases = [p for p in phrases if p.source == "pop909"] + guitar
    save_phrases(phrases, out)
    splits = defaultdict(list)
    for r in con.execute("SELECT w.key, s.name FROM split s JOIN work w ON w.id = s.work_id"):
        splits[r["name"]].append(r["key"])
    keys = {p.song_id for p in phrases}
    ids = {name: sorted(k for k in v if k in keys) for name, v in splits.items()}
    Path(out).with_name("splits.json").write_text(json.dumps(ids, indent=1))
    return len(phrases), {k: len(v) for k, v in ids.items()}


def stats(con: sqlite3.Connection) -> str:
    lines = []
    for r in con.execute("""SELECT s.name, COUNT(DISTINCT r.work_id) works, COUNT(r.id) renditions,
                                   (SELECT COUNT(*) FROM track t JOIN rendition r2 ON r2.id=t.rendition_id WHERE r2.source_id=s.id) tracks,
                                   (SELECT COUNT(*) FROM note n JOIN track t ON t.id=n.track_id JOIN rendition r3 ON r3.id=t.rendition_id WHERE r3.source_id=s.id) notes,
                                   (SELECT COUNT(*) FROM phrase p JOIN rendition r4 ON r4.id=p.rendition_id WHERE r4.source_id=s.id) phrases
                            FROM source s JOIN rendition r ON r.source_id = s.id GROUP BY s.id"""):
        lines.append(f"{r['name']:12s} works {r['works']:6d}  renditions {r['renditions']:6d}  tracks {r['tracks']:7d}  "
                     f"notes {r['notes']:9d}  phrases {r['phrases']:7d}")
    multi = con.execute("SELECT COUNT(*) c FROM (SELECT work_id FROM rendition GROUP BY work_id HAVING COUNT(*) > 1)").fetchone()["c"]
    lines.append(f"works with more than one rendition: {multi}")
    sp = {r["name"]: r["c"] for r in con.execute("SELECT name, COUNT(*) c FROM split GROUP BY name")}
    lines.append(f"splits (works): {sp}")
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
    p.add_argument("--min-notes", type=int, default=5)
    p.add_argument("--max-notes", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="max files per source (debug)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p = sub.add_parser("split")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-splits", type=Path, default=None)
    p = sub.add_parser("export")
    p.add_argument("--out", type=Path, default=Path("data/processed/phrases.jsonl"))
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--max-per-song", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("stats")
    p = sub.add_parser("sql")
    p.add_argument("query")
    args = ap.parse_args(argv)

    con = connect(args.db)
    init(con)
    if args.cmd == "init":
        print(f"initialised {args.db}")
    elif args.cmd == "ingest":
        jobs = jobs_for(args.pop909_dir, args.guitar_midi_dir, args.dadagp_dir, args.min_notes, args.max_notes, args.limit)
        print(ingest(con, jobs, args.workers, args.min_notes, args.max_notes))
    elif args.cmd == "split":
        base = json.loads(args.base_splits.read_text()) if args.base_splits else None
        print(assign_splits(con, args.val_frac, args.test_frac, args.seed, base))
    elif args.cmd == "export":
        n, sp = export(con, args.out, args.sources, args.max_per_song, args.seed)
        print(f"wrote {n} phrases to {args.out}; splits {sp}")
    elif args.cmd == "stats":
        print(stats(con))
    elif args.cmd == "sql":
        for row in con.execute(args.query):
            print(dict(row))


if __name__ == "__main__":
    main()
