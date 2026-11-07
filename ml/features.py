"""Features computed from what the song database already stores.

Everything here reads note rows and writes the feature tables with extractor
provenance (``extractor`` rows name and version each procedure), so a feature
can be recomputed or replaced without touching notes or phrases.

Per rendition (``rendition_feature``):
  key / mode / tonic / key_confidence   Krumhansl-Schmuckler fit of the duration-weighted
                                         pitch-class histogram of all stored pitched notes
  time_signature / beats_per_bar         from the file's first time signature (4/4 if none)
  swing / n_offbeat                      median position inside the beat of melody and guitar
                                         notes that fall between the eighth and the dotted
                                         eighth: 0.5 is straight, 0.67 is a swung eighth
  doubled_tracks                         guitar tracks that copy an earlier guitar track

Per song (``song_feature``): key / mode / tonic, the rendition keys weighted by confidence.

Per phrase (``phrase_feature``): where it starts in the bar, which of its notes land with a
bass note, a kick or a snare (strings of 0/1, one per note), and the trope labels of the
catalogue.  Per-note rows are reserved for things that are really per note (an articulation
classifier); the metric and accompaniment context lives on the phrase because a phrase knows
its start and its notes, and 22M note rows times four keys would double the database.

    uv run python -m ml.features run              # everything not yet computed
    uv run python -m ml.features run --recompute  # replace
    uv run python -m ml.features stats
"""
from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from core.motifs import label_tropes
from ml.datasets import GENRES, genre_of_source

# Krumhansl-Kessler key profiles (major, minor), index = semitones above the tonic
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
HIT_TOL = 0.05  # beats: a bass/drum onset this close to a riff note counts as "together"
KICK, SNARE = {35, 36}, {38, 40}

EXTRACTORS = {
    "key": ("rendition_key", "1"),
    "metre": ("rendition_metre", "1"),
    "context": ("phrase_context", "1"),
    "tropes": ("phrase_tropes", "1"),
    "song_key": ("song_key", "1"),
}


def estimate_key(pitches: list[int], weights: list[float]) -> tuple[int, str, float, float]:
    """(tonic pitch class, "major"/"minor", correlation, margin over the runner-up)."""
    hist = np.zeros(12)
    for p, w in zip(pitches, weights):
        hist[int(p) % 12] += max(float(w), 1e-3)
    if hist.sum() <= 0:
        return 0, "major", 0.0, 0.0
    scores = []
    for tonic in range(12):
        for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
            r = np.corrcoef(np.roll(prof, tonic), hist)[0, 1]
            scores.append((float(np.nan_to_num(r)), tonic, mode))
    scores.sort(reverse=True)
    (best, tonic, mode), (second, _, _) = scores[0], scores[1]
    return tonic, mode, best, best - second


def beats_per_bar(time_signature: str | None) -> float:
    """Quarter-note beats in a bar: 4/4 -> 4, 3/4 -> 3, 6/8 -> 3, 2/2 -> 4."""
    if not time_signature or "/" not in time_signature:
        return 4.0
    num, den = time_signature.split("/", 1)
    try:
        return float(num) * 4.0 / float(den)
    except ValueError:
        return 4.0


def swing_ratio(onsets: list[float], min_notes: int = 20) -> tuple[float | None, int]:
    """Median within-beat position of off-beat notes (between 0.4 and 0.75 of a beat)."""
    frac = np.asarray(onsets, dtype=float) % 1.0
    off = frac[(frac >= 0.4) & (frac <= 0.75)]
    if len(off) < min_notes:
        return None, int(len(off))
    return float(np.median(off)), int(len(off))


def doubled_tracks(track_notes: dict[int, list[tuple[float, int]]], threshold: float = 0.8) -> list[int]:
    """Ids of tracks whose (onset, pitch) set is mostly contained in an earlier track's."""
    sets = {tid: {(round(o, 2), p) for o, p in notes} for tid, notes in track_notes.items() if notes}
    ids = sorted(sets)
    out = []
    for j, tj in enumerate(ids):
        for ti in ids[:j]:
            common = len(sets[ti] & sets[tj])
            if common / max(1, min(len(sets[ti]), len(sets[tj]))) >= threshold:
                out.append(tj)
                break
    return out


def hits(onsets: list[float], reference: list[float], tol: float = HIT_TOL) -> str:
    """'1' per onset that has a reference onset within ``tol`` beats, else '0'."""
    ref = sorted(reference)
    out = []
    for o in onsets:
        i = bisect.bisect_left(ref, o - tol)
        out.append("1" if i < len(ref) and ref[i] <= o + tol else "0")
    return "".join(out)


def _put(con: sqlite3.Connection, table: str, id_col: str, rows: list[tuple]) -> None:
    con.executemany(f"INSERT OR REPLACE INTO {table}({id_col}, extractor_id, key, value_num, value_text) VALUES (?,?,?,?,?)", rows)


def _rendition_ids(con: sqlite3.Connection, table: str, ext: int, recompute: bool) -> list[int]:
    if recompute:
        con.execute(f"DELETE FROM {table} WHERE extractor_id=?", (ext,))
        return [int(r["id"]) for r in con.execute("SELECT id FROM rendition ORDER BY id")]
    col = "rendition_id" if table == "rendition_feature" else "phrase_id"
    if table == "phrase_feature":
        return [int(r["id"]) for r in con.execute(
            "SELECT DISTINCT r.id FROM rendition r JOIN phrase p ON p.rendition_id=r.id "
            f"WHERE NOT EXISTS (SELECT 1 FROM phrase_feature f WHERE f.phrase_id=p.id AND f.extractor_id=?) ORDER BY r.id", (ext,))]
    return [int(r["id"]) for r in con.execute(
        f"SELECT id FROM rendition r WHERE NOT EXISTS (SELECT 1 FROM {table} f WHERE f.{col}=r.id AND f.extractor_id=?) ORDER BY id", (ext,))]


def _load_notes(con: sqlite3.Connection, rid: int) -> dict[int, dict[str, Any]]:
    """track id -> {role, onsets, pitches, durations} for the rendition's stored tracks."""
    tracks: dict[int, dict[str, Any]] = {}
    for t in con.execute("SELECT id, role FROM track WHERE rendition_id=? AND notes_stored=1", (rid,)):
        rows = con.execute("SELECT onset_beats, duration_beats, pitch FROM note WHERE track_id=? ORDER BY idx", (t["id"],)).fetchall()
        tracks[int(t["id"])] = {"role": t["role"], "onsets": [float(r[0]) for r in rows],
                                "durations": [float(r[1]) for r in rows], "pitches": [int(r[2]) for r in rows]}
    return tracks


def rendition_features(con: sqlite3.Connection, which: set[str], recompute: bool = False, log=print) -> int:
    """Key, metre and doubled-track features of every rendition (or those lacking them)."""
    from ml.db import extractor_id

    exts = {k: extractor_id(con, *EXTRACTORS[k], {}) for k in ("key", "metre") if k in which}
    todo: set[int] = set()
    for k, ext in exts.items():
        todo.update(_rendition_ids(con, "rendition_feature", ext, recompute))
    n = 0
    for rid in sorted(todo):
        tracks = _load_notes(con, rid)
        meta = con.execute("SELECT meta_json FROM rendition WHERE id=?", (rid,)).fetchone()["meta_json"]
        meta = json.loads(meta) if meta else {}
        rows: list[tuple] = []
        if "key" in exts:
            pitched = [(p, d) for t in tracks.values() if t["role"] != "drums" for p, d in zip(t["pitches"], t["durations"])]
            tonic, mode, corr, margin = estimate_key([p for p, _ in pitched], [d for _, d in pitched])
            e = exts["key"]
            rows += [(rid, e, "key", None, f"{PC_NAMES[tonic]} {mode}"), (rid, e, "mode", None, mode),
                     (rid, e, "tonic", tonic, None), (rid, e, "key_confidence", round(margin, 4), None),
                     (rid, e, "key_corr", round(corr, 4), None)]
        if "metre" in exts:
            e = exts["metre"]
            ts = meta.get("time_signature")
            lead = [o for t in tracks.values() if t["role"] in ("melody", "guitar") for o in t["onsets"]]
            swing, n_off = swing_ratio(lead)
            dbl = doubled_tracks({tid: list(zip(t["onsets"], t["pitches"])) for tid, t in tracks.items() if t["role"] == "guitar"})
            rows += [(rid, e, "time_signature", None, ts or "4/4"), (rid, e, "beats_per_bar", beats_per_bar(ts), None),
                     (rid, e, "swing", swing, None), (rid, e, "n_offbeat", n_off, None),
                     (rid, e, "doubled_tracks", len(dbl), json.dumps(dbl))]
        _put(con, "rendition_feature", "rendition_id", rows)
        n += 1
        if n % 500 == 0:
            con.commit()
            log(f"  renditions {n}/{len(todo)}")
    con.commit()
    return n


def song_keys(con: sqlite3.Connection, log=print) -> int:
    """Song key = the rendition key with the largest summed confidence."""
    from ml.db import extractor_id

    key_ext = extractor_id(con, *EXTRACTORS["key"], {})
    ext = extractor_id(con, *EXTRACTORS["song_key"], {})
    votes: dict[int, Counter] = defaultdict(Counter)
    for r in con.execute("""SELECT r.song_id, k.value_text AS key, c.value_num AS conf
                            FROM rendition r JOIN rendition_feature k ON k.rendition_id=r.id AND k.extractor_id=? AND k.key='key'
                            JOIN rendition_feature c ON c.rendition_id=r.id AND c.extractor_id=? AND c.key='key_confidence'""",
                         (key_ext, key_ext)):
        votes[int(r["song_id"])][r["key"]] += float(r["conf"] or 0.0) + 1e-3
    rows = []
    for sid, c in votes.items():
        key, _ = c.most_common(1)[0]
        name, mode = key.split(" ", 1)
        rows += [(sid, ext, "key", None, key), (sid, ext, "mode", None, mode), (sid, ext, "tonic", PC_NAMES.index(name), None)]
    _put(con, "song_feature", "song_id", rows)
    con.commit()
    return len(votes)


def phrase_features(con: sqlite3.Connection, which: set[str], recompute: bool = False, log=print) -> int:
    """Bar position, accompaniment hits and trope labels of every phrase (or those lacking them)."""
    from ml.db import extractor_id

    exts = {k: extractor_id(con, *EXTRACTORS[k], {"tol": HIT_TOL} if k == "context" else {}) for k in ("context", "tropes") if k in which}
    todo: set[int] = set()
    for ext in exts.values():
        todo.update(_rendition_ids(con, "phrase_feature", ext, recompute))
    metre_ext = extractor_id(con, *EXTRACTORS["metre"], {})
    n = 0
    for rid in sorted(todo):
        tracks = _load_notes(con, rid) if "context" in exts else {}
        bass = sorted(o for t in tracks.values() if t["role"] == "bass" for o in t["onsets"])
        kick = sorted(o for t in tracks.values() if t["role"] == "drums" for o, p in zip(t["onsets"], t["pitches"]) if p in KICK)
        snare = sorted(o for t in tracks.values() if t["role"] == "drums" for o, p in zip(t["onsets"], t["pitches"]) if p in SNARE)
        bpb_row = con.execute("SELECT value_num FROM rendition_feature WHERE rendition_id=? AND extractor_id=? AND key='beats_per_bar'",
                              (rid, metre_ext)).fetchone()
        bpb = float(bpb_row["value_num"]) if bpb_row and bpb_row["value_num"] else 4.0
        song = con.execute("SELECT w.genre, s.name AS source FROM rendition r JOIN song w ON w.id=r.song_id "
                           "JOIN source s ON s.id=r.source_id WHERE r.id=?", (rid,)).fetchone()
        genre = song["genre"] if song["genre"] in GENRES else genre_of_source(song["source"])
        rows: list[tuple] = []
        for p in con.execute("SELECT id, start_beats, pitches_json, onsets_json FROM phrase WHERE rendition_id=?", (rid,)):
            pid, start = int(p["id"]), float(p["start_beats"] or 0.0)
            pitches, onsets = json.loads(p["pitches_json"]), json.loads(p["onsets_json"])
            absolute = [start + o for o in onsets]
            if "context" in exts:
                e = exts["context"]
                b, k, s = hits(absolute, bass), hits(absolute, kick), hits(absolute, snare)
                rows += [(pid, e, "bar_position", round(start % bpb, 4), None), (pid, e, "beats_per_bar", bpb, None),
                         (pid, e, "bass_hits", b.count("1") / max(1, len(b)) if bass else None, b if bass else None),
                         (pid, e, "kick_hits", k.count("1") / max(1, len(k)) if kick else None, k if kick else None),
                         (pid, e, "snare_hits", s.count("1") / max(1, len(s)) if snare else None, s if snare else None)]
            if "tropes" in exts:
                e = exts["tropes"]
                lab = label_tropes(pitches, [b - a for a, b in zip(onsets, onsets[1:])], genre)
                rows += [(pid, e, "tropes", len(lab["tropes"]), json.dumps([t["kind"] for t in lab["tropes"]])),
                         (pid, e, "familiarity", round(float(lab.get("familiarity", 0.0)), 4), None)]
        _put(con, "phrase_feature", "phrase_id", rows)
        n += 1
        if n % 500 == 0:
            con.commit()
            log(f"  renditions {n}/{len(todo)} (phrases)")
    con.commit()
    return n


def run(con: sqlite3.Connection, which: set[str] | None = None, recompute: bool = False, log=print) -> dict[str, int]:
    which = which or set(EXTRACTORS)
    out: dict[str, int] = {}
    if which & {"key", "metre"}:
        out["renditions"] = rendition_features(con, which, recompute, log)
    if "song_key" in which:
        out["songs"] = song_keys(con, log)
    if which & {"context", "tropes"}:
        out["phrase_renditions"] = phrase_features(con, which, recompute, log)
    return out


def stats(con: sqlite3.Connection) -> str:
    lines = []
    for table, col in (("song_feature", "song_id"), ("rendition_feature", "rendition_id"),
                       ("phrase_feature", "phrase_id"), ("note_feature", "note_id")):
        for r in con.execute(f"SELECT e.name, e.version, COUNT(*) n, COUNT(DISTINCT f.{col}) ids FROM {table} f "
                             "JOIN extractor e ON e.id=f.extractor_id GROUP BY e.id"):
            lines.append(f"{table:18s} {r['name']} v{r['version']}: {r['n']} values over {r['ids']} rows")
    for r in con.execute("SELECT value_text k, COUNT(*) n FROM song_feature WHERE key='key' GROUP BY k ORDER BY n DESC LIMIT 6"):
        lines.append(f"  song key {r['k']:9s} {r['n']}")
    for r in con.execute("SELECT value_text k, COUNT(*) n FROM rendition_feature WHERE key='time_signature' GROUP BY k ORDER BY n DESC LIMIT 5"):
        lines.append(f"  time signature {r['k']:5s} {r['n']}")
    r = con.execute("SELECT AVG(value_num) a, COUNT(value_num) n FROM rendition_feature WHERE key='swing'").fetchone()
    if r and r["n"]:
        lines.append(f"  swing: mean {r['a']:.3f} over {r['n']} renditions with enough off-beat notes")
    for key in ("bass_hits", "kick_hits", "snare_hits"):
        r = con.execute(f"SELECT AVG(value_num) a, COUNT(value_num) n FROM phrase_feature WHERE key='{key}'").fetchone()
        if r and r["n"]:
            lines.append(f"  {key}: mean share {r['a']:.3f} over {r['n']} phrases with that track")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from ml.db import connect, init

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/riffs.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--only", default=None, help="comma list of " + ",".join(EXTRACTORS))
    p.add_argument("--recompute", action="store_true")
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    con = connect(args.db)
    init(con)  # creates feature tables missing from an older database
    if args.cmd == "run":
        which = set(args.only.split(",")) if args.only else None
        print(run(con, which, args.recompute))
    else:
        print(stats(con))


if __name__ == "__main__":
    main()
