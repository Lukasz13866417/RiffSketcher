"""The song database: schema, ingest from MIDI, work matching, splits, export, features."""

import json

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

from ml import db  # noqa: E402
from ml.datasets import load_phrases  # noqa: E402


def write_midi(path, melody=True, guitar=True, tempo=120):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    if melody:
        ins = pretty_midi.Instrument(program=0, name="MELODY")
        t = 0.0
        for p in [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65]:
            ins.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=t, end=t + 0.4))
            t += 0.5
        pm.instruments.append(ins)
    if guitar:
        g = pretty_midi.Instrument(program=29, name="Distortion Guitar")
        t = 0.0
        for p in [40, 40, 40, 43, 40, 40, 40, 46, 45, 43, 40, 40]:
            g.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=t, end=t + 0.2))
            t += 0.25
        pm.instruments.append(g)
        d = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        d.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0, end=0.1))
        pm.instruments.append(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


@pytest.fixture
def raw(tmp_path):
    pop = tmp_path / "pop909"
    for sid in ("001", "002"):
        write_midi(pop / sid / f"{sid}.mid", guitar=False)
    write_midi(pop / "001" / "versions" / "001-v2.mid", guitar=False)  # must be skipped
    lakh = tmp_path / "lakh"
    write_midi(lakh / "Band" / "Song.mid", melody=False)
    write_midi(lakh / "Band" / "Song.1.mid", melody=False, tempo=100)  # second transcription, same work
    write_midi(lakh / "Band" / "song.mid", melody=False)  # case variant of the title: same work
    write_midi(lakh / "Other" / "Tune.mid", melody=False)
    return pop, lakh


def ingest_all(con, pop, lakh):
    return db.ingest(con, db.jobs_for(pop, lakh, None, 5, 16), workers=1, log=lambda *a: None)


def test_ingest_ids_works_renditions_tracks_notes_phrases(tmp_path, raw):
    pop, lakh = raw
    con = db.connect(tmp_path / "t.db")
    db.init(con)
    counts = ingest_all(con, pop, lakh)
    assert counts["renditions"] == 6 and counts.get("failed", 0) == 0
    works = con.execute("SELECT id, title, artist FROM work ORDER BY id").fetchall()
    assert [w["id"] for w in works] == [1, 2, 3, 4]  # integer song ids
    by_name = {(w["artist"], w["title"]): w["id"] for w in works}
    assert (None, None) in dict.fromkeys(by_name) or any(w["title"] is None for w in works)  # POP909 has no names
    song = con.execute("SELECT id FROM work WHERE artist_norm='band' AND title_norm='song'").fetchone()["id"]
    rends = con.execute("SELECT id, original_id FROM rendition WHERE work_id=? ORDER BY id", (song,)).fetchall()
    assert len(rends) == 3  # Song.mid, Song.1.mid and song.mid all joined one work
    assert {r["original_id"] for r in rends} == {"Band/Song.mid", "Band/Song.1.mid", "Band/song.mid"}
    pop_rend = con.execute("SELECT original_id FROM rendition r JOIN source s ON s.id=r.source_id WHERE s.name='pop909' ORDER BY r.id").fetchall()
    assert [r["original_id"] for r in pop_rend] == ["001", "002"]
    roles = {r["role"]: r["c"] for r in con.execute("SELECT role, COUNT(*) c FROM track GROUP BY role")}
    assert roles["melody"] == 2 and roles["guitar"] == 4 and roles["drums"] == 4
    row = con.execute("""SELECT n.onset_beats, n.pitch FROM note n JOIN track t ON t.id=n.track_id
                         WHERE t.role='melody' ORDER BY t.id, n.idx LIMIT 2""").fetchall()
    assert [r["pitch"] for r in row] == [60, 62] and row[1]["onset_beats"] == pytest.approx(1.0)
    assert con.execute("SELECT COUNT(*) c FROM note n JOIN track t ON t.id=n.track_id WHERE t.role='drums'").fetchone()["c"] == 0
    assert con.execute("SELECT COUNT(*) c FROM phrase").fetchone()["c"] > 0
    assert ingest_all(con, pop, lakh).get("renditions", 0) == 0  # already present: no-op


def test_split_export_and_features(tmp_path, raw):
    pop, lakh = raw
    con = db.connect(tmp_path / "t.db")
    db.init(con)
    ingest_all(con, pop, lakh)
    counts = db.assign_splits(con, 0.25, 0.25, seed=0)
    assert sum(counts.values()) == 4 and counts["test"] == 1 and counts["val"] == 1
    first = {r["work_id"]: r["name"] for r in con.execute("SELECT work_id, name FROM split")}
    db.assign_splits(con, 0.25, 0.25, seed=5)  # re-running keeps every existing assignment
    assert {r["work_id"]: r["name"] for r in con.execute("SELECT work_id, name FROM split")} == first
    out = tmp_path / "phrases.jsonl"
    n, sp = db.export(con, out, max_per_song=None)
    phrases = load_phrases(out)
    assert n == len(phrases) == con.execute("SELECT COUNT(*) c FROM phrase").fetchone()["c"]
    assert {p.source for p in phrases} == {"pop909", "lakh_guitar"}
    assert all(p.song_id.isdigit() and p.rendition_id is not None for p in phrases)
    ids = json.loads((tmp_path / "splits.json").read_text())
    assert sum(len(v) for v in ids.values()) == 4 and all(k.isdigit() for v in ids.values() for k in v)
    # the cap applies per work across renditions, and not to exempt sources
    n2, _ = db.export(con, out, max_per_song=1)
    capped = load_phrases(out)
    assert max(sum(p.song_id == s for p in capped if p.source == "lakh_guitar") for s in {p.song_id for p in capped}) == 1
    assert sum(p.source == "pop909" for p in capped) == sum(p.source == "pop909" for p in phrases)
    # feature tables with extractor provenance
    ex = db.extractor_id(con, "test_extractor", "0.1", {"k": 1})
    assert db.extractor_id(con, "test_extractor", "0.1", {"k": 1}) == ex
    con.execute("INSERT INTO work_feature(work_id, extractor_id, key, value_num) VALUES (?,?,?,?)", (1, ex, "chug_runs", 3))
    nid = con.execute("SELECT id FROM note LIMIT 1").fetchone()["id"]
    con.execute("INSERT INTO note_feature(note_id, extractor_id, key, value_num) VALUES (?,?,?,?)", (nid, ex, "palm_mute_p", 0.9))
    con.commit()
    assert con.execute("SELECT value_num FROM note_feature WHERE note_id=?", (nid,)).fetchone()["value_num"] == 0.9
    assert "works with more than one rendition: 1" in db.stats(con)


def test_norm_text():
    assert db.norm_text("  The Song! (Live) ") == "the song live"
    assert db.norm_text(None) == ""
