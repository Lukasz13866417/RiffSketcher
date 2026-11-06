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
    write_midi(lakh / "Band" / "Song.1.mid", melody=False, tempo=100)  # same work, second rendition
    write_midi(lakh / "Other" / "Tune.mid", melody=False)
    return pop, lakh


def test_ingest_works_renditions_tracks_notes_phrases(tmp_path, raw):
    pop, lakh = raw
    con = db.connect(tmp_path / "t.db")
    db.init(con)
    counts = db.ingest(con, db.jobs_for(pop, lakh, None, 5, 16), workers=1, log=lambda *a: None)
    assert counts["renditions"] == 5 and counts.get("failed", 0) == 0
    works = {r["key"]: r for r in con.execute("SELECT * FROM work")}
    assert set(works) == {"001", "002", "Band/Song", "Other/Tune"}
    assert works["Band/Song"]["artist"] == "Band" and works["Band/Song"]["title_norm"] == "song"
    n_rend = con.execute("SELECT COUNT(*) c FROM rendition r JOIN work w ON w.id=r.work_id WHERE w.key='Band/Song'").fetchone()["c"]
    assert n_rend == 2  # the .1 version joined the same work
    roles = {r["role"]: r["c"] for r in con.execute("SELECT role, COUNT(*) c FROM track GROUP BY role")}
    assert roles["melody"] == 2 and roles["guitar"] == 3 and roles["drums"] == 3
    # notes stored only for kept roles; onsets in beats at 120 bpm: 0.5 s = 1 beat
    row = con.execute("""SELECT n.onset_beats, n.pitch, n.velocity FROM note n JOIN track t ON t.id=n.track_id
                         WHERE t.role='melody' ORDER BY t.id, n.idx LIMIT 2""").fetchall()
    assert [r["pitch"] for r in row] == [60, 62] and row[1]["onset_beats"] == pytest.approx(1.0)
    assert con.execute("SELECT COUNT(*) c FROM note n JOIN track t ON t.id=n.track_id WHERE t.role='drums'").fetchone()["c"] == 0
    assert con.execute("SELECT COUNT(*) c FROM phrase").fetchone()["c"] > 0
    # re-running ingest is a no-op for files already present
    again = db.ingest(con, db.jobs_for(pop, lakh, None, 5, 16), workers=1, log=lambda *a: None)
    assert again.get("renditions", 0) == 0


def test_split_export_and_features(tmp_path, raw):
    pop, lakh = raw
    con = db.connect(tmp_path / "t.db")
    db.init(con)
    db.ingest(con, db.jobs_for(pop, lakh, None, 5, 16), workers=1, log=lambda *a: None)
    counts = db.assign_splits(con, 0.25, 0.25, seed=0, base_ids={"test": ["001"]})
    assert sum(counts.values()) == 4
    split = {r["key"]: r["name"] for r in con.execute("SELECT w.key, s.name FROM split s JOIN work w ON w.id=s.work_id")}
    assert split["001"] == "test"  # kept from the base splits
    assert db.assign_splits(con, 0.25, 0.25, seed=1)["test"] >= 1 and {r["key"]: r["name"] for r in con.execute(
        "SELECT w.key, s.name FROM split s JOIN work w ON w.id=s.work_id")} == split  # re-running keeps assignments
    out = tmp_path / "phrases.jsonl"
    n, sp = db.export(con, out, max_per_song=None)
    phrases = load_phrases(out)
    assert n == len(phrases) == con.execute("SELECT COUNT(*) c FROM phrase").fetchone()["c"]
    assert {p.source for p in phrases} == {"pop909", "lakh_guitar"}
    ids = json.loads((tmp_path / "splits.json").read_text())
    assert "001" in ids["test"] and sum(len(v) for v in ids.values()) == 4
    # feature tables with extractor provenance
    ex = db.extractor_id(con, "test_extractor", "0.1", {"k": 1})
    assert db.extractor_id(con, "test_extractor", "0.1", {"k": 1}) == ex
    wid = con.execute("SELECT id FROM work WHERE key='001'").fetchone()["id"]
    con.execute("INSERT INTO work_feature(work_id, extractor_id, key, value_num) VALUES (?,?,?,?)", (wid, ex, "chug_runs", 3))
    nid = con.execute("SELECT id FROM note LIMIT 1").fetchone()["id"]
    con.execute("INSERT INTO note_feature(note_id, extractor_id, key, value_num) VALUES (?,?,?,?)", (nid, ex, "palm_mute_p", 0.9))
    con.commit()
    assert con.execute("SELECT value_num FROM note_feature WHERE note_id=?", (nid,)).fetchone()["value_num"] == 0.9
    assert "works with more than one rendition: 1" in db.stats(con)


def test_norm_text_and_lakh_key(tmp_path):
    assert db.norm_text("  The Song! (Live) ") == "the song live"
    root = tmp_path
    key, artist, title = db.lakh_song_key(root / "AC DC" / "Back In Black.2.mid", root)
    assert key == "AC DC/Back In Black" and artist == "AC DC" and title == "Back In Black"
