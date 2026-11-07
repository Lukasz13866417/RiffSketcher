"""The song database: schema, ingest from MIDI, song matching, splits, export, features."""

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
    write_midi(lakh / "Band" / "Song.1.mid", melody=False, tempo=100)  # second transcription, same song
    write_midi(lakh / "Band" / "song.mid", melody=False)  # case variant of the title: same song
    write_midi(lakh / "Other" / "Tune.mid", melody=False)
    return pop, lakh


def ingest_all(con, pop, lakh):
    return db.ingest(con, db.jobs_for(pop, lakh, None, 5, 16), workers=1, log=lambda *a: None)


def test_ingest_ids_songs_renditions_tracks_notes_phrases(tmp_path, raw):
    pop, lakh = raw
    con = db.connect(tmp_path / "t.db")
    db.init(con)
    counts = ingest_all(con, pop, lakh)
    assert counts["renditions"] == 6 and counts.get("failed", 0) == 0
    songs = con.execute("SELECT id, title, artist FROM song ORDER BY id").fetchall()
    assert [w["id"] for w in songs] == [1, 2, 3, 4]  # integer song ids
    by_name = {(w["artist"], w["title"]): w["id"] for w in songs}
    assert (None, None) in dict.fromkeys(by_name) or any(w["title"] is None for w in songs)  # POP909 has no names
    song = con.execute("SELECT id FROM song WHERE artist_norm='band' AND title_norm='song'").fetchone()["id"]
    rends = con.execute("SELECT id, original_id FROM rendition WHERE song_id=? ORDER BY id", (song,)).fetchall()
    assert len(rends) == 3  # Song.mid, Song.1.mid and song.mid all joined one song
    assert {r["original_id"] for r in rends} == {"Band/Song.mid", "Band/Song.1.mid", "Band/song.mid"}
    pop_rend = con.execute("SELECT original_id FROM rendition r JOIN source s ON s.id=r.source_id WHERE s.name='pop909' ORDER BY r.id").fetchall()
    assert [r["original_id"] for r in pop_rend] == ["001", "002"]
    roles = {r["role"]: r["c"] for r in con.execute("SELECT role, COUNT(*) c FROM track GROUP BY role")}
    assert roles["melody"] == 2 and roles["guitar"] == 4 and roles["drums"] == 4
    row = con.execute("""SELECT n.onset_beats, n.pitch FROM note n JOIN track t ON t.id=n.track_id
                         WHERE t.role='melody' ORDER BY t.id, n.idx LIMIT 2""").fetchall()
    assert [r["pitch"] for r in row] == [60, 62] and row[1]["onset_beats"] == pytest.approx(1.0)
    # drum tracks keep kick and snare hits (the fixture's one kick per file), nothing else
    assert con.execute("SELECT COUNT(*) c FROM note n JOIN track t ON t.id=n.track_id WHERE t.role='drums'").fetchone()["c"] == 4
    assert con.execute("SELECT COUNT(*) c FROM phrase").fetchone()["c"] > 0
    assert ingest_all(con, pop, lakh).get("renditions", 0) == 0  # already present: no-op


def test_split_export_and_features(tmp_path, raw):
    pop, lakh = raw
    con = db.connect(tmp_path / "t.db")
    db.init(con)
    ingest_all(con, pop, lakh)
    counts = db.assign_splits(con, 0.25, 0.25, seed=0)
    assert sum(counts.values()) == 4 and counts["test"] == 1 and counts["val"] == 1
    first = {r["song_id"]: r["name"] for r in con.execute("SELECT song_id, name FROM split")}
    db.assign_splits(con, 0.25, 0.25, seed=5)  # re-running keeps every existing assignment
    assert {r["song_id"]: r["name"] for r in con.execute("SELECT song_id, name FROM split")} == first
    out = tmp_path / "phrases.jsonl"
    n, sp = db.export(con, out, max_per_song=None)
    phrases = load_phrases(out)
    assert n == len(phrases) == con.execute("SELECT COUNT(*) c FROM phrase").fetchone()["c"]
    assert {p.source for p in phrases} == {"pop909", "lakh_guitar"}
    assert all(p.song_id.isdigit() and p.rendition_id is not None for p in phrases)
    ids = json.loads((tmp_path / "splits.json").read_text())
    assert sum(len(v) for v in ids.values()) == 4 and all(k.isdigit() for v in ids.values() for k in v)
    # the cap applies per song across renditions, and not to exempt sources
    n2, _ = db.export(con, out, max_per_song=1)
    capped = load_phrases(out)
    assert max(sum(p.song_id == s for p in capped if p.source == "lakh_guitar") for s in {p.song_id for p in capped}) == 1
    assert sum(p.source == "pop909" for p in capped) == sum(p.source == "pop909" for p in phrases)
    # feature tables with extractor provenance
    ex = db.extractor_id(con, "test_extractor", "0.1", {"k": 1})
    assert db.extractor_id(con, "test_extractor", "0.1", {"k": 1}) == ex
    con.execute("INSERT INTO song_feature(song_id, extractor_id, key, value_num) VALUES (?,?,?,?)", (1, ex, "chug_runs", 3))
    nid = con.execute("SELECT id FROM note LIMIT 1").fetchone()["id"]
    con.execute("INSERT INTO note_feature(note_id, extractor_id, key, value_num) VALUES (?,?,?,?)", (nid, ex, "palm_mute_p", 0.9))
    con.commit()
    assert con.execute("SELECT value_num FROM note_feature WHERE note_id=?", (nid,)).fetchone()["value_num"] == 0.9
    assert "songs with more than one rendition: 1" in db.stats(con)


def test_norm_text():
    assert db.norm_text("  The Song! (Live) ") == "the song live"
    assert db.norm_text(None) == ""


def test_artist_and_title_keys():
    for a in ("Guns N' Roses", "Guns 'n Roses", "Guns & Roses", "Guns And Roses", "guns n roses", "The Guns N' Roses"):
        assert db.artist_key(a) == "guns and roses", a
    assert db.artist_key("The Beatles") == "beatles" and db.artist_key("Beatles") == "beatles"
    assert db.artist_key("Motörhead") == "motorhead"
    assert db.title_key("Sweet Child O' Mine (Live)") == db.title_key("Sweet Child O'Mine") == "sweet child o mine"
    assert db.title_key("Enter Sandman - Remastered 2011") == "enter sandman"
    assert db.title_key("Country Grammar (Hot ...)") == "country grammar"
    assert db.title_key("(Live)") == "live"  # a title that is only a parenthetical keeps its text


def test_fuzzy_matching_merges_spellings(tmp_path):
    con = db.connect(tmp_path / "m.db")
    db.init(con)
    a = db.get_or_create_song(con, "Sweet Child O' Mine", "Guns N' Roses")
    assert db.get_or_create_song(con, "Sweet Child O'Mine (Live)", "Guns And Roses") == a
    assert db.get_or_create_song(con, "Sweet Child O Mine", "Guns 'n Roses") == a
    assert db.get_or_create_song(con, "Sweet Child O' Mine", "Guns N Rose") == a  # typo: fuzzy backstop
    assert db.get_or_create_song(con, "Sweet Child O' Mine", "Metallica") != a  # different artist, same title
    assert db.get_or_create_song(con, "Paradise City", "Guns N' Roses") != a
    assert db.get_or_create_song(con, "Paradise Citty", "Guns N' Roses") == db.get_or_create_song(con, "Paradise City", "Guns N' Roses")  # typo
    assert db.get_or_create_song(con, None, None) != db.get_or_create_song(con, None, None)  # nameless: never matched
    assert con.execute("SELECT COUNT(*) c FROM song").fetchone()["c"] == 5


def test_dedupe_and_merge(tmp_path):
    con = db.connect(tmp_path / "d.db")
    db.init(con)
    # insert with the raw (old-style) keys so dedupe has something to find
    for i, (title, artist) in enumerate([("Song", "Guns N' Roses"), ("Song", "Guns And Roses"), ("Song (Live)", "Guns & Roses"), ("Other", "Band")]):
        con.execute("INSERT INTO song(title, artist, title_norm, artist_norm) VALUES (?,?,?,?)",
                    (title, artist, db.norm_text(title), db.norm_text(artist)))
    sid = db.source_id(con, "pop909")
    for i in range(1, 5):
        con.execute("INSERT INTO rendition(song_id, source_id, original_id, path) VALUES (?,?,?,?)", (i, sid, f"o{i}", f"p{i}"))
    con.execute("INSERT INTO split(song_id, name, seed) VALUES (2, 'test', 0)")
    con.commit()
    dry = db.dedupe_songs(con, dry_run=True, log=lambda *a: None)
    assert len(dry) == 2 and con.execute("SELECT COUNT(*) c FROM song").fetchone()["c"] == 4
    merges = db.dedupe_songs(con, log=lambda *a: None)
    assert sorted((k, d) for k, d, _ in merges) == [(1, 2), (1, 3)]
    assert con.execute("SELECT COUNT(*) c FROM song").fetchone()["c"] == 2
    assert con.execute("SELECT COUNT(*) c FROM rendition WHERE song_id=1").fetchone()["c"] == 3
    assert con.execute("SELECT name FROM split WHERE song_id=1").fetchone()["name"] == "test"  # moved from the dropped song
    db.merge_songs(con, 1, 4); con.commit()
    assert con.execute("SELECT COUNT(*) c FROM song").fetchone()["c"] == 1


def test_phrases_are_cut_from_notes_and_link_to_them(tmp_path, raw):
    pop, lakh = raw
    con = db.connect(tmp_path / "c.db")
    db.init(con)
    counts = ingest_all(con, pop, lakh)
    assert counts["phrases"] > 0
    rows = con.execute("""SELECT p.id, p.track_id, p.note_ids_json, p.pitches_json, p.start_beats, t.role
                          FROM phrase p JOIN track t ON t.id = p.track_id""").fetchall()
    assert rows and {r["role"] for r in rows} == {"melody", "guitar"}
    for r in rows:
        note_ids = json.loads(r["note_ids_json"]); pitches = json.loads(r["pitches_json"])
        assert len(note_ids) == len(pitches)
        stored = [con.execute("SELECT pitch, track_id, onset_beats FROM note WHERE id=?", (i,)).fetchone() for i in note_ids]
        assert [x["pitch"] for x in stored] == pitches  # the phrase's pitches are exactly its notes' pitches
        assert all(x["track_id"] == r["track_id"] for x in stored)
        assert stored[0]["onset_beats"] == r["start_beats"]
    ext = con.execute("SELECT name, version FROM extractor e JOIN phrase p ON p.extractor_id=e.id LIMIT 1").fetchone()
    assert (ext["name"], ext["version"]) == (db.CUT_EXTRACTOR, db.CUT_VERSION)
    # recut replaces, never duplicates
    n_before = con.execute("SELECT COUNT(*) c FROM phrase").fetchone()["c"]
    assert db.cut_phrases(con, None, log=lambda *a: None) == n_before
    assert con.execute("SELECT COUNT(*) c FROM phrase").fetchone()["c"] == n_before
    # the export carries phrase ids
    out = tmp_path / "p.jsonl"
    db.export(con, out)
    assert all(p.phrase_id is not None for p in load_phrases(out))


def test_onsets_with_ids_groups_chords_and_keeps_lowest_note():
    import sqlite3
    con = sqlite3.connect(":memory:"); con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE note(id INTEGER PRIMARY KEY, onset_beats REAL, duration_beats REAL, pitch INTEGER)")
    con.executemany("INSERT INTO note VALUES (?,?,?,?)", [(1, 0.0, 1.0, 52), (2, 0.01, 0.5, 40), (3, 0.02, 0.7, 47), (4, 1.0, 1.0, 43)])
    onsets, lowest = db._onsets_with_ids(con.execute("SELECT * FROM note").fetchall())
    assert len(onsets) == 2 and sorted(onsets[0][2]) == [40, 47, 52] and lowest == [2, 4]
    assert onsets[0][1] == pytest.approx(0.51)  # the onset ends with its lowest note


def write_jams(path, tempo=120.0, notes=((0, 0.0, 0.5, 40), (0, 0.5, 0.5, 43), (1, 1.0, 0.5, 45), (1, 1.5, 0.5, 47), (2, 2.0, 0.5, 50))):
    """A minimal GuitarSet-style JAMS: notes as (string, time_s, duration_s, midi), beats every 0.5 s."""
    import json
    spb = 60.0 / tempo
    per_string = {}
    for s, t, d, p in notes:
        per_string.setdefault(s, []).append({"time": t, "duration": d, "value": p + 0.03, "confidence": None})
    anns = [{"namespace": "note_midi", "annotation_metadata": {"data_source": str(s)}, "data": data, "sandbox": {}}
            for s, data in per_string.items()]
    anns.append({"namespace": "beat_position", "annotation_metadata": {},
                 "data": [{"time": k * spb, "duration": 0.0, "value": {"position": k % 4 + 1, "beat_units": 4, "measure": k // 4 + 1, "num_beats": 4},
                           "confidence": None} for k in range(8)], "sandbox": {}})
    anns.append({"namespace": "tempo", "annotation_metadata": {}, "data": [{"time": 0.0, "duration": 4.0, "value": tempo, "confidence": 1.0}], "sandbox": {}})
    anns.append({"namespace": "key_mode", "annotation_metadata": {}, "data": [{"time": 0.0, "duration": 4.0, "value": "E:minor", "confidence": 1}], "sandbox": {}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"annotations": anns, "file_metadata": {"title": path.stem, "duration": 4.0}, "sandbox": {}}))


def test_guitarset_ingest_stores_strings_frets_beats_and_style_genre(tmp_path):
    gs = tmp_path / "guitarset"
    write_jams(gs / "00_Rock1-120-E_comp.jams")
    write_jams(gs / "00_Rock1-120-E_solo.jams")
    write_jams(gs / "02_BN3-100-A_solo.jams", tempo=100.0)
    con = db.connect(tmp_path / "t.db")
    db.init(con)
    counts = db.ingest(con, db.jobs_for(None, None, None, 5, 16, guitarset_dir=gs), workers=1, log=lambda *a: None)
    assert counts["renditions"] == 3 and counts["notes"] == 15
    songs = {r["title"]: r for r in con.execute("SELECT title, artist, genre, (SELECT COUNT(*) FROM rendition x WHERE x.song_id=song.id) n FROM song")}
    assert songs["Rock1-120-E"]["n"] == 2 and songs["Rock1-120-E"]["genre"] == "rock" and songs["Rock1-120-E"]["artist"] == "GuitarSet"
    assert songs["BN3-100-A"]["genre"] == "jazz"
    rows = con.execute("""SELECT n.onset_beats, n.duration_beats, n.pitch, n.string, n.fret FROM note n JOIN track t ON t.id=n.track_id
                          JOIN rendition r ON r.id=t.rendition_id WHERE r.original_id='00_Rock1-120-E_comp' ORDER BY n.idx""").fetchall()
    assert [r["pitch"] for r in rows] == [40, 43, 45, 47, 50]
    assert [r["string"] for r in rows] == [0, 0, 1, 1, 2] and [r["fret"] for r in rows] == [0, 3, 0, 2, 0]
    assert [round(r["onset_beats"], 3) for r in rows] == [0.0, 1.0, 2.0, 3.0, 4.0]  # 120 BPM: 0.5 s is one beat
    assert all(abs(r["duration_beats"] - 1.0) < 1e-6 for r in rows)
    import json
    meta = json.loads(con.execute("SELECT meta_json FROM rendition WHERE original_id='02_BN3-100-A_solo'").fetchone()["meta_json"])
    assert meta["time_signature"] == "4/4" and meta["key"] == "E:minor" and meta["style"] == "BN" and meta["mode"] == "solo"
    # phrases were cut from the stored notes and link to them
    assert con.execute("SELECT COUNT(*) c FROM phrase").fetchone()["c"] >= 1
    # the genre lookup leaves a source-given genre alone when it knows nothing about the artist
    from ml.genres import apply
    apply(con, {}, log=lambda *a: None)
    assert con.execute("SELECT genre FROM song WHERE title='Rock1-120-E'").fetchone()["genre"] == "rock"
