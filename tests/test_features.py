import json

import pretty_midi
import pytest

from ml import db, features


def write_band_midi(path, tempo=120):
    """Guitar riff on eighths, bass on beats, kick on 1 and 3, snare on 2 and 4, in 4/4."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    beat = 60.0 / tempo
    g = pretty_midi.Instrument(program=29, name="Distortion Guitar")
    for i, p in enumerate([40, 40, 43, 43, 45, 45, 47, 47, 40, 40, 43, 43, 45, 45, 47, 47]):
        g.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=i * beat / 2, end=i * beat / 2 + 0.2))
    pm.instruments.append(g)
    g2 = pretty_midi.Instrument(program=30, name="Guitar 2")  # doubles the first one
    g2.notes = list(g.notes)
    pm.instruments.append(g2)
    b = pretty_midi.Instrument(program=33, name="Bass")
    for i in range(8):
        b.notes.append(pretty_midi.Note(velocity=100, pitch=28, start=i * beat, end=i * beat + 0.3))
    pm.instruments.append(b)
    d = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for i in range(8):
        d.notes.append(pretty_midi.Note(velocity=100, pitch=36 if i % 2 == 0 else 38, start=i * beat, end=i * beat + 0.05))
        d.notes.append(pretty_midi.Note(velocity=80, pitch=42, start=i * beat + beat / 2, end=i * beat + beat / 2 + 0.05))  # hi-hat
    pm.instruments.append(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def test_key_metre_and_hits_helpers():
    tonic, mode, corr, margin = features.estimate_key([60, 62, 64, 65, 67, 69, 71, 72], [1] * 8)
    assert (tonic, mode) == (0, "major") and corr > 0.5 and margin > 0
    tonic, mode, _, _ = features.estimate_key([57, 59, 60, 62, 64, 65, 67, 69], [1] * 8)
    assert (tonic, mode) == (9, "minor")
    assert features.beats_per_bar("4/4") == 4 and features.beats_per_bar("6/8") == 3 and features.beats_per_bar(None) == 4
    swing, n = features.swing_ratio([i + f for i in range(30) for f in (0.0, 0.67)])
    assert n == 30 and abs(swing - 0.67) < 1e-6
    assert features.swing_ratio([0.0, 0.5])[0] is None
    assert features.hits([0.0, 0.5, 1.0, 1.5], [0.0, 1.0, 2.0]) == "1010"
    assert features.doubled_tracks({1: [(0, 40), (1, 43)], 2: [(0, 40), (1, 43)], 3: [(0, 50)]}) == [2]


def test_features_over_an_ingested_database(tmp_path):
    lakh = tmp_path / "lakh"
    write_band_midi(lakh / "Band" / "Riff.mid")
    con = db.connect(tmp_path / "t.db")
    db.init(con)
    counts = db.ingest(con, db.jobs_for(None, lakh, None, 5, 16), workers=1, log=lambda *a: None)
    # bass and kick/snare notes are stored, the hi-hat is not
    roles = {r["role"]: r["n"] for r in con.execute(
        "SELECT t.role, COUNT(n.id) n FROM track t LEFT JOIN note n ON n.track_id=t.id GROUP BY t.role")}
    assert roles["bass"] == 8 and roles["drums"] == 8 and roles["guitar"] == 32
    meta = json.loads(con.execute("SELECT meta_json FROM rendition").fetchone()["meta_json"])
    assert meta["time_signature"] == "4/4"

    out = features.run(con, log=lambda *a: None)
    assert out["renditions"] == 1 and out["songs"] == 1 and out["phrase_renditions"] == 1
    rf = {r["key"]: (r["value_num"], r["value_text"]) for r in con.execute("SELECT key, value_num, value_text FROM rendition_feature")}
    assert rf["key"][1] in ("E minor", "G major") and rf["beats_per_bar"][0] == 4 and rf["time_signature"][1] == "4/4"
    assert rf["doubled_tracks"][0] == 1
    sf = {r["key"]: r["value_text"] for r in con.execute("SELECT key, value_text FROM song_feature")}
    assert sf["key"] == rf["key"][1]
    pf = {r["key"]: (r["value_num"], r["value_text"]) for r in con.execute(
        "SELECT key, value_num, value_text FROM phrase_feature WHERE phrase_id=(SELECT MIN(id) FROM phrase)")}
    n = con.execute("SELECT n_notes FROM phrase ORDER BY id LIMIT 1").fetchone()["n_notes"]
    assert len(pf["bass_hits"][1]) == n and pf["bass_hits"][1].startswith("10")  # eighths: every other with the bass
    assert pf["kick_hits"][1].startswith("1000") and pf["snare_hits"][1].startswith("0010")
    assert pf["bar_position"][0] == 0.0 and pf["beats_per_bar"][0] == 4
    assert isinstance(json.loads(pf["tropes"][1]), list)
    # a second run finds nothing to do; recompute redoes it
    assert features.run(con, log=lambda *a: None) == {"renditions": 0, "songs": 1, "phrase_renditions": 0}
    assert features.run(con, recompute=True, log=lambda *a: None)["renditions"] == 1
    assert "rendition_key v1" in features.stats(con)


def test_add_notes_fills_tracks_ingested_without_them(tmp_path):
    lakh = tmp_path / "lakh"
    write_band_midi(lakh / "Band" / "Riff.mid")
    con = db.connect(tmp_path / "t.db")
    db.init(con)
    jobs = [(k, p, s, o, ("guitar",), a, b, ar, ti) for k, p, s, o, _, a, b, ar, ti in db.jobs_for(None, lakh, None, 5, 16)]
    db.ingest(con, jobs, workers=1, log=lambda *a: None)
    assert con.execute("SELECT COUNT(*) c FROM track WHERE notes_stored=0 AND role IN ('bass','drums')").fetchone()["c"] == 2
    out = db.add_notes(con, ("bass", "drums"), log=lambda *a: None)
    assert out == {"renditions": 1, "tracks": 2, "notes": 16}
    assert con.execute("SELECT COUNT(*) c FROM track WHERE notes_stored=0 AND role IN ('bass','drums')").fetchone()["c"] == 0
    assert json.loads(con.execute("SELECT meta_json FROM rendition").fetchone()["meta_json"])["time_signature"] == "4/4"


def test_genre_from_tags():
    from ml.genres import genre_from_tags
    assert genre_from_tags([("heavy metal", 41), ("rock", 1)]) == "metal"
    assert genre_from_tags([("electronic", 42), ("pop", 1)]) == "electronic"
    assert genre_from_tags([("pop punk", 3)]) == "punk"  # ties go to the more specific genre
    assert genre_from_tags([("rock", 4), ("industrial rock", -2)]) == "rock"
    assert genre_from_tags([("french", 2)]) is None and genre_from_tags([]) is None
