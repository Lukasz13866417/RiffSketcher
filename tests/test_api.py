from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SKETCH = {
    "observations": [{"t": i * 0.5, "y": y, "confidence": 0.8} for i, y in enumerate([0, 3, 5, 7, 5, 3, 0])],
    "gestures": [], "constraints": [], "tempo_hint": None, "meta": {},
}


def test_health_and_gestures():
    assert client.get("/api/health").json() == {"status": "ok"}
    lib = client.get("/api/gestures").json()
    assert "articulation" in lib and "palm_mute" in lib["articulation"]["items"]


def test_reconstruct_and_synthesize():
    r = client.post("/api/reconstruct", json={"sketch": SKETCH, "n_candidates": 3})
    assert r.status_code == 200
    cands = r.json()["candidates"]
    assert 1 <= len(cands) <= 3
    assert len(cands[0]["notes"]) == 7 and len(cands[0]["note_names"]) == 7
    wav = client.post("/api/synthesize", json=cands[0])
    assert wav.status_code == 200 and wav.headers["content-type"] == "audio/wav"
    assert wav.content[:4] == b"RIFF"


def test_reconstruct_empty_is_400():
    assert client.post("/api/reconstruct", json={"sketch": {"observations": []}}).status_code == 400


def test_suggest_merge_quantise():
    s = client.post("/api/suggest-gestures", json=SKETCH).json()["gestures"]
    assert any(g["kind"] == "ascending_run" for g in s)
    m = client.post("/api/merge-passes", json={"passes": [[0, 0.5, 1], [5, 5.5, 6, 6.5]]}).json()
    assert len(m["onsets"]) == 4
    q = client.post("/api/quantise", json={"onsets": [0, 0.5, 1.0, 1.5]}).json()
    assert q["onsets_beats"][0] == 0.0 and q["tempo"] > 0


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and b"Riff Sketcher" in r.content


def test_engines_and_explicit_engine():
    e = client.get("/api/engines").json()
    names = {x["name"] for x in e["engines"]}
    assert "deterministic" in names and e["default"] in names
    r = client.post("/api/reconstruct", json={"sketch": SKETCH, "n_candidates": 2, "engine": "deterministic"})
    assert r.status_code == 200 and r.json()["engine"] == "deterministic"
    bad = client.post("/api/reconstruct", json={"sketch": SKETCH, "engine": "nope"})
    assert bad.status_code == 400


def test_reconstruct_accepts_genre_and_engines_report_genres():
    e = client.get("/api/engines").json()
    model = next((x for x in e["engines"] if x["name"] == "model"), None)
    if model is not None:
        assert isinstance(model["genres"], list)
    r = client.post("/api/reconstruct", json={"sketch": SKETCH, "n_candidates": 2, "engine": "deterministic", "genre": "metal"})
    assert r.status_code == 200


def test_tropes_endpoint_and_candidate_labels():
    r = client.get("/api/tropes", params={"genre": "rock"}).json()
    assert "available" in r
    sketch = dict(SKETCH)
    sketch["observations"] = [{"t": i * 0.25, "y": y, "confidence": 0.9} for i, y in enumerate([0, 0, 0, 0, 6, 0, 3, 0])]
    c = client.post("/api/reconstruct", json={"sketch": sketch, "n_candidates": 2, "engine": "deterministic"}).json()
    cand = c["candidates"][0]
    assert "tropes" in cand and "familiarity" in cand
    assert all({"kind", "label", "start", "end"} <= set(t) for t in cand["tropes"])
