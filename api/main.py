"""FastAPI backend for Riff Sketcher.

Run with::

    uv run uvicorn api.main:app --reload

The frontend is served statically from ``frontend/`` at ``/``.
All music logic lives in :mod:`core`; this module only maps JSON to it.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.constraints import build_constraints, violations
from core.inference import DeterministicReconstructor, InferenceConfig
from core.motifs import GESTURE_LIBRARY, catalogue_summary, label_tropes, suggest_gestures
from core.representation import Riff, Sketch, midi_to_name
from core.rhythm import fill_missed_taps, merge_passes, quantise
from core.synthesis import riff_to_wav

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Riff Sketcher", version="0.1.0")

# Reconstruction engines. "deterministic" always exists; "model" is added when a
# trained checkpoint (env RIFF_MODEL, default runs/mixed/best.pt) can be loaded.
ENGINES: dict[str, Any] = {"deterministic": DeterministicReconstructor(InferenceConfig())}
try:
    from ml.reconstructor import load_default as _load_model

    _model_engine = _load_model()
    if _model_engine is not None:
        ENGINES["model"] = _model_engine
except Exception:  # ml extras not installed
    _model_engine = None
DEFAULT_ENGINE = "model" if "model" in ENGINES else "deterministic"


# --------------------------------------------------------------------------- #
# request / response models
# --------------------------------------------------------------------------- #


class SketchIn(BaseModel):
    observations: list[dict[str, Any]] = Field(default_factory=list)
    gestures: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    tempo_hint: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_sketch(self) -> Sketch:
        return Sketch.from_dict(self.model_dump())


class ReconstructIn(BaseModel):
    sketch: SketchIn
    n_candidates: int = 5
    engine: str = "auto"  # "auto" | "model" | "deterministic"
    genre: str | None = None  # overrides sketch.meta["genre"]; see ml.datasets.GENRES


class RiffIn(BaseModel):
    notes: list[dict[str, Any]]
    tempo: float = 120.0
    key: str | None = None
    score: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_riff(self) -> Riff:
        return Riff.from_dict(self.model_dump())


class MergeIn(BaseModel):
    passes: list[list[float]]
    tolerance_s: float = 0.08


class QuantiseIn(BaseModel):
    onsets: list[float]
    tempo_hint: float | None = None


def _riff_out(r: Riff, genre: str | None = None) -> dict[str, Any]:
    d = r.to_dict()
    d["note_names"] = [midi_to_name(n.pitch) for n in r.notes]
    d["intervals"] = r.intervals
    gaps = [b.onset - a.onset for a, b in zip(r.notes, r.notes[1:])]
    d.update(label_tropes(r.pitches, gaps, genre))
    return d


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/engines")
def engines() -> dict[str, Any]:
    info = []
    for name, eng in ENGINES.items():
        d: dict[str, Any] = {"name": name, "default": name == DEFAULT_ENGINE}
        if name == "model":
            d["checkpoint"] = eng.checkpoint
            d["device"] = eng.device
            d["trained_on"] = eng.meta.get("sim")
            d["genres"] = eng.genres
        info.append(d)
    return {"engines": info, "default": DEFAULT_ENGINE}


@app.get("/api/tropes")
def tropes(genre: str | None = None) -> dict[str, Any]:
    """Mined trope catalogue for a genre ('all' when unknown); see ml/tropes.py."""
    return catalogue_summary(genre)


@app.get("/api/gestures")
def gestures() -> dict[str, Any]:
    return GESTURE_LIBRARY


@app.post("/api/merge-passes")
def merge(body: MergeIn) -> dict[str, Any]:
    return {"onsets": merge_passes(body.passes, body.tolerance_s)}


@app.post("/api/quantise")
def quantise_endpoint(body: QuantiseIn) -> dict[str, Any]:
    r = quantise(body.onsets, tempo_hint=body.tempo_hint)
    return {"tempo": r.tempo, "onsets_beats": r.onsets_beats,
            "durations_beats": r.durations_beats, "grid": r.grid, "error_s": r.error}


@app.post("/api/fill-gaps")
def fill_gaps(body: SketchIn) -> dict[str, Any]:
    """Insert probable missed taps inside regions tagged as tapping / fast runs."""
    sketch, n = fill_missed_taps(body.to_sketch())
    return {"sketch": sketch.to_dict(), "inserted": n}


@app.post("/api/suggest-gestures")
def suggest(body: SketchIn) -> dict[str, Any]:
    tags = suggest_gestures(body.to_sketch())
    return {"gestures": [t.to_dict() for t in tags]}


@app.post("/api/reconstruct")
def reconstruct(body: ReconstructIn) -> dict[str, Any]:
    sketch = body.sketch.to_sketch()
    if len(sketch) == 0:
        raise HTTPException(400, "sketch has no observations")
    if body.genre is not None:
        sketch.meta["genre"] = body.genre
    n = max(1, min(body.n_candidates, 20))
    name = DEFAULT_ENGINE if body.engine == "auto" else body.engine
    if name not in ENGINES:
        raise HTTPException(400, f"unknown engine {name!r}; available: {sorted(ENGINES)}")
    riffs = ENGINES[name].reconstruct(sketch, n)
    cs = build_constraints(sketch)
    genre = sketch.meta.get("genre")
    return {
        "engine": name,
        "candidates": [{**_riff_out(r, genre), "violations": violations(r, cs)} for r in riffs],
        "constraints": {
            "fixed_pitch": cs.fixed_pitch, "direction": cs.direction,
            "interval": cs.interval, "same": cs.same,
        },
    }


@app.post("/api/synthesize")
def synthesize(body: RiffIn) -> Response:
    riff = body.to_riff()
    if not riff.notes:
        raise HTTPException(400, "riff has no notes")
    return Response(content=riff_to_wav(riff), media_type="audio/wav")


@app.post("/api/synthesize-b64")
def synthesize_b64(body: RiffIn) -> dict[str, str]:
    riff = body.to_riff()
    if not riff.notes:
        raise HTTPException(400, "riff has no notes")
    return {"wav_base64": base64.b64encode(riff_to_wav(riff)).decode()}


# --------------------------------------------------------------------------- #
# static frontend
# --------------------------------------------------------------------------- #

if FRONTEND_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
