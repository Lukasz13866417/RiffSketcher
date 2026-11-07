"""Song genres from the artist, looked up on MusicBrainz.

The model's genre token used to come from the data source alone (every Lakh
song was "rock"), so this module asks MusicBrainz for each artist's community
tags, folds them into the small vocabulary in :data:`ml.datasets.GENRES` and
writes the result to ``song.genre``.  Lookups are cached in
``data/artist_genres.json`` (committed, a few hundred KB), so the API is hit
once per artist, at the one-request-per-second rate it asks for.

    uv run python -m ml.genres lookup     # fill the cache (about 40 minutes for 2,200 artists)
    uv run python -m ml.genres apply      # song.genre from the cache; POP909 songs are "pop"
    uv run python -m ml.genres stats      # how the songs are now labelled

Tags map to genres by keyword vote: each tag's count goes to every genre whose
keyword it contains, and the genre with most votes wins.  A tag like "pop punk"
therefore votes for both pop and punk, and "death metal" only for metal.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ml.datasets import GENRES

CACHE_PATH = Path("data/artist_genres.json")
EXTRACTOR = ("musicbrainz_genre", "1")
API = "https://musicbrainz.org/ws/2/artist/"
USER_AGENT = "RiffSketcher/0.1 (research project; melody reconstruction)"
MIN_SCORE = 90  # MusicBrainz search score below which the match is not trusted

# keyword -> genre; checked as substrings of the lower-cased tag, every match votes
GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "metal": ("metal", "grindcore", "deathcore", "metalcore", "thrash", "doom"),
    "punk": ("punk", "hardcore", "emo", "ska", "oi!"),
    "hiphop": ("hip hop", "hip-hop", "rap", "trap", "grime"),
    "electronic": ("electronic", "electro", "techno", "house", "trance", "edm", "eurodance", "synth",
                   "drum and bass", "dubstep", "ambient", "idm", "breakbeat", "dance"),
    "jazz": ("jazz", "bebop", "bossa", "big band", "swing"),
    "blues": ("blues",),
    "country": ("country", "bluegrass", "americana"),
    "folk": ("folk", "singer-songwriter", "singer/songwriter", "acoustic", "celtic"),
    "classical": ("classical", "baroque", "opera", "orchestral", "composer", "film score", "soundtrack",
                  "symphon", "chamber"),
    "soul": ("soul", "r&b", "rnb", "rhythm and blues", "funk", "motown", "disco", "gospel"),
    "reggae": ("reggae", "dancehall", "dub"),
    "rock": ("rock", "grunge", "alternative", "indie", "new wave", "psychedelic", "progressive", "britpop",
             "shoegaze", "post-"),
    "pop": ("pop", "schlager", "boy band", "girl group", "idol"),
}
assert set(GENRE_KEYWORDS) <= set(GENRES)


def genre_from_tags(tags: list[tuple[str, int]]) -> str | None:
    """Fold MusicBrainz tags ``(name, count)`` into one of :data:`GENRES`, or None."""
    votes: Counter[str] = Counter()
    for name, count in tags:
        low = name.lower()
        for genre, keys in GENRE_KEYWORDS.items():
            if any(k in low for k in keys):
                votes[genre] += max(int(count), 0)  # a negative count is a downvoted tag
    if not votes:
        return None
    best = max(votes.values())
    # ties go to the more specific genre: the order of GENRE_KEYWORDS puts rock and pop last
    for genre in GENRE_KEYWORDS:
        if votes.get(genre) == best:
            return genre
    return None


def _lucene_escape(name: str) -> str:
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", name)


class MusicBrainz:
    """Minimal client: one request per second, retries on 503."""

    def __init__(self, min_interval: float = 1.05) -> None:
        import httpx

        self.client = httpx.Client(headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, timeout=30)
        self.min_interval = min_interval
        self._last = 0.0

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(5):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.client.get(url, params=params)
            if r.status_code == 503 or r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"MusicBrainz kept answering {r.status_code} for {params}")

    def search_artist(self, name: str) -> dict[str, Any] | None:
        data = self._get(API, {"query": f'artist:"{_lucene_escape(name)}"', "fmt": "json", "limit": 3})
        artists = data.get("artists") or []
        if not artists or int(artists[0].get("score", 0)) < MIN_SCORE:
            return None
        return artists[0]

    def artist_tags(self, mbid: str) -> list[tuple[str, int]]:
        data = self._get(API + mbid, {"inc": "tags+genres", "fmt": "json"})
        tags = [(t["name"], int(t.get("count", 1))) for t in data.get("tags") or []]
        tags += [(g["name"], int(g.get("count", 1))) for g in data.get("genres") or []]
        return tags


def lookup_artist(mb: MusicBrainz, name: str) -> dict[str, Any]:
    """One cache entry for an artist name: the match, its tags and the folded genre."""
    hit = mb.search_artist(name)
    if hit is None:
        return {"query": name, "mb_id": None, "mb_name": None, "score": None, "tags": [], "genre": None}
    tags = [(t["name"], int(t.get("count", 1))) for t in hit.get("tags") or []]
    if not tags:
        tags = mb.artist_tags(hit["id"])
    return {"query": name, "mb_id": hit["id"], "mb_name": hit.get("name"), "score": int(hit.get("score", 0)),
            "tags": tags, "genre": genre_from_tags(tags)}


def load_cache(path: Path = CACHE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_cache(cache: dict[str, Any], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=0, ensure_ascii=False, sort_keys=True))


def artists_in_db(con: sqlite3.Connection) -> dict[str, str]:
    """artist_norm -> a display name to query with (the most common spelling)."""
    names: dict[str, Counter] = {}
    for r in con.execute("SELECT artist_norm, artist FROM song WHERE artist_norm IS NOT NULL AND artist IS NOT NULL"):
        names.setdefault(r["artist_norm"], Counter())[r["artist"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in names.items()}


def lookup(con: sqlite3.Connection, cache_path: Path = CACHE_PATH, limit: int | None = None, log=print) -> dict:
    cache = load_cache(cache_path)
    todo = [(k, v) for k, v in sorted(artists_in_db(con).items()) if k not in cache]
    if limit:
        todo = todo[:limit]
    log(f"{len(cache)} artists cached, {len(todo)} to look up")
    mb = MusicBrainz()
    for i, (key, name) in enumerate(todo, 1):
        try:
            cache[key] = lookup_artist(mb, name)
        except Exception as e:  # keep going; the artist is retried next run
            log(f"  {name!r}: {e}")
            continue
        if i % 50 == 0:
            save_cache(cache, cache_path)
            log(f"  {i}/{len(todo)}")
    save_cache(cache, cache_path)
    return cache


def apply(con: sqlite3.Connection, cache: dict[str, Any] | None = None, log=print) -> dict[str, int]:
    """``song.genre`` from the artist cache; songs of a melody source without an artist
    (POP909) are labelled by the source. Provenance goes to ``song_feature``."""
    from ml.db import extractor_id

    cache = load_cache() if cache is None else cache
    ext = extractor_id(con, EXTRACTOR[0], EXTRACTOR[1], {"min_score": MIN_SCORE})
    counts: Counter[str] = Counter()
    for r in con.execute("SELECT id, artist_norm, genre FROM song").fetchall():
        entry = cache.get(r["artist_norm"]) if r["artist_norm"] else None
        genre = entry.get("genre") if entry else None
        if genre is None and r["artist_norm"] is None:
            src = con.execute("SELECT s.name FROM rendition x JOIN source s ON s.id=x.source_id WHERE x.song_id=? LIMIT 1",
                              (r["id"],)).fetchone()
            if src and src["name"] == "pop909":
                genre = "pop"
        if genre is None and r["genre"] is not None:
            genre = r["genre"]  # a genre the source itself gave (GuitarSet styles) stays
        con.execute("UPDATE song SET genre=? WHERE id=?", (genre, r["id"]))
        if entry and entry.get("mb_id"):
            con.execute("INSERT OR REPLACE INTO song_feature(song_id, extractor_id, key, value_num, value_text) VALUES (?,?,?,?,?)",
                        (r["id"], ext, "mb_artist", None, entry["mb_id"]))
            con.execute("INSERT OR REPLACE INTO song_feature(song_id, extractor_id, key, value_num, value_text) VALUES (?,?,?,?,?)",
                        (r["id"], ext, "mb_tags", None, json.dumps(entry["tags"][:10], ensure_ascii=False)))
        counts[genre or "unknown"] += 1
    con.commit()
    log("songs by genre: " + ", ".join(f"{g} {n}" for g, n in counts.most_common()))
    return dict(counts)


def main(argv: list[str] | None = None) -> None:
    from ml.db import connect

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/riffs.db")
    ap.add_argument("--cache", type=Path, default=CACHE_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lookup", help="query MusicBrainz for every artist not yet cached")
    p.add_argument("--limit", type=int, default=None)
    sub.add_parser("apply", help="write song.genre from the cache")
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    con = connect(args.db)
    if args.cmd == "lookup":
        lookup(con, args.cache, args.limit)
    elif args.cmd == "apply":
        apply(con, load_cache(args.cache))
    elif args.cmd == "stats":
        for r in con.execute("SELECT COALESCE(genre, 'unknown') g, COUNT(*) n FROM song GROUP BY g ORDER BY n DESC"):
            print(f"{r['g']:12s} {r['n']:6d}")


if __name__ == "__main__":
    main()
