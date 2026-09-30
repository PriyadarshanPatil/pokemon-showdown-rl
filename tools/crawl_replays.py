#!/usr/bin/env python3
"""A1: Pokemon Showdown replay crawler.

Uses the documented `before` cursor (WEB-API.md: `page` is deprecated).
Resumable, rate-limited, on-disk cached. Usernames are salted-hashed at ingest (A2).
"""
import argparse, hashlib, json, os, sqlite3, sys, time, urllib.error, urllib.request, zlib

UA = "psrl-research/0.1 (non-commercial Pokemon RL research; polite crawler)"
BASE = "https://replay.pokemonshowdown.com"
DB = os.path.join(os.path.dirname(__file__), "..", "data", "replays.db")
# Stable pseudonymisation: lets us split train/test by player without storing names.
SALT = b"psrl-v1"


def player_hash(name: str) -> str:
    return hashlib.blake2b(SALT + name.strip().lower().encode(), digest_size=8).hexdigest()


class Limiter:
    def __init__(self, rate): self.min_gap = 1.0 / rate; self.last = 0.0
    def wait(self):
        gap = time.monotonic() - self.last
        if gap < self.min_gap: time.sleep(self.min_gap - gap)
        self.last = time.monotonic()


def fetch(url, limiter, tries=5):
    for attempt in range(tries):
        limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 403) and attempt == 0 and e.code == 404: return None
            if e.code == 404: return None
            wait = 2 ** attempt
            print(f"  HTTP {e.code} on {url} - retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  {type(e).__name__} on {url} - retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return None


def db_open():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript("""
      CREATE TABLE IF NOT EXISTS replays(
        id TEXT PRIMARY KEY, formatid TEXT, uploadtime INTEGER, rating INTEGER,
        p1 TEXT, p2 TEXT, private INTEGER, fetched INTEGER DEFAULT 0, has_inputlog INTEGER);
      CREATE INDEX IF NOT EXISTS ix_rating ON replays(rating);
      CREATE INDEX IF NOT EXISTS ix_time ON replays(uploadtime);
      CREATE INDEX IF NOT EXISTS ix_fetched ON replays(fetched);
      CREATE TABLE IF NOT EXISTS state(k TEXT PRIMARY KEY, v TEXT);
      CREATE TABLE IF NOT EXISTS bodies(id TEXT PRIMARY KEY, z BLOB);
    """)
    return c


def get_state(c, k, default=None):
    r = c.execute("SELECT v FROM state WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def set_state(c, k, v):
    c.execute("INSERT OR REPLACE INTO state(k,v) VALUES(?,?)", (k, str(v)))


def index(fmt, limiter, start_before=None):
    """Walk the `before` cursor backwards through the whole archive."""
    c = db_open()
    key = f"cursor:{fmt}"
    before = start_before or get_state(c, key) or int(time.time())
    before = int(before)
    total = c.execute("SELECT COUNT(*) FROM replays WHERE formatid=?", (fmt,)).fetchone()[0]
    t0 = time.time(); calls = 0
    while True:
        body = fetch(f"{BASE}/search.json?format={fmt}&before={before}", limiter)
        if body is None: print("index: giving up on this page", file=sys.stderr); break
        rows = json.loads(body); calls += 1
        if not rows: break
        new = 0
        for r in rows:
            players = r.get("players") or ["", ""]
            try:
                c.execute("INSERT INTO replays(id,formatid,uploadtime,rating,p1,p2,private) VALUES(?,?,?,?,?,?,?)",
                          (r["id"], r.get("formatid") or fmt, r["uploadtime"], r.get("rating"),
                           player_hash(players[0]), player_hash(players[1] if len(players) > 1 else ""),
                           r.get("private", 0)))
                new += 1
            except sqlite3.IntegrityError:
                pass
        total += new
        oldest = min(x["uploadtime"] for x in rows)
        # Official pagination: use the uploadtime of the last replay. Guard against a
        # stall when >50 replays share one timestamp.
        before = oldest if oldest < before else before - 1
        set_state(c, key, before); c.commit()
        if calls % 20 == 0:
            el = time.time() - t0
            print(f"  {total:,} indexed | cursor {time.strftime('%Y-%m-%d', time.gmtime(before))} "
                  f"| {calls} calls | {calls/el:.1f} req/s", flush=True)
        if len(rows) < 51: print("index: reached end of archive"); break
    c.commit()
    print(f"index done: {total:,} rows for {fmt}")


def sample(fmt, limiter, n, points):
    """Stratified sample of replay ids spread across the archive (for A3)."""
    c = db_open()
    now = int(time.time()); start = 1690000000
    per = max(1, n // points)
    got = 0
    for i in range(points):
        before = int(start + (now - start) * i / (points - 1))
        while True:
            body = fetch(f"{BASE}/search.json?format={fmt}&before={before}", limiter)
            if body is None: break
            rows = json.loads(body)
            if not rows: break
            for r in rows:
                players = r.get("players") or ["", ""]
                try:
                    c.execute("INSERT INTO replays(id,formatid,uploadtime,rating,p1,p2,private) VALUES(?,?,?,?,?,?,?)",
                              (r["id"], r.get("formatid") or fmt, r["uploadtime"], r.get("rating"),
                               player_hash(players[0]), player_hash(players[1] if len(players) > 1 else ""),
                               r.get("private", 0)))
                    got += 1
                except sqlite3.IntegrityError:
                    pass
            before = min(x["uploadtime"] for x in rows) - 1
            if got >= per * (i + 1) or len(rows) < 51: break
        c.commit()
    print(f"sample: {got} new ids across {points} time points")


def fetch_bodies(limiter, where, limit):
    """Fetch replay bodies into SQLite as zlib blobs. Atomic and resumable:
    a body row and its fetched flag are committed together."""
    c = db_open()
    rows = c.execute(f"SELECT id FROM replays WHERE fetched=0 AND {where} ORDER BY uploadtime DESC LIMIT ?",
                     (limit,)).fetchall()
    print(f"fetching {len(rows)} replay bodies...", flush=True)
    ok = 0
    for i, (rid,) in enumerate(rows):
        body = fetch(f"{BASE}/{rid}.json", limiter)
        if body is None:
            c.execute("UPDATE replays SET fetched=-1 WHERE id=?", (rid,)); c.commit(); continue
        j = json.loads(body)
        c.execute("INSERT OR REPLACE INTO bodies(id,z) VALUES(?,?)",
                  (rid, zlib.compress(body, 6)))
        c.execute("UPDATE replays SET fetched=1, has_inputlog=? WHERE id=?",
                  (1 if j.get("inputlog") else 0, rid))
        c.commit()
        ok += 1
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(rows)}", flush=True)
    print(f"fetched {ok}/{len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["index", "sample", "fetch", "stats"])
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--where", default="1=1")
    a = ap.parse_args()
    lim = Limiter(a.rate)
    if a.mode == "index": index(a.format, lim)
    elif a.mode == "sample": sample(a.format, lim, a.n, a.points)
    elif a.mode == "fetch": fetch_bodies(lim, a.where, a.n)
    else:
        c = db_open()
        for q, label in [("SELECT COUNT(*) FROM replays", "indexed"),
                         ("SELECT COUNT(*) FROM replays WHERE fetched=1", "fetched"),
                         ("SELECT COUNT(*) FROM replays WHERE has_inputlog=1", "with inputlog"),
                         ("SELECT COUNT(*) FROM bodies", "bodies stored")]:
            print(f"{label}: {c.execute(q).fetchone()[0]:,}")
