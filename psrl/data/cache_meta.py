"""Fingerprint and field layout shared by the cache builder and the cache reader.

Lives here rather than in tools/ so both sides import the same definition: a cache written
against one notion of "what makes a sequence" and read against another is exactly the drift
node/src/replay_worker.js avoided by never caching at all.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from psrl.nets import layout as L

REPO = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO / "data" / "cache"

# Everything whose change would alter an encoded sequence.
FINGERPRINT_FILES = [
    "node/src/encode.js", "node/src/damage.js", "node/src/tracker.js",
    "node/src/set_posterior.js", "node/src/item_posterior.js", "node/src/actions.js",
    "node/src/replay.js", "node/src/replay_worker.js",
    "node/src/id_tables.json", "node/src/item_table.json",
]
FIELDS = {
    "ids": ("<i4", (L.N_IDS,)),
    "scalars": ("<f4", (L.N_SCALARS,)),
    "mask": ("u1", (L.N_ACTIONS,)),
    "action": ("u1", ()),
    "ret": ("<f4", ()),
    "valid": ("u1", ()),
}


def fingerprint(min_rating: int, min_time: int, seqlen: int, ps_dir: str) -> dict:
    h = hashlib.sha256()
    for rel in FINGERPRINT_FILES:
        h.update(rel.encode())
        h.update((REPO / rel).read_bytes())
    return {
        "sources": h.hexdigest()[:32],
        "layout": {"ids": L.N_IDS, "scalars": L.N_SCALARS, "actions": L.N_ACTIONS},
        "min_rating": min_rating, "min_time": min_time, "seqlen": seqlen,
        "ps_dir": Path(ps_dir).name,
    }
