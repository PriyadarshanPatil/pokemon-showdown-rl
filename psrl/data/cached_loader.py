"""Read materialised BC sequences, in place of re-simulating replays every epoch.

Same `next_batch()` / `close()` surface as ReplayLoader, so bc.py can swap one for the other.
The cache holds the worker's own records field for field, so a batch from here is indistinguishable
from a batch from the live pipeline - it is not a second implementation of the sampling.

Refuses to load a cache whose fingerprint does not match the live tree. That check is the whole
reason caching is acceptable here: `node/src/replay_worker.js` lists "the data can never drift
from the current encoder" as a reason not to cache, and an unchecked cache would hand that
guarantee back in exchange for speed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from psrl.data.cache_meta import fingerprint

REPO = Path(__file__).resolve().parents[2]


def _live_fingerprint(meta: dict) -> dict:
    return fingerprint(meta["min_rating"], meta["min_time"], meta["seqlen"], meta["ps_dir"])


class CachedReplayLoader:
    """Shuffled sequence batches from a cache directory built by tools/build_cache.py."""

    def __init__(self, path: str | Path, batch: int = 16, split: str = "train",
                 seed: int = 0, check: bool = True):
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        if check:
            live = _live_fingerprint(self.meta)
            for k in ("sources", "layout"):
                if live[k] != self.meta[k]:
                    raise RuntimeError(
                        f"cache {self.dir.name} is stale: {k} is {self.meta[k]} but the tree is "
                        f"{live[k]}. Rebuild with tools/build_cache.py --min-rating "
                        f"{self.meta['min_rating']}.")
        self.seqlen = self.meta["seqlen"]
        self.batch = batch
        n = self.meta["counts"][split]
        self.n = n
        d = self.dir / split
        self.arr = {}
        for k, (dtype, shape) in self.meta["fields"].items():
            a = np.memmap(d / f"{k}.dat", dtype=np.dtype(dtype), mode="r")
            self.arr[k] = a.reshape(n, self.seqlen, *shape) if shape else a.reshape(n, self.seqlen)
        self.rng = np.random.default_rng(seed)
        self._order = self.rng.permutation(n)
        self._i = 0

    def next_batch(self) -> dict | None:
        if self._i + self.batch > self.n:      # reshuffle and keep going, as the worker wraps
            self._order = self.rng.permutation(self.n)
            self._i = 0
        idx = np.sort(self._order[self._i:self._i + self.batch])   # sorted: kinder to the page cache
        self._i += self.batch
        a = self.arr
        return {
            "ids": a["ids"][idx].astype(np.int64),
            "scalars": np.ascontiguousarray(a["scalars"][idx]),
            "mask": a["mask"][idx].astype(bool),
            "action": a["action"][idx].astype(np.int64),
            "ret": np.ascontiguousarray(a["ret"][idx]),
            "valid": a["valid"][idx].astype(bool),
        }

    def close(self) -> None:
        self.arr.clear()
