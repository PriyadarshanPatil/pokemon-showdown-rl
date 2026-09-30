#!/usr/bin/env python3
"""Materialise encoded BC sequences to disk, so training stops re-simulating replays.

`node/src/replay_worker.js` rebuilds every replay from its inputlog and re-encodes it on every
epoch. That was the right call at 964k replays - the docstring's 120 GB estimate is real - but it
costs about 88% of training wall clock, and measurement puts the data pipeline at 19% of what the
4.1M-parameter model could consume. With a rating filter the corpus is small enough to materialise:
161k replays is roughly 320k sequences.

The cache stores exactly what the worker emits, field for field, so it is not a reimplementation
of the sampling and cannot disagree with it about chunking, padding or discounted returns.

**Drift is the hazard this design has to answer for**, since replay_worker.js lists "the data can
never drift from the current encoder" as a reason not to cache. So the cache carries a fingerprint
over the encoder sources, the id tables, the layout and the filter; `cached_loader.py` refuses to
load a cache whose fingerprint does not match the live tree.

    uv run python tools/build_cache.py --min-rating 1800 --loaders 8
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import shutil
import time
from pathlib import Path


from psrl.data.cache_meta import CACHE_ROOT, FIELDS, fingerprint
from psrl.data.replay_loader import ReplayLoader


def build_split(split: str, out: Path, n_loaders: int, seqlen: int, min_rating: int,
                min_time: int) -> int:
    """Pull one full sweep over `split` through n_loaders shards; returns sequences written."""
    out.mkdir(parents=True, exist_ok=True)
    handles = {k: open(out / f"{k}.dat", "wb") for k in FIELDS}  # noqa: SIM115
    loaders = [ReplayLoader(batch=16, seqlen=seqlen, split=split, min_rating=min_rating,
                            min_time=min_time, shard=i, nshards=n_loaders, one_pass=True)
               for i in range(n_loaders)]
    alive = [True] * n_loaders
    for ldr in loaders:
        ldr._request()

    n_seq = 0
    t0 = time.time()
    try:
        while any(alive):
            for i, ldr in enumerate(loaders):
                if not alive[i]:
                    continue
                b = ldr._collect()
                if b is None:
                    alive[i] = False
                    continue
                ldr._request()
                # write back in the worker's own dtypes; astype here would be a silent lossy step
                handles["ids"].write(b["ids"].astype("<i4").tobytes())
                handles["scalars"].write(b["scalars"].astype("<f4").tobytes())
                handles["mask"].write(b["mask"].astype("u1").tobytes())
                handles["action"].write(b["action"].astype("u1").tobytes())
                handles["ret"].write(b["ret"].astype("<f4").tobytes())
                handles["valid"].write(b["valid"].astype("u1").tobytes())
                n_seq += b["ids"].shape[0]
                if n_seq % 5000 < 16:
                    el = time.time() - t0
                    print(f"  {split}: {n_seq:,} sequences  {n_seq*seqlen/max(el,1):,.0f} steps/s  "
                          f"{el/60:.1f} min  ({sum(alive)}/{n_loaders} shards live)", flush=True)
    finally:
        for ldr in loaders:
            ldr.close()
        for f in handles.values():
            f.close()
    return n_seq


def prune(dry_run: bool = True) -> None:
    """Report, or delete, caches the live tree can no longer read.

    cached_loader refuses a stale cache but cannot remove it, so without this every layout change
    strands another ~46 GiB that nothing will ever open again.
    """
    if not CACHE_ROOT.exists():
        print("no cache directory")
        return
    for d in sorted(CACHE_ROOT.iterdir()):
        meta_path = d / "meta.json"
        if not d.is_dir():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files) / 2**30
        # A build in progress has no meta.json yet and must never be deleted out from under
        # itself, so recency is the guard: anything written in the last half hour is live.
        newest = max((f.stat().st_mtime for f in files), default=0)
        if time.time() - newest < 1800:
            print(f"  {d.name}  {size:.1f} GiB  IN PROGRESS (written in the last 30 min) - skipped")
            continue
        if not meta_path.exists():
            state = "INCOMPLETE (no meta.json - an interrupted build)"
            stale = True
        else:
            meta = json.loads(meta_path.read_text())
            live = fingerprint(meta["min_rating"], meta["min_time"], meta["seqlen"],
                               meta["ps_dir"])
            stale = any(live[k] != meta[k] for k in ("sources", "layout"))
            state = "STALE (encoder or layout moved)" if stale else "current"
        print(f"  {d.name}  {size:.1f} GiB  {state}")
        if stale and not dry_run:
            shutil.rmtree(d)
            print(f"    deleted {d}")
    if dry_run:
        print("\n(dry run - pass --delete to remove the stale ones)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true",
                    help="list caches the live tree can no longer read, then exit")
    ap.add_argument("--delete", action="store_true", help="with --prune, actually remove them")
    ap.add_argument("--min-rating", type=int)
    ap.add_argument("--min-time", type=int, default=1751328000)
    ap.add_argument("--seqlen", type=int, default=64)
    ap.add_argument("--loaders", type=int, default=8)
    # Mirrors node/src/ps_dir.js. Only the basename reaches the fingerprint, so a
    # differently-rooted checkout of the same name still reads an existing cache.
    ap.add_argument("--ps-dir",
                    default=os.environ.get("PS_DIR")
                    or str(Path(__file__).resolve().parents[1].parent / "pokemon-showdown"))
    a = ap.parse_args()

    if a.prune:
        prune(dry_run=not a.delete)
        return
    if a.min_rating is None:
        ap.error("--min-rating is required unless --prune")

    fp = fingerprint(a.min_rating, a.min_time, a.seqlen, a.ps_dir)
    key = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    out = CACHE_ROOT / f"r{a.min_rating}-{key}"
    if out.exists():
        raise SystemExit(f"{out} already exists - delete it to rebuild, never overwrite a build")
    out.mkdir(parents=True)
    print(f"cache -> {out}\nfingerprint {fp['sources']}  layout {fp['layout']}")

    counts = {}
    t0 = time.time()
    for split in ("train", "val"):
        counts[split] = build_split(split, out / split, a.loaders, a.seqlen, a.min_rating,
                                    a.min_time)
        print(f"  {split}: {counts[split]:,} sequences")

    meta = {**fp, "counts": counts, "fields": {k: [d, list(s)] for k, (d, s) in FIELDS.items()},
            "built": time.strftime("%F %T"), "minutes": round((time.time() - t0) / 60, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    total = sum(f.stat().st_size for f in out.rglob("*.dat"))
    print(f"\n{sum(counts.values()):,} sequences, {total/2**30:.1f} GiB, "
          f"{meta['minutes']:.1f} min -> {out}")


if __name__ == "__main__":
    main()
