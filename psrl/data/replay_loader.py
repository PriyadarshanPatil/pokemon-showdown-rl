"""Streams behaviour-cloning batches from the replay worker (no materialised dataset)."""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import numpy as np

MAGIC = 0x50535233
REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "node" / "src" / "replay_worker.js"


class ReplayLoader:
    def __init__(self, batch: int = 16, seqlen: int = 64, split: str = "train",
                 min_time: int = 1751328000, ps_dir: str | None = None,
                 shard: int = 0, nshards: int = 1, no_posterior: bool = False,
                 min_rating: int = 0, one_pass: bool = False):
        env = dict(os.environ)
        if ps_dir:
            env["PS_DIR"] = ps_dir
        if no_posterior:
            env["PSRL_NO_POSTERIOR"] = "1"
        self.proc = subprocess.Popen(
            ["node", "--max-old-space-size=2048", str(WORKER),
             f"--batch={batch}", f"--seqlen={seqlen}", f"--split={split}",
             f"--minTime={min_time}", f"--shard={shard}", f"--nshards={nshards}",
             f"--minRating={min_rating}", f"--onePass={1 if one_pass else 0}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        self.batch, self.seqlen = batch, seqlen
        self._dtype = None

    def _read_exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            c = self.proc.stdout.read(n - len(out))
            if not c:
                err = self.proc.stderr.read().decode(errors="replace")[-2000:]
                raise RuntimeError(f"replay worker died:\n{err}")
            out += c
        return bytes(out)

    def _request(self) -> None:
        self.proc.stdin.write(struct.pack("<I", 0))
        self.proc.stdin.flush()

    def next_batch(self) -> dict | None:
        self._request()
        return self._collect()

    def _collect(self) -> dict | None:
        (length,) = struct.unpack("<I", self._read_exact(4))
        buf = self._read_exact(length)
        magic, n, seqlen, n_ids, n_scal, n_act, _ = struct.unpack("<7I", buf[:28])
        if magic != MAGIC:
            raise RuntimeError(f"bad magic {magic:#x}")
        if n == 0:
            return None
        if self._dtype is None:
            self._dtype = np.dtype([
                ("ids", "<i4", (n_ids,)), ("scalars", "<f4", (n_scal,)),
                ("mask", "u1", (n_act,)), ("action", "u1"), ("ret", "<f4"), ("valid", "u1"),
            ])
            expected = n_ids * 4 + n_scal * 4 + n_act + 6
            if self._dtype.itemsize != expected:
                raise RuntimeError(f"dtype {self._dtype.itemsize} != worker {expected}")
        rec = np.frombuffer(buf, dtype=self._dtype, count=n * seqlen, offset=28)
        rs = lambda a, *tail: a.reshape(n, seqlen, *tail)
        return {
            "ids": rs(rec["ids"], n_ids).astype(np.int64),
            "scalars": rs(rec["scalars"], n_scal),
            "mask": rs(rec["mask"], n_act).astype(bool),
            "action": rs(rec["action"]).astype(np.int64),
            "ret": rs(rec["ret"]),
            "valid": rs(rec["valid"]).astype(bool),
        }

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class MultiReplayLoader:
    """Pipelined over disjoint shards.

    A round-robin that blocks on one loader at a time gives no speedup - the workers sit
    idle while Python waits. Instead every shard is asked for a batch up front, so all N
    processes reconstruct concurrently, and each collected batch immediately triggers the
    next request on that shard.
    """

    def __init__(self, n: int = 4, **kw):
        kw.pop("shard", None); kw.pop("nshards", None)
        self.loaders = [ReplayLoader(shard=i, nshards=n, **kw) for i in range(n)]
        self.i = 0
        for l in self.loaders:
            l._request()

    def next_batch(self):
        for _ in range(len(self.loaders)):
            l = self.loaders[self.i]
            self.i = (self.i + 1) % len(self.loaders)
            b = l._collect()
            l._request()               # keep that shard busy while we use this batch
            if b is not None:
                return b
        return None

    def close(self):
        for l in self.loaders:
            l.close()
