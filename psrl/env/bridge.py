"""Client for the Node batched simulator worker (Phase 1 step 8).

Length-prefixed binary frames over pipes. No JSON or protocol text in the hot loop.
Slot layout is [battle0 p1, battle0 p2, battle1 p1, ...].
"""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import numpy as np

MAGIC = 0x50535231
REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "node" / "src" / "worker.js"


class WorkerBridge:
    def __init__(self, batch: int = 32, fmt: str = "gen9randombattle",
                 seed: str = "psrl", turn_cap: int = 300, ps_dir: str | None = None,
                 opponent: str | None = None, shaping: float = 0.0,
                 no_posterior: bool = False):
        """opponent: name of a scripted bot to drive p2 inside the worker (no extra IPC),
        or None for self-play where Python supplies both sides."""
        env = dict(os.environ)
        if ps_dir:
            env["PS_DIR"] = ps_dir
        argv = ["node", str(WORKER), f"--batch={batch}", f"--format={fmt}",
                f"--seed={seed}", f"--turnCap={turn_cap}"]
        if opponent:
            argv.append(f"--opponent={opponent}")
        if shaping:
            argv.append(f"--shaping={shaping}")
        if no_posterior:
            env["PSRL_NO_POSTERIOR"] = "1"
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        self.batch = batch
        self._dtype = None
        self.n_slots = self.n_ids = self.n_scalars = self.n_actions = 0

    # ---- framing ----
    def _send(self, payload: bytes) -> None:
        self.proc.stdin.write(struct.pack("<I", len(payload)))
        if payload:
            self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def _read_exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self.proc.stdout.read(n - len(out))
            if not chunk:
                err = self.proc.stderr.read().decode(errors="replace")[-2000:]
                raise RuntimeError(f"worker died. stderr:\n{err}")
            out += chunk
        return bytes(out)

    def _recv(self) -> dict:
        (length,) = struct.unpack("<I", self._read_exact(4))
        buf = self._read_exact(length)
        magic, n_slots, n_ids, n_scalars, n_actions, _ = struct.unpack("<6I", buf[:24])
        if magic != MAGIC:
            raise RuntimeError(f"bad frame magic {magic:#x}")
        if self._dtype is None:
            self.n_slots, self.n_ids, self.n_scalars, self.n_actions = n_slots, n_ids, n_scalars, n_actions
            # packed (align=False) to match the worker's byte layout exactly
            self._dtype = np.dtype([
                ("ids", "<i4", (n_ids,)),
                ("scalars", "<f4", (n_scalars,)),
                ("mask", "u1", (n_actions,)),
                ("needs", "u1"),
                ("reward", "<f4"),
                ("done", "u1"),
            ])
            expected = n_ids * 4 + n_scalars * 4 + n_actions + 6
            if self._dtype.itemsize != expected:
                raise RuntimeError(f"dtype itemsize {self._dtype.itemsize} != worker {expected}")
        rec = np.frombuffer(buf, dtype=self._dtype, count=n_slots, offset=24)
        return {
            "ids": rec["ids"], "scalars": rec["scalars"], "mask": rec["mask"].astype(bool),
            "needs": rec["needs"].astype(bool), "reward": rec["reward"], "done": rec["done"].astype(bool),
        }

    # ---- api ----
    def observe(self) -> dict:
        """Fetch the current frame without advancing (also used as reset)."""
        self.send_actions(None)
        return self.recv()

    def step(self, actions: np.ndarray) -> dict:
        self.send_actions(actions)
        return self.recv()

    # Split send/recv so a vectorized env can dispatch to every worker before
    # blocking on any of them; they then run concurrently instead of serially.
    def send_actions(self, actions: np.ndarray | None) -> None:
        if actions is None:
            self._send(b"")
            return
        a = np.ascontiguousarray(actions, dtype="<i4")
        if self.n_slots and a.size != self.n_slots:
            raise ValueError(f"expected {self.n_slots} actions, got {a.size}")
        self._send(a.tobytes())

    def recv(self) -> dict:
        return self._recv()

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
