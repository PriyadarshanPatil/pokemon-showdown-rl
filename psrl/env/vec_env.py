"""Vectorized env: several Node workers stepped in lockstep."""
from __future__ import annotations

import numpy as np

from .bridge import WorkerBridge


class VecBattleEnv:
    def __init__(self, n_workers: int = 8, batch: int = 32,
                 fmt: str = "gen9randombattle", seed: str = "psrl", turn_cap: int = 300,
                 shaping: float = 0.0, no_posterior: bool = False):
        self.workers = [
            WorkerBridge(batch=batch, fmt=fmt, seed=f"{seed}-w{i}", turn_cap=turn_cap,
                         shaping=shaping, no_posterior=no_posterior)
            for i in range(n_workers)
        ]
        self.n_workers, self.batch = n_workers, batch
        obs = [w.observe() for w in self.workers]
        self.slots_per_worker = self.workers[0].n_slots
        self.n_slots = self.slots_per_worker * n_workers
        self._last = self._concat(obs)

    @staticmethod
    def _concat(frames: list[dict]) -> dict:
        return {k: np.concatenate([f[k] for f in frames], axis=0) for k in frames[0]}

    def observe(self) -> dict:
        return self._last

    def step(self, actions: np.ndarray) -> dict:
        a = np.asarray(actions, dtype=np.int32).reshape(self.n_workers, self.slots_per_worker)
        # dispatch to every worker before blocking on any: keeps them concurrent
        for w, chunk in zip(self.workers, a):
            w.send_actions(chunk)
        self._last = self._concat([w.recv() for w in self.workers])
        return self._last

    def close(self) -> None:
        for w in self.workers:
            w.close()
