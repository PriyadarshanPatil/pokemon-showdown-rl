"""The ladder agent: observation service + policy + choice strings.

Mirrors psrl/train/eval_policy.py so a bug shows up in both places rather than only on
the ladder. Nothing here parses protocol or touches the network.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from psrl.nets.model import BattlePolicy, load_policy
from psrl.nets import layout as L

REPO = Path(__file__).resolve().parents[2]
SERVICE = REPO / "node" / "src" / "encode_service.js"


class EncodeService:
    """Thin client for node/src/encode_service.js (newline-delimited JSON)."""

    def __init__(self, ps_dir: str | None = None):
        env = dict(os.environ)
        if ps_dir:
            env["PS_DIR"] = ps_dir
        self.proc = subprocess.Popen(
            ["node", str(SERVICE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
        )
        meta = self._call({"op": "ping"})
        L.assert_matches(meta["n_ids"], meta["n_scalars"], meta["n_actions"])
        # action index -> choice string, served by actions.js so the two cannot drift
        self.choices: list[str] = self._call({"op": "actions"})["choices"]

    def _call(self, cmd: dict) -> dict:
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()[-2000:]
            raise RuntimeError(f"encode_service died:\n{err}")
        res = json.loads(line)
        if not res.get("ok"):
            raise RuntimeError(f"encode_service error: {res.get('error')} (cmd={cmd.get('op')})")
        return res

    def init_room(self, room: str, perspective: str) -> None:
        self._call({"op": "init", "room": room, "perspective": perspective})

    def feed(self, room: str, lines: list[str]) -> None:
        if lines:
            self._call({"op": "lines", "room": room, "lines": lines})

    def encode(self, room: str, request: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self._call({"op": "encode", "room": room, "request": request})
        return (np.asarray(r["ids"], dtype=np.int64),
                np.asarray(r["scalars"], dtype=np.float32),
                np.asarray(r["mask"], dtype=bool))

    def drop(self, room: str) -> None:
        self._call({"op": "drop", "room": room})

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class LadderAgent:
    """Wraps a trained policy. One hidden state per room, since the core is recurrent."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", greedy: bool = True,
                 switch_tau: float | None = None):
        self.dev = torch.device(device)
        self.model = load_policy(checkpoint, self.dev)
        self.greedy = greedy
        self.switch_tau = switch_tau
        self.hidden: dict[str, torch.Tensor] = {}

    def reset(self, room: str) -> None:
        self.hidden.pop(room, None)

    @torch.no_grad()
    def act(self, room: str, ids: np.ndarray, scalars: np.ndarray, mask: np.ndarray) -> int:
        if not mask.any():
            raise ValueError("no legal action")
        # (batch=1, time=1, features) - the model is recurrent, so time is explicit
        as_t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.dev).view(1, 1, -1)
        logits, _, h = self.model(
            as_t(ids, torch.long), as_t(scalars, torch.float32), as_t(mask, torch.bool),
            self.hidden.get(room))
        self.hidden[room] = h
        logits = logits.view(-1)
        if self.greedy:
            return int(BattlePolicy.choose(logits, self.switch_tau))
        return int(torch.distributions.Categorical(logits=logits).sample())
