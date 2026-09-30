#!/usr/bin/env python3
"""Phase 6 gate 3.2 — oracle one-ply look-ahead vs greedy, on paired seeds.

The gate: continue with search only if the oracle is >= 4 points better than greedy against
`heuristic`. Both arms play the SAME seeds, so the same teams and the same battle RNG - the
comparison is paired and the difference is the statistic that matters.

**The oracle cheats, deliberately.** node/src/search_service.js forks the live battle, which
carries the foe's true set and the live PRNG, so each candidate action is scored on what WOULD
actually happen, damage roll included. No deployable agent can do that. The point is an upper
bound: if perfect one-step foresight is not worth 4 points, no honest search will be, and step 3.3
(11-15 days) should not start.

Greedy is the policy head's argmax. The oracle picks the successor the VALUE head scores highest,
with a terminal successor scored by its actual result. Both carry GRU hidden state forward.

    uv run python tools/gate_32_oracle.py --checkpoint runs/bc-v10-pad476/bc.pt --battles 400
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from psrl.nets import layout as L
from psrl.nets.model import BattlePolicy, load_policy

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "node" / "src" / "search_service.js"
MAX_STEPS = 1000


class SearchService:
    """Newline-JSON client for node/src/search_service.js."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["node", str(SERVICE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        meta = self.call({"op": "ping"})
        L.assert_matches(meta["n_ids"], meta["n_scalars"], meta["n_actions"])

    def call(self, cmd: dict) -> dict:
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"search_service died:\n{self.proc.stderr.read()[-2000:]}")
        res = json.loads(line)
        if not res.get("ok"):
            raise RuntimeError(f"search_service error: {res.get('error')} (op={cmd.get('op')})")
        return res

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def commit(svc, game: str, action: int | None, foe=..., ) -> dict:
    """Commit one action, re-picking if the sim reports the choice unavailable.

    Trapping is only discovered by attempting it (SIM-PROTOCOL.md), so the service hands back a
    live mask instead of a state. Switches are what a trap removes, so the first remaining legal
    index is a move, which is the right fallback.
    """
    cmd: dict = {"op": "commit", "game": game}
    if action is not None:
        cmd["action"] = action
    if foe is not ...:
        cmd["foeAction"] = foe
    st = svc.call(cmd)
    tried = {action}
    while st.get("retry"):
        legal = [i for i, m in enumerate(st["mask"]) if m and i not in tried]
        if not legal:
            break
        action = legal[0]
        tried.add(action)
        st = svc.call({"op": "commit", "game": game, "action": action})
    return st


def _tensors(obs: dict, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.tensor(obs["ids"], dtype=torch.long, device=dev).view(1, 1, -1)
    scal = torch.tensor(obs["scalars"], dtype=torch.float32, device=dev).view(1, 1, -1)
    mask = torch.tensor(obs["mask"], dtype=torch.bool, device=dev).view(1, 1, -1)
    mask = mask | (~mask.any(-1, keepdim=True))
    return ids, scal, mask


def play_greedy(svc, model, dev, game: str, ep: int, opponent: str) -> int:
    st = svc.call({"op": "new", "game": game, "ep": ep, "opponent": opponent})
    hidden = None
    for _ in range(MAX_STEPS):
        if st["ended"]:
            break
        obs = st.get("obs")
        if obs is None:
            st = commit(svc, game, None)
            continue
        ids, scal, mask = _tensors(obs, dev)
        with torch.no_grad():
            logits, _v, hidden = model(ids, scal, mask, hidden)
        a = int(BattlePolicy.choose(logits.squeeze(1), None).item())
        st = commit(svc, game, a)
    svc.call({"op": "drop", "game": game})
    return st.get("result") or 0


def play_oracle(svc, model, dev, game: str, ep: int, opponent: str) -> tuple[int, int, int]:
    """Returns (result, decisions, times the oracle disagreed with greedy)."""
    st = svc.call({"op": "new", "game": game, "ep": ep, "opponent": opponent})
    hidden = None
    decisions = disagreements = 0
    for _ in range(MAX_STEPS):
        if st["ended"]:
            break
        obs = st.get("obs")
        if obs is None:
            st = commit(svc, game, None)
            continue

        # advance the recurrent state on the position we actually face
        ids, scal, mask = _tensors(obs, dev)
        with torch.no_grad():
            logits, _v, h_next = model(ids, scal, mask, hidden)
        greedy_a = int(BattlePolicy.choose(logits.squeeze(1), None).item())

        exp = svc.call({"op": "expand", "game": game})
        succ = [s for s in exp["successors"] if s["ok"]]
        scored: list[tuple[float, int]] = []
        batch = [s for s in succ if "ids" in s]
        if batch:
            b_ids = torch.tensor(np.array([s["ids"] for s in batch]), dtype=torch.long,
                                 device=dev).unsqueeze(1)
            b_scal = torch.tensor(np.array([s["scalars"] for s in batch]), dtype=torch.float32,
                                  device=dev).unsqueeze(1)
            b_mask = torch.tensor(np.array([s["mask"] for s in batch]), dtype=torch.bool,
                                  device=dev).unsqueeze(1)
            b_mask = b_mask | (~b_mask.any(-1, keepdim=True))
            # h_next, not hidden: the successor follows the position we just consumed
            h = h_next.expand(-1, len(batch), -1).contiguous()
            with torch.no_grad():
                _lg, vals, _h = model(b_ids, b_scal, b_mask, h)
            v = vals.view(len(batch), -1)[:, -1].tolist()
            scored += [(float(x), s["action"]) for x, s in zip(v, batch)]
        # A finished successor is scored by what actually happened. One that is still live but
        # carries no request could not be advanced to our next decision, so it has no value we
        # can trust - drop it rather than score it as a draw.
        scored += [(float(s["result"] or 0), s["action"])
                   for s in succ if "ids" not in s and s["ended"]]

        if scored:
            a = max(scored)[1]
        else:
            a = greedy_a
        decisions += 1
        if a != greedy_a:
            disagreements += 1
        hidden = h_next
        st = commit(svc, game, a, foe=exp["foeAction"])
    svc.call({"op": "drop", "game": game})
    return st.get("result") or 0, decisions, disagreements


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--battles", type=int, default=400)
    ap.add_argument("--opponent", default="heuristic")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed-tag", default="g32")
    a = ap.parse_args()

    dev = torch.device(a.device)
    model = load_policy(a.checkpoint, dev)

    svc = SearchService()
    g_wins = o_wins = 0
    both = 0
    dec_tot = dis_tot = 0
    paired = []
    try:
        for ep in range(a.battles):
            gr = play_greedy(svc, model, dev, f"g{ep}", ep, a.opponent)
            orr, dec, dis = play_oracle(svc, model, dev, f"o{ep}", ep, a.opponent)
            g_wins += gr == 1
            o_wins += orr == 1
            dec_tot += dec
            dis_tot += dis
            paired.append((gr == 1, orr == 1))
            both += 1
            if (ep + 1) % 50 == 0:
                print(f"  {ep+1}/{a.battles}: greedy {100*g_wins/both:.1f}%  "
                      f"oracle {100*o_wins/both:.1f}%  (disagree {100*dis_tot/max(1,dec_tot):.1f}% "
                      f"of {dec_tot} decisions)", flush=True)
    finally:
        svc.close()

    n = both
    gp, op_ = 100 * g_wins / n, 100 * o_wins / n
    # paired difference: McNemar-style, only discordant pairs carry information
    b = sum(1 for g, o in paired if g and not o)
    c = sum(1 for g, o in paired if o and not g)
    diff = op_ - gp
    se = 100 * math.sqrt(b + c) / n if (b + c) else 0.0
    print(f"\nvs {a.opponent}, {n} paired seeds, checkpoint {a.checkpoint}")
    print(f"  greedy {gp:.1f}% {list(map(lambda x: round(x,1), wilson(g_wins, n)))}  ({g_wins}/{n})")
    print(f"  oracle {op_:.1f}% {list(map(lambda x: round(x,1), wilson(o_wins, n)))}  ({o_wins}/{n})")
    print(f"  paired difference {diff:+.1f} pts, SE {se:.1f} "
          f"(discordant: oracle-only {c}, greedy-only {b})")
    print(f"  oracle disagreed with greedy on {100*dis_tot/max(1,dec_tot):.1f}% "
          f"of {dec_tot} decisions")
    print(f"\ngate 3.2 needs >= +4.0 points: {'PASS' if diff >= 4.0 else 'FAIL'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
