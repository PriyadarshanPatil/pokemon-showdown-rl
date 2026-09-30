"""Head-to-head evaluation of two checkpoints (Phase 7's replacement instrument).

Why this exists. Scripted baselines cannot separate 1%-switching from 13%-switching play - shown
twice, once with `heuristic` and once with a baseline built specifically to expose it - and the
ladder needs about 75 hours per arm to resolve a 5-point difference. Playing the two policies
against each other offline answers the same question with a standard error near 1.1 points at
2,000 battles, in roughly a minute.

Model A drives p1 (even slots) and model B drives p2 (odd slots), the slot scheme league.py
already uses. Both sides are deterministic - greedy, or greedy-within-class when a --switch-tau
is given - so the measurement does not inherit the sampling noise that made gate 1.1 misleading.

**What it measures:** whether A beats B. **What it does not measure:** whether A beats humans.
The human anchor remains the replay corpus, where players switch on 13.49% of free decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from psrl.env.vec_env import VecBattleEnv
from psrl.nets import layout as L
from psrl.nets.model import BattlePolicy, load_policy
from psrl.train.run_meta import write_run_meta

REPO = Path(__file__).resolve().parents[2]


def _load(path: str, dev) -> BattlePolicy:
    return load_policy(path, dev)


def _act(model, f, idx, hidden, dev, tau):
    """One decision for a subset of slots. Returns actions, new hidden, and switch counts."""
    ids = torch.from_numpy(f["ids"][idx].astype(np.int64)).unsqueeze(1).to(dev)
    scal = torch.from_numpy(f["scalars"][idx].copy()).unsqueeze(1).to(dev)
    mask = torch.from_numpy(f["mask"][idx].copy()).unsqueeze(1).to(dev)
    # a fully-masked row would make softmax undefined; those slots are not acting anyway
    mask = mask | (~mask.any(-1, keepdim=True))
    with torch.no_grad():
        logits, _, h2 = model(ids, scal, mask, hidden)
    a = BattlePolicy.choose(logits.squeeze(1), tau)
    # a decision is "free" when attacking was legal, so a switch was a choice not a replacement
    free = mask.squeeze(1)[:, :L.SWITCH_OFFSET].any(-1)
    switched = free & (a >= L.SWITCH_OFFSET)
    return a.cpu().numpy(), h2, int(free.sum()), int(switched.sum())


def head2head(model_a, model_b, n_battles: int = 1000, workers: int = 6, batch: int = 32,
              seed: str = "h2h", tau_a: float | None = None, tau_b: float | None = None,
              device: str = "mps", no_posterior: bool = False) -> dict:
    """Play A (p1) against B (p2) until `n_battles` finish. Win rate is from A's side."""
    dev = torch.device(device)
    env = VecBattleEnv(n_workers=workers, batch=batch, seed=seed, no_posterior=no_posterior)
    a_slots = np.arange(0, env.n_slots, 2)
    b_slots = np.arange(1, env.n_slots, 2)
    n_pairs = env.n_slots // 2
    hidden = {"a": None, "b": None}
    free = {"a": 0, "b": 0}
    switches = {"a": 0, "b": 0}
    wins = losses = ties = finished = 0

    f = env.observe()
    try:
        while finished < n_battles:
            acts = np.full(env.n_slots, -1, dtype=np.int32)
            need = f["needs"]
            for tag, slots, model, tau in (("a", a_slots, model_a, tau_a),
                                           ("b", b_slots, model_b, tau_b)):
                idx = slots[need[slots]]
                if not idx.size:
                    continue
                if hidden[tag] is None:
                    hidden[tag] = torch.zeros(1, n_pairs, model.d_hidden, device=dev)
                sel = idx // 2          # both sides index their own hidden state by battle
                chosen, h2, nfree, nsw = _act(model, f, idx, hidden[tag][:, sel], dev, tau)
                hidden[tag][:, sel] = h2
                acts[idx] = chosen
                free[tag] += nfree
                switches[tag] += nsw

            f = env.step(acts)
            r, d = f["reward"], f["done"]
            for s in np.flatnonzero(d):
                if s % 2:
                    continue            # score once per battle, from A's slot
                finished += 1
                if r[s] > 0:
                    wins += 1
                elif r[s] < 0:
                    losses += 1
                else:
                    ties += 1
            if d.any():
                for tag in ("a", "b"):
                    if hidden[tag] is not None:
                        hidden[tag][:, torch.from_numpy(d[0::2]).to(dev)] = 0.0
    finally:
        env.close()

    n = wins + losses + ties
    wr = (wins + 0.5 * ties) / max(n, 1)
    return {"n": n, "wins": wins, "losses": losses, "ties": ties, "win_rate": wr,
            "a_free": free["a"], "a_switches": switches["a"],
            "b_free": free["b"], "b_switches": switches["b"]}


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    model_a, model_b = _load(a.checkpoint_a, dev), _load(a.checkpoint_b, dev)
    out = REPO / "runs" / a.name
    out.mkdir(parents=True, exist_ok=False)      # never overwrite a run's outputs
    write_run_meta(out, a)

    r = head2head(model_a, model_b, n_battles=a.battles, workers=a.workers, batch=a.batch,
                  seed=a.battle_seed, tau_a=a.switch_tau_a, tau_b=a.switch_tau_b,
                  device=a.device, no_posterior=a.no_posterior)
    ci = 1.96 * np.sqrt(max(r["win_rate"] * (1 - r["win_rate"]), 1e-9) / max(r["n"], 1))
    pct = lambda s, fr: f"{100 * s / max(fr, 1):.2f}%"
    print(f"  A {Path(a.checkpoint_a).parent.name}"
          f"{f' tau={a.switch_tau_a}' if a.switch_tau_a else ' greedy'}"
          f"  vs  B {Path(a.checkpoint_b).parent.name}"
          f"{f' tau={a.switch_tau_b}' if a.switch_tau_b else ' greedy'}")
    print(f"  A win rate {r['win_rate']*100:5.2f}%  [{(r['win_rate']-ci)*100:.2f}, "
          f"{(r['win_rate']+ci)*100:.2f}]  (n={r['n']}, W{r['wins']}/L{r['losses']}/T{r['ties']})")
    print(f"  voluntary switches: A {pct(r['a_switches'], r['a_free'])} of {r['a_free']}, "
          f"B {pct(r['b_switches'], r['b_free'])} of {r['b_free']}", flush=True)
    (out / "h2h.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-a", required=True, help="drives p1, even slots")
    ap.add_argument("--checkpoint-b", required=True, help="drives p2, odd slots")
    ap.add_argument("--name", required=True)
    ap.add_argument("--battles", type=int, default=1000)
    ap.add_argument("--switch-tau-a", type=float, default=None)
    ap.add_argument("--switch-tau-b", type=float, default=None)
    ap.add_argument("--battle-seed", default="h2h")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-posterior", action="store_true")
    main(ap.parse_args())
