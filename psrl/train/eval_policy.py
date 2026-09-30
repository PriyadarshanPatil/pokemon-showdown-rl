"""Evaluate a torch policy against an in-worker scripted baseline."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from psrl.env.bridge import WorkerBridge
from psrl.nets import layout as L
from psrl.nets.model import BattlePolicy, load_policy
from psrl.train.run_meta import REPO, write_run_meta

# Active-Pokemon flags inside the scalar block (encode.js:100 own side, :140 foe side).
_OWN_ACTIVE = [i * L.SCAL_PER_MON + 3 for i in range(L.N_MON)]
_FOE_ACTIVE = [(L.N_MON + j) * L.SCAL_PER_MON + 3 for j in range(L.N_MON)]


def play_stats(ids, scal, mask, need, acts, prev, c):
    """Count one step of p1 decisions into `c`: free decisions, voluntary switches, and
    back-to-back moves with the same own and foe Pokemon (`pairs`) that reuse the move
    (`repeats`). `prev` holds each row's previous decision."""
    rows = np.arange(len(acts))
    own = scal[:, _OWN_ACTIVE].argmax(1)
    foe = scal[:, _FOE_ACTIVE]
    own_sp = ids[rows, own * L.IDS_PER_MON]
    foe_sp = np.where(foe.max(1) > 0, ids[rows, (L.N_MON + foe.argmax(1)) * L.IDS_PER_MON], -1)
    free = need & mask[:, :L.SWITCH_OFFSET].any(1)   # 0-3 moves, 4-7 move+tera, 8-13 switches
    move = free & (acts >= 0) & (acts < L.SWITCH_OFFSET)
    move_id = ids[rows, own * L.IDS_PER_MON + 5
                  + np.where(acts >= L.TERA_OFFSET, acts - L.TERA_OFFSET, acts)
                  .clip(0, L.TERA_OFFSET - 1)]
    pair = move & prev["move"] & (own_sp == prev["own"]) & (foe_sp == prev["foe"])
    c["free"] += int(free.sum())
    c["switches"] += int((free & (acts >= L.SWITCH_OFFSET)).sum())
    c["pairs"] += int(pair.sum())
    c["repeats"] += int((pair & (move_id == prev["id"])).sum())
    for k, v in (("move", move), ("own", own_sp), ("foe", foe_sp), ("id", move_id)):
        prev[k] = np.where(need, v, prev[k])


@torch.no_grad()
def evaluate(model, opponent: str, n_battles: int = 200, batch: int = 32,
             device: str = "mps", seed: str = "eval", greedy: bool = True,
             no_posterior: bool = False, switch_tau: float | None = None) -> dict:
    """Returns win rate of `model` (as p1) against `opponent`."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    # must match the feature setting the policy was trained with
    w = WorkerBridge(batch=batch, seed=seed, opponent=opponent, no_posterior=no_posterior)
    f = w.observe()
    n_p1 = w.n_slots // 2
    hidden = None
    wins = losses = ties = 0
    finished = 0
    counts = dict(free=0, switches=0, pairs=0, repeats=0)
    prev = {"move": np.zeros(n_p1, bool), "own": np.full(n_p1, -1), "foe": np.full(n_p1, -2),
            "id": np.full(n_p1, -1)}
    while finished < n_battles:
        ids = torch.from_numpy(f["ids"][0::2].astype(np.int64)).unsqueeze(1).to(dev)
        scal = torch.from_numpy(f["scalars"][0::2].copy()).unsqueeze(1).to(dev)
        mask = torch.from_numpy(f["mask"][0::2].copy()).unsqueeze(1).to(dev)
        # a fully-masked row (slot not acting) would make softmax undefined
        dead = ~mask.any(-1, keepdim=True)
        mask_safe = mask | dead
        logits, _, hidden = model(ids, scal, mask_safe, hidden)
        if greedy:
            a = BattlePolicy.choose(logits, switch_tau)
        else:
            a = torch.distributions.Categorical(logits=logits.squeeze(1)).sample().unsqueeze(1)
        a = a.squeeze(1).squeeze(-1) if a.dim() > 2 else a.squeeze(1)
        acts = np.full(w.n_slots, -1, dtype=np.int32)
        a_np = a.detach().cpu().numpy().reshape(-1)
        need = f["needs"][0::2]
        acts[0::2] = np.where(need, a_np, -1)
        play_stats(f["ids"][0::2], f["scalars"][0::2], f["mask"][0::2], need.astype(bool),
                   acts[0::2], prev, counts)
        f = w.step(acts)
        done = f["done"][0::2]
        if done.any():
            prev["move"][done] = False
            r = f["reward"][0::2][done]
            wins += int((r > 0).sum()); losses += int((r < 0).sum()); ties += int((r == 0).sum())
            finished += int(done.sum())
            # reset hidden state for slots whose battle ended
            if hidden is not None:
                keep = torch.from_numpy(~done).to(dev).view(1, -1, 1)
                hidden = hidden * keep
    w.close()
    n = wins + losses + ties
    return {"opponent": opponent, "n": n, "wins": wins, "losses": losses, "ties": ties,
            "win_rate": (wins + 0.5 * ties) / max(1, n), **counts}


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = load_policy(a.checkpoint, torch.device("cpu"))
    out = REPO / "runs" / a.name
    out.mkdir(parents=True, exist_ok=False)          # never overwrite a run's outputs
    write_run_meta(out, a)
    results = []
    for o in a.opponents.split(","):
        r = evaluate(model, o, n_battles=a.battles, device=a.device, seed=f"{a.battle_seed}-{o}",
                     greedy=not a.sample, no_posterior=a.no_posterior, switch_tau=a.switch_tau)
        ci = 1.96 * np.sqrt(max(r["win_rate"] * (1 - r["win_rate"]), 1e-9) / r["n"])
        print(f"  vs {o:<10} {r['win_rate']*100:5.1f}%  [{(r['win_rate']-ci)*100:.1f}, "
              f"{(r['win_rate']+ci)*100:.1f}]  (n={r['n']})  repeats {r['repeats']}/{r['pairs']}  "
              f"voluntary switches {r['switches']}/{r['free']}", flush=True)
        results.append(r)
    (out / "eval.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--battles", type=int, default=400)
    ap.add_argument("--opponents", default="random,maxdamage,heuristic")
    ap.add_argument("--sample", action="store_true", help="sample actions instead of taking the argmax")
    ap.add_argument("--seed", type=int, default=0, help="torch/numpy seed; only matters with --sample")
    ap.add_argument("--battle-seed", default="final",
                    help="battles use f'{battle_seed}-{opponent}'; 'final' matches ppo.py's final eval")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-posterior", action="store_true")
    ap.add_argument("--switch-tau", type=float, default=None,
                    help="switch when the summed switch probability exceeds this; 0.30 matches "
                         "the human rate. Default is plain argmax, which discards that mass")
    main(ap.parse_args())
