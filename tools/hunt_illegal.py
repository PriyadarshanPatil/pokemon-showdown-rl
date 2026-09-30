"""Hunt the intermittent illegal-action bug with a TRAINED policy.

The 10k-episode random-action soak never surfaced it: random play does not visit the
states a trained policy concentrates on. This drives self-play with a real checkpoint and
counts the worker's guard substitutions, which are reported on worker exit.
"""
from __future__ import annotations
import argparse, re, sys
import numpy as np
import torch

from psrl.env.vec_env import VecBattleEnv
from psrl.nets.model import load_policy


@torch.no_grad()
def main(a):
    dev = torch.device(a.device)
    model = load_policy(a.checkpoint, dev)
    env = VecBattleEnv(n_workers=a.workers, batch=a.batch, seed=f"hunt{a.seed}")
    f = env.observe()
    hidden = None
    episodes = 0
    steps = 0
    while episodes < a.episodes:
        need = np.flatnonzero(f["needs"])
        acts = np.full(env.n_slots, -1, dtype=np.int32)
        if need.size:
            ids = torch.from_numpy(f["ids"][need].astype(np.int64)).unsqueeze(1).to(dev)
            scal = torch.from_numpy(f["scalars"][need].copy()).unsqueeze(1).to(dev)
            mask = torch.from_numpy(f["mask"][need].copy()).unsqueeze(1).to(dev)
            if hidden is None:
                hidden = torch.zeros(1, env.n_slots, model.d_hidden, device=dev)
            logits, _, h = model(ids, scal, mask, hidden[:, need])
            hidden[:, need] = h
            lg = logits.squeeze(1)
            # greedy concentrates hardest, which is what triggered the original report
            chosen = lg.argmax(-1) if a.greedy else torch.distributions.Categorical(logits=lg).sample()
            acts[need] = chosen.cpu().numpy()
        f = env.step(acts)
        steps += 1
        d = f["done"]
        if d.any():
            episodes += int(d.sum())
            hidden[:, torch.from_numpy(d).to(dev)] = 0.0
        if steps % 500 == 0:
            print(f"  {episodes:,}/{a.episodes:,} episodes", flush=True)

    # drain each worker's stderr for guard reports
    total = 0
    details = []
    for w in env.workers:
        try:
            w.proc.stdin.close()
        except Exception:
            pass
    for w in env.workers:
        try:
            err = w.proc.stderr.read().decode(errors="replace")
        except Exception:
            err = ""
        for m in re.finditer(r"substituted (\d+) illegal client actions", err):
            total += int(m.group(1))
        for blk in re.findall(r"ILLEGAL ACTION #\d+[\s\S]{0,400}?stale=\S+ retrySide=\S+ reqState=\S+", err):
            details.append(blk)
    env.close()
    print(f"\n{episodes:,} episodes, greedy={a.greedy}")
    print(f"illegal-action substitutions: {total}")
    for blk in details[:3]:
        print("---\n" + blk)
    return 0 if total == 0 else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/bc-v5-pad443/bc.pt")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--sample", dest="greedy", action="store_false")
    sys.exit(main(ap.parse_args()))
