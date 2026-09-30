"""PPO self-play (Phase 4 step 15).

Pokemon is turn-based and a slot only acts at some barriers, so a "transition" is a
decision point, not a barrier. Reward earned between two decisions is credited to the
earlier one; the terminal +/-1 lands on a slot's last action.

Self-play means both sides are the SAME policy, so every rollout contains each battle
from both perspectives and the reward is exactly zero-sum.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from psrl.env.vec_env import VecBattleEnv
from psrl.nets.model import BattlePolicy, load_policy_dims
from psrl.nets import layout as L
from psrl.train.eval_policy import evaluate
from psrl.train.run_meta import write_run_meta

REPO = Path(__file__).resolve().parents[2]


class Rollout:
    """Per-slot buffers of decision points."""

    def __init__(self, n_slots, horizon, dev):
        self.n, self.T, self.dev = n_slots, horizon, dev
        self.clear()

    def clear(self):
        self.ids = [[] for _ in range(self.n)]
        self.scal = [[] for _ in range(self.n)]
        self.mask = [[] for _ in range(self.n)]
        self.act = [[] for _ in range(self.n)]
        self.logp = [[] for _ in range(self.n)]
        self.val = [[] for _ in range(self.n)]
        self.rew = [[] for _ in range(self.n)]
        self.done = [[] for _ in range(self.n)]

    def ready(self):
        return min(len(a) for a in self.act) >= self.T

    def stack(self):
        T = self.T
        def pack(src, dtype, extra=()):
            arr = np.zeros((self.n, T) + extra, dtype=dtype)
            for i in range(self.n):
                for t in range(T):
                    arr[i, t] = src[i][t]
            return arr
        return dict(
            ids=torch.from_numpy(pack(self.ids, np.int64, (L.N_IDS,))).to(self.dev),
            scal=torch.from_numpy(pack(self.scal, np.float32, (L.N_SCALARS,))).to(self.dev),
            mask=torch.from_numpy(pack(self.mask, bool, (L.N_ACTIONS,))).to(self.dev),
            act=torch.from_numpy(pack(self.act, np.int64)).to(self.dev),
            logp=torch.from_numpy(pack(self.logp, np.float32)).to(self.dev),
            val=torch.from_numpy(pack(self.val, np.float32)).to(self.dev),
            rew=torch.from_numpy(pack(self.rew, np.float32)).to(self.dev),
            done=torch.from_numpy(pack(self.done, np.float32)).to(self.dev),
        )

    def drop(self):
        for i in range(self.n):
            for buf in (self.ids, self.scal, self.mask, self.act, self.logp,
                        self.val, self.rew, self.done):
                del buf[i][:self.T]


def collect(env, model, roll, hidden, dev, stats):
    """Run barriers until every slot has `horizon` decision points buffered."""
    f = env.observe()
    while not roll.ready():
        need = f["needs"]
        idx = np.flatnonzero(need)
        acts = np.full(env.n_slots, -1, dtype=np.int32)
        if idx.size:
            ids = torch.from_numpy(f["ids"][idx].astype(np.int64)).unsqueeze(1).to(dev)
            scal = torch.from_numpy(f["scalars"][idx].copy()).unsqueeze(1).to(dev)
            mask = torch.from_numpy(f["mask"][idx].copy()).unsqueeze(1).to(dev)
            h = hidden[:, idx] if hidden is not None else None
            with torch.no_grad():
                logits, value, h2 = model(ids, scal, mask, h)
            dist = torch.distributions.Categorical(logits=logits.squeeze(1))
            a = dist.sample()
            lp = dist.log_prob(a)
            if hidden is None:
                hidden = torch.zeros(1, env.n_slots, model.d_hidden, device=dev)
            hidden[:, idx] = h2
            a_np = a.cpu().numpy()
            acts[idx] = a_np
            for k, i in enumerate(idx):
                roll.ids[i].append(f["ids"][i]); roll.scal[i].append(f["scalars"][i])
                roll.mask[i].append(f["mask"][i]); roll.act[i].append(int(a_np[k]))
                roll.logp[i].append(float(lp[k])); roll.val[i].append(float(value[k, 0]))
                roll.rew[i].append(0.0); roll.done[i].append(0.0)
        f = env.step(acts)
        # credit reward to each slot's most recent decision
        r, d = f["reward"], f["done"]
        for i in np.flatnonzero((r != 0) | d):
            if roll.act[i]:
                roll.rew[i][-1] += float(r[i])
                if d[i]:
                    roll.done[i][-1] = 1.0
                    stats["episodes"] += 1
                    stats["returns"].append(float(r[i]))
        if d.any() and hidden is not None:
            hidden[:, torch.from_numpy(d).to(dev)] = 0.0   # fresh battle, fresh memory
    return hidden, f


def gae(rew, val, done, last_val, gamma, lam):
    T = rew.shape[1]
    adv = torch.zeros_like(rew)
    nextval, nextadv = last_val, torch.zeros_like(last_val)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[:, t]
        delta = rew[:, t] + gamma * nextval * nonterm - val[:, t]
        nextadv = delta + gamma * lam * nonterm * nextadv
        adv[:, t] = nextadv
        nextval = val[:, t]
    return adv, adv + val


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    dims: dict = {}
    if a.init:
        model, dims, ck = load_policy_dims(a.init, dev)
        model.train()
    else:
        model = BattlePolicy().to(dev)
    if a.init:
        print(f"initialised from {a.init}"
              + (f" (BC val acc {ck['best_val_acc']*100:.1f}%)" if ck.get("best_val_acc", -1) >= 0 else ""))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    # Phase 7 step 4: an optional anchor to the BC prior. Self-play against an opponent that
    # never pivots never punishes staying in, so PPO correctly unlearns switching for that
    # opponent distribution - all eight 2.7 runs took bc-v9's 12.35% switch mass down to
    # 2.69-6.89%. The reference is the --init checkpoint, frozen before the optimiser runs.
    reference = None
    if a.kl_coef and a.init:
        reference = BattlePolicy(**dims).to(dev).eval()
        reference.load_state_dict(torch.load(a.init, map_location=dev)["model"])
        for p in reference.parameters():
            p.requires_grad_(False)
        print(f"anchored to {a.init} with kl_coef={a.kl_coef}")

    env = VecBattleEnv(n_workers=a.workers, batch=a.batch, seed=f"ppo{a.seed}",
                       shaping=a.shaping, no_posterior=a.no_posterior)
    roll = Rollout(env.n_slots, a.horizon, dev)
    hidden = None
    stats = {"episodes": 0, "returns": []}
    hist = []
    out = REPO / "runs" / a.name
    if out.exists():
        raise SystemExit(f"{out} exists - runs/ is gitignored, so overwriting it is "
                         f"unrecoverable. Pick a new --name, or delete it deliberately.")
    out.mkdir(parents=True)
    write_run_meta(out, a)

    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        hidden, f = collect(env, model, roll, hidden, dev, stats)
        b = roll.stack()
        with torch.no_grad():
            lids = torch.from_numpy(f["ids"].astype(np.int64)).unsqueeze(1).to(dev)
            lscal = torch.from_numpy(f["scalars"].copy()).unsqueeze(1).to(dev)
            lmask = torch.from_numpy(f["mask"].copy()).to(dev).unsqueeze(1)
            lmask = lmask | (~lmask.any(-1, keepdim=True))
            _, lastv, _ = model(lids, lscal, lmask, hidden)
            last_val = lastv[:, 0]
        adv, ret = gae(b["rew"], b["val"], b["done"], last_val, a.gamma, a.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl = vl = ent = kl = klref = 0.0
        for _ in range(a.epochs):
            perm = torch.randperm(env.n_slots, device=dev)
            for s in range(0, env.n_slots, a.minibatch):
                mb = perm[s:s + a.minibatch]
                m = b["mask"][mb]
                m = m | (~m.any(-1, keepdim=True))
                logits, value, _ = model(b["ids"][mb], b["scal"][mb], m)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(b["act"][mb])
                ratio = (lp - b["logp"][mb]).exp()
                a1 = ratio * adv[mb]
                a2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv[mb]
                p_loss = -torch.min(a1, a2).mean()
                v_loss = F.mse_loss(value, ret[mb])
                e = model.masked_entropy(logits, b["mask"][mb]).mean()
                loss = p_loss + a.vf_coef * v_loss - a.ent_coef * e
                if reference is not None:
                    with torch.no_grad():
                        rlogits, _, _ = reference(b["ids"][mb], b["scal"][mb], m)
                    # KL(reference || current), so the penalty falls on DROPPING mass the prior
                    # put somewhere - which is the failure mode. The reverse direction would
                    # punish adding mass instead, the opposite of what is wanted here.
                    rlogp = F.log_softmax(rlogits, dim=-1)
                    clogp = F.log_softmax(logits, dim=-1)
                    legal = b["mask"][mb]
                    d = (rlogp.exp() * (rlogp - clogp)).masked_fill(~legal, 0.0).sum(-1)
                    kl_ref = d.mean()
                    loss = loss + a.kl_coef * kl_ref
                    klref += kl_ref.item()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                pl += p_loss.item(); vl += v_loss.item(); ent += e.item()
                kl += (b["logp"][mb] - lp).mean().item()
        nmb = a.epochs * max(1, env.n_slots // a.minibatch)
        roll.drop()
        hidden = hidden.detach() if hidden is not None else None

        if it % a.log_every == 0:
            el = time.perf_counter() - t0
            rets = stats["returns"][-2000:]
            print(f"iter {it:4d} | eps {stats['episodes']:6,d} | {stats['episodes']/el:5.1f} eps/s | "
                  f"pl {pl/nmb:+.4f} vl {vl/nmb:.4f} ent {ent/nmb:.3f} kl {kl/nmb:+.4f}"
                  f"{f' klref {klref/nmb:.4f}' if reference is not None else ''} | "
                  f"mean return {np.mean(rets) if rets else 0:+.3f}", flush=True)
            hist.append(dict(iter=it, episodes=stats["episodes"], pl=pl/nmb, vl=vl/nmb,
                             ent=ent/nmb, kl=kl/nmb, klref=klref/nmb))
        if a.eval_every and it % a.eval_every == 0:
            res = [evaluate(model, o, n_battles=a.eval_battles, device=a.device, seed=f"ev{it}-{o}",
                    no_posterior=a.no_posterior)
                   for o in a.opponents.split(",")]
            print("   eval: " + "  ".join(f"{r['opponent']} {r['win_rate']*100:.1f}%" for r in res), flush=True)
            hist.append(dict(iter=it, evals=res))
            torch.save({"model": model.state_dict(), "iter": it,
                        "layout": {"ids": L.N_IDS, "scalars": L.N_SCALARS, "actions": L.N_ACTIONS},
                        "dims": dims},
                       out / "ppo.pt")
            (out / "ppo_history.json").write_text(json.dumps(hist, indent=1))
            model.train()
    env.close()
    torch.save({"model": model.state_dict(), "iter": a.iters,
                "layout": {"ids": L.N_IDS, "scalars": L.N_SCALARS, "actions": L.N_ACTIONS},
                "dims": dims},
               out / "ppo.pt")
    print("\nfinal evaluation")
    finals = []
    for o in a.opponents.split(","):
        r = evaluate(model, o, n_battles=a.final_eval, device=a.device, seed=f"final-{o}",
                     no_posterior=a.no_posterior)
        ci = 1.96 * np.sqrt(max(r["win_rate"] * (1 - r["win_rate"]), 1e-9) / r["n"])
        print(f"  vs {o:<10} {r['win_rate']*100:5.1f}%  [{(r['win_rate']-ci)*100:.1f}, {(r['win_rate']+ci)*100:.1f}]  (n={r['n']})")
        finals.append(r)
    hist.append({"final": finals})
    (out / "ppo_history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="ppo-dev")
    ap.add_argument("--init", default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-battles", type=int, default=150)
    ap.add_argument("--final-eval", type=int, default=400)
    ap.add_argument("--opponents", default="random,maxdamage,heuristic")
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="anchor strength to the --init policy, KL(init || current) over legal "
                         "actions. 0 reproduces unanchored PPO exactly; self-play otherwise "
                         "unlearns switching (see docs/PHASE7.md)")
    ap.add_argument("--shaping", type=float, default=0.0,
                    help="ablation: per-step HP-differential reward coefficient")
    ap.add_argument("--no-posterior", action="store_true",
                    help="ablation: zero the set-posterior features")
    main(ap.parse_args())
