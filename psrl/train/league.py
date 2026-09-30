"""League training (Phase 4 step 16).

Naive self-play in a simultaneous-move imperfect-information game cycles: the policy
chases its own current weakness and forgets how to beat older strategies. So the learner
trains against a POOL of frozen snapshots sampled by prioritised fictitious self-play,
which keeps pressure on the strategies it currently loses to.

Slot layout from the worker is [battle0 p1, battle0 p2, ...]; the learner always plays
p1 (even slots) and a frozen opponent always plays p2 (odd slots), so gradients only ever
come from the learner's own decisions.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from psrl.env.vec_env import VecBattleEnv
from psrl.nets.model import BattlePolicy, load_policy_dims
from psrl.nets import layout as L
from psrl.train.eval_policy import evaluate
from psrl.train.ppo import Rollout, gae
from psrl.train.run_meta import write_run_meta

REPO = Path(__file__).resolve().parents[2]


class Pool:
    """Frozen snapshots with PFSP sampling and online Elo."""

    K = 16.0

    def __init__(self, dev, max_size=12):
        self.dev, self.max_size = dev, max_size
        self.entries = []      # {"sd", "elo", "wins", "games", "tag"}
        # The learner has a rating too. The previous version anchored it at 0, so as the
        # learner improved its wins pushed opponent Elo DOWN and "this opponent is weak"
        # became indistinguishable from "I got stronger".
        self.learner_elo = 0.0

    def add(self, model, tag, elo=None):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # A snapshot is a frozen copy of the learner, so it starts at the learner's rating.
        self.entries.append({"sd": sd, "elo": self.learner_elo if elo is None else elo,
                             "wins": 0.0, "games": 0, "tag": tag})
        if len(self.entries) > self.max_size:
            # keep the newest and the strongest; drop the weakest middle entry
            keep_idx = set([len(self.entries) - 1, int(np.argmax([e["elo"] for e in self.entries]))])
            drop = min((i for i in range(len(self.entries)) if i not in keep_idx),
                       key=lambda i: self.entries[i]["elo"])
            self.entries.pop(drop)

    def win_rate(self, i):
        e = self.entries[i]
        return 0.5 if e["games"] < 10 else e["wins"] / e["games"]

    def sample(self, n, rng):
        """PFSP: weight ~ (1 - learner_win_rate)^2, i.e. focus on opponents we lose to."""
        w = np.array([max(1e-3, (1.0 - self.win_rate(i)) ** 2) for i in range(len(self.entries))])
        w = w / w.sum()
        return rng.choice(len(self.entries), size=n, p=w)

    def record(self, i, learner_score):
        """learner_score: 1 win, 0 loss, 0.5 tie. Standard Elo, both ratings moving."""
        e = self.entries[i]
        e["games"] += 1
        e["wins"] += learner_score
        expected = 1.0 / (1.0 + 10 ** ((e["elo"] - self.learner_elo) / 400))
        delta = self.K * (learner_score - expected)
        self.learner_elo += delta
        e["elo"] -= delta


def forward_actions(model, f, idx, hidden, dev, sample=True):
    """Policy forward for a subset of slots; returns actions, logp, value, new hidden."""
    ids = torch.from_numpy(f["ids"][idx].astype(np.int64)).unsqueeze(1).to(dev)
    scal = torch.from_numpy(f["scalars"][idx].copy()).unsqueeze(1).to(dev)
    mask = torch.from_numpy(f["mask"][idx].copy()).unsqueeze(1).to(dev)
    mask = mask | (~mask.any(-1, keepdim=True))
    with torch.no_grad():
        logits, value, h2 = model(ids, scal, mask, hidden)
    dist = torch.distributions.Categorical(logits=logits.squeeze(1))
    a = dist.sample() if sample else logits.squeeze(1).argmax(-1)
    return a, dist.log_prob(a), value[:, 0], h2


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)

    learner, dims, ck = load_policy_dims(a.init, dev)
    learner.train()
    print(f"learner initialised from {a.init} at dims {dims or 'default'} "
          f"({sum(p.numel() for p in learner.parameters()):,} params)")

    # the snapshot holder must be the learner's width, or wide checkpoints fail to load into it
    frozen = BattlePolicy(**dims).to(dev).eval()
    pool = Pool(dev, max_size=a.pool_size)
    pool.add(learner, tag="init", elo=0.0)
    reference = {k: v.detach().clone() for k, v in learner.state_dict().items()}

    opt = torch.optim.Adam(learner.parameters(), lr=a.lr)
    env = VecBattleEnv(n_workers=a.workers, batch=a.batch, seed=f"league{a.seed}")
    n_battles = env.n_slots // 2
    learner_slots = np.arange(0, env.n_slots, 2)
    opp_slots = np.arange(1, env.n_slots, 2)

    # one opponent per battle, resampled when that battle ends
    battle_opp = pool.sample(n_battles, rng)
    roll = Rollout(len(learner_slots), a.horizon, dev)
    h_learn = None
    h_opp = None
    hist = []
    out = REPO / "runs" / a.name
    if out.exists():
        raise SystemExit(f"{out} exists - runs/ is gitignored, so overwriting it is "
                         f"unrecoverable. Pick a new --name, or delete it deliberately.")
    out.mkdir(parents=True)
    write_run_meta(out, a)
    t0 = time.perf_counter()
    episodes = 0

    f = env.observe()
    for it in range(1, a.iters + 1):
        while not roll.ready():
            acts = np.full(env.n_slots, -1, dtype=np.int32)
            need = f["needs"]

            # ---- learner (p1, even slots) ----
            li = learner_slots[need[learner_slots]]
            if li.size:
                if h_learn is None:
                    h_learn = torch.zeros(1, len(learner_slots), learner.d_hidden, device=dev)
                sel = (li // 2)
                a_l, lp_l, v_l, h2 = forward_actions(learner, f, li, h_learn[:, sel], dev)
                h_learn[:, sel] = h2
                acts[li] = a_l.cpu().numpy()
                an, lpn, vn = a_l.cpu().numpy(), lp_l.cpu().numpy(), v_l.cpu().numpy()
                for k, s in enumerate(li):
                    b = s // 2
                    roll.ids[b].append(f["ids"][s]); roll.scal[b].append(f["scalars"][s])
                    roll.mask[b].append(f["mask"][s]); roll.act[b].append(int(an[k]))
                    roll.logp[b].append(float(lpn[k])); roll.val[b].append(float(vn[k]))
                    roll.rew[b].append(0.0); roll.done[b].append(0.0)

            # ---- frozen opponents (p2, odd slots), grouped by snapshot ----
            oi = opp_slots[need[opp_slots]]
            if oi.size:
                if h_opp is None:
                    h_opp = torch.zeros(1, len(opp_slots), learner.d_hidden, device=dev)
                for pid in np.unique(battle_opp[oi // 2]):
                    grp = oi[battle_opp[oi // 2] == pid]
                    if not grp.size:
                        continue
                    frozen.load_state_dict(pool.entries[pid]["sd"])
                    sel = (grp // 2)
                    a_o, _, _, h2 = forward_actions(frozen, f, grp, h_opp[:, sel], dev)
                    h_opp[:, sel] = h2
                    acts[grp] = a_o.cpu().numpy()

            f = env.step(acts)

            # ---- credit rewards, record results, resample finished battles ----
            r, d = f["reward"], f["done"]
            for s in np.flatnonzero((r != 0) | d):
                if s % 2 == 0 and roll.act[s // 2]:
                    roll.rew[s // 2][-1] += float(r[s])
                    if d[s]:
                        roll.done[s // 2][-1] = 1.0
                        episodes += 1
                        score = 1.0 if r[s] > 0 else (0.0 if r[s] < 0 else 0.5)
                        pool.record(int(battle_opp[s // 2]), score)
            if d.any():
                ended = np.flatnonzero(d[learner_slots] | d[opp_slots])
                if ended.size:
                    battle_opp[ended] = pool.sample(ended.size, rng)
                if h_learn is not None:
                    h_learn[:, torch.from_numpy(d[learner_slots]).to(dev)] = 0.0
                if h_opp is not None:
                    h_opp[:, torch.from_numpy(d[opp_slots]).to(dev)] = 0.0

        # ---- PPO update on the learner's own transitions only ----
        b = roll.stack()
        with torch.no_grad():
            lm = torch.from_numpy(f["mask"][learner_slots].copy()).to(dev).unsqueeze(1)
            lm = lm | (~lm.any(-1, keepdim=True))
            _, lastv, _ = learner(
                torch.from_numpy(f["ids"][learner_slots].astype(np.int64)).unsqueeze(1).to(dev),
                torch.from_numpy(f["scalars"][learner_slots].copy()).unsqueeze(1).to(dev),
                lm, h_learn)
            last_val = lastv[:, 0]
        adv, ret = gae(b["rew"], b["val"], b["done"], last_val, a.gamma, a.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(learner_slots)
        pl = vl = ent = 0.0
        for _ in range(a.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, a.minibatch):
                mb = perm[s:s + a.minibatch]
                m = b["mask"][mb]; m = m | (~m.any(-1, keepdim=True))
                logits, value, _ = learner(b["ids"][mb], b["scal"][mb], m)
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(b["act"][mb]) - b["logp"][mb]).exp()
                p_loss = -torch.min(ratio * adv[mb],
                                    torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv[mb]).mean()
                v_loss = F.mse_loss(value, ret[mb])
                e = learner.masked_entropy(logits, b["mask"][mb]).mean()
                loss = p_loss + a.vf_coef * v_loss - a.ent_coef * e
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(learner.parameters(), 0.5); opt.step()
                pl += p_loss.item(); vl += v_loss.item(); ent += e.item()
        nmb = a.epochs * max(1, n // a.minibatch)
        roll.drop()
        if h_learn is not None: h_learn = h_learn.detach()
        if h_opp is not None: h_opp = h_opp.detach()

        if it % a.snapshot_every == 0:
            pool.add(learner, tag=f"it{it}")
            print(f"   snapshot added (pool size {len(pool.entries)})", flush=True)

        if it % a.log_every == 0:
            el = time.perf_counter() - t0
            wr = [f"{pool.entries[i]['tag']}:{pool.win_rate(i)*100:.0f}%" for i in range(len(pool.entries))]
            print(f"iter {it:4d} | eps {episodes:6,d} | {episodes/el:5.1f} eps/s | "
                  f"pl {pl/nmb:+.4f} vl {vl/nmb:.4f} ent {ent/nmb:.3f} | pool {' '.join(wr)}", flush=True)
            hist.append(dict(iter=it, episodes=episodes, pl=pl/nmb, vl=vl/nmb, ent=ent/nmb,
                             pool=[{"tag": e["tag"], "elo": e["elo"], "games": e["games"],
                                    "wr": pool.win_rate(i)} for i, e in enumerate(pool.entries)]))
        if a.eval_every and it % a.eval_every == 0:
            # gate signal: Elo vs the FROZEN reference must rise, not cycle
            frozen.load_state_dict(reference)
            res = [evaluate(learner, o, n_battles=a.eval_battles, device=a.device, seed=f"lg{it}-{o}")
                   for o in a.opponents.split(",")]
            print("   eval: " + "  ".join(f"{r['opponent']} {r['win_rate']*100:.1f}%" for r in res), flush=True)
            hist.append(dict(iter=it, evals=res))
            torch.save({"model": learner.state_dict(), "iter": it,
                        "layout": {"ids": L.N_IDS, "scalars": L.N_SCALARS, "actions": L.N_ACTIONS},
                        "dims": dims},
                       out / "league.pt")
            (out / "league_history.json").write_text(json.dumps(hist, indent=1))
            learner.train()

    env.close()
    torch.save({"model": learner.state_dict(), "iter": a.iters,
                "layout": {"ids": L.N_IDS, "scalars": L.N_SCALARS, "actions": L.N_ACTIONS},
                "dims": dims},
               out / "league.pt")
    print("\nfinal evaluation")
    finals = []
    for o in a.opponents.split(","):
        r = evaluate(learner, o, n_battles=a.final_eval, device=a.device, seed=f"lgfinal-{o}")
        ci = 1.96 * math.sqrt(max(r["win_rate"] * (1 - r["win_rate"]), 1e-9) / r["n"])
        print(f"  vs {o:<10} {r['win_rate']*100:5.1f}%  [{(r['win_rate']-ci)*100:.1f}, {(r['win_rate']+ci)*100:.1f}]  (n={r['n']})")
        finals.append(r)
    hist.append({"final": finals, "pool": [{"tag": e["tag"], "elo": e["elo"], "games": e["games"]}
                                           for e in pool.entries]})
    (out / "league_history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--pool-size", type=int, default=10)
    ap.add_argument("--snapshot-every", type=int, default=150)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="league-dev")
    ap.add_argument("--init", required=True)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--eval-battles", type=int, default=200)
    ap.add_argument("--final-eval", type=int, default=400)
    ap.add_argument("--opponents", default="random,maxdamage,heuristic")
    main(ap.parse_args())
