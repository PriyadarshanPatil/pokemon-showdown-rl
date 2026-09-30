"""Behaviour cloning from human replays (Phase 4 step 14)."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from psrl.data.replay_loader import ReplayLoader, MultiReplayLoader
from psrl.nets.model import BattlePolicy
from psrl.nets import layout as L
from psrl.train.eval_policy import evaluate
from psrl.train.run_meta import write_run_meta

REPO = Path(__file__).resolve().parents[2]


def losses_for(model, batch, dev):
    ids = torch.from_numpy(batch["ids"]).to(dev)
    scal = torch.from_numpy(batch["scalars"].copy()).to(dev)
    mask = torch.from_numpy(batch["mask"].copy()).to(dev)
    act = torch.from_numpy(batch["action"]).to(dev)
    ret = torch.from_numpy(batch["ret"].copy()).to(dev)
    valid = torch.from_numpy(batch["valid"].copy()).to(dev)
    # padded steps have an all-false mask; make softmax well defined then drop them
    mask_safe = mask | (~mask.any(-1, keepdim=True))
    logits, value, _ = model(ids, scal, mask_safe)
    ce = F.cross_entropy(logits.reshape(-1, L.N_ACTIONS), act.reshape(-1), reduction="none")
    ce = (ce.reshape(act.shape) * valid).sum() / valid.sum().clamp(min=1)
    vl = (((value - ret) ** 2) * valid).sum() / valid.sum().clamp(min=1)
    correct = ((logits.argmax(-1) == act) & valid).sum() / valid.sum().clamp(min=1)
    return ce, vl, correct, int(valid.sum())


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    out = REPO / "runs" / a.name
    if out.exists():
        raise SystemExit(f"{out} exists - runs/ is gitignored, so overwriting it is "
                         f"unrecoverable. Pick a new --name, or delete it deliberately.")
    out.mkdir(parents=True)
    write_run_meta(out, a)
    dims = {"d_emb": a.d_emb, "d_mon": a.d_mon, "d_hidden": a.d_hidden}
    model = BattlePolicy(**dims).to(dev)
    print(f"model dims {dims}: {sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    if a.cache:
        from psrl.data.cached_loader import CachedReplayLoader
        print(f"reading materialised sequences from {a.cache}")
        train = CachedReplayLoader(a.cache, batch=a.batch, split="train", seed=a.seed)
        if train.seqlen != a.seqlen:
            raise SystemExit(f"cache seqlen {train.seqlen} != --seqlen {a.seqlen}; the cache "
                             f"decides, so this flag would be silently ignored")
        val = CachedReplayLoader(a.cache, batch=a.batch, split="val", seed=a.seed + 1)
    else:
        train = MultiReplayLoader(n=a.loaders, batch=a.batch, seqlen=a.seqlen, split="train",
                                  no_posterior=a.no_posterior, min_rating=a.min_rating)
        # the split hashes on replay id, so val is filtered too: it measures imitation of the
        # same population the model is being trained to imitate
        val = ReplayLoader(batch=a.batch, seqlen=a.seqlen, split="val",
                           no_posterior=a.no_posterior, min_rating=a.min_rating)

    seen = 0; t0 = time.perf_counter(); hist = []
    # v2 peaked at 46.4% val accuracy then decayed to 39.2% by the last step while train
    # accuracy kept rising - classic overfitting on a small corpus. Saving the final step
    # threw the good model away, so keep the best-val checkpoint instead.
    best_acc = -1.0; best_state = None; best_step = 0; since_best = 0
    for step in range(1, a.steps + 1):
        b = train.next_batch()
        if b is None:
            print("train stream exhausted"); break
        model.train()
        ce, vl, acc, n = losses_for(model, b, dev)
        loss = ce + a.value_coef * vl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        seen += n
        if step % a.log_every == 0:
            model.eval()
            vs = []
            with torch.no_grad():
                # one batch is ~960 decisions, +/-3 points of noise: enough to make the
                # best-val pick a coin flip (bc-v7 chose step 13,000 of 60,000)
                for _ in range(a.val_batches):
                    vb = val.next_batch()
                    if vb is None:
                        break
                    vs.append(losses_for(model, vb, dev))
            if not vs:
                print("val stream exhausted"); break
            vce = sum(x[0] for x in vs) / len(vs)
            vvl = sum(x[1] for x in vs) / len(vs)
            vacc = sum(x[2] for x in vs) / len(vs)
            rate = seen / (time.perf_counter() - t0)
            print(f"step {step:5d} | seen {seen:8,d} ({rate:,.0f}/s) | "
                  f"train ce {ce.item():.4f} acc {acc.item()*100:5.1f}% | "
                  f"val ce {vce.item():.4f} acc {vacc.item()*100:5.1f}% | vloss {vvl.item():.4f}",
                  flush=True)
            hist.append(dict(step=step, seen=seen, train_ce=ce.item(), train_acc=acc.item(),
                             val_ce=vce.item(), val_acc=vacc.item(), val_v=vvl.item()))
            if vacc.item() > best_acc:
                best_acc, best_step, since_best = vacc.item(), step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
                if a.patience and since_best >= a.patience:
                    print(f"early stop: no val improvement for {a.patience} evals "
                          f"(best {best_acc*100:.1f}% at step {best_step})")
                    break
    train.close(); val.close()

    layout = {"ids": L.N_IDS, "scalars": L.N_SCALARS, "actions": L.N_ACTIONS}
    # loaders still build BattlePolicy() with defaults, so a non-default width is
    # only usable once they read this back - see docs/CODEBASE_NOTES.md
    dims_meta = dims
    # keep the last step too: a noisy best-val pick must never be the only model we keep
    torch.save({"model": model.state_dict(), "step": step, "layout": layout,
                "dims": dims_meta}, out / "bc_final.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nrestored best checkpoint: val acc {best_acc*100:.1f}% at step {best_step}")
    torch.save({"model": model.state_dict(), "best_val_acc": best_acc, "best_step": best_step,
                "layout": layout, "dims": dims_meta}, out / "bc.pt")
    print(f"checkpoint -> {out/'bc.pt'} (last step also saved as bc_final.pt)")

    print("\nevaluating as a policy (greedy, paired vs in-worker baselines)")
    evals = []
    for opp in a.opponents.split(","):
        r = evaluate(model, opp, n_battles=a.eval_battles, device=a.device, seed=f"eval-{opp}",
                     no_posterior=a.no_posterior)
        lo = r["win_rate"] - 1.96 * np.sqrt(max(r["win_rate"] * (1 - r["win_rate"]), 1e-9) / r["n"])
        hi = r["win_rate"] + 1.96 * np.sqrt(max(r["win_rate"] * (1 - r["win_rate"]), 1e-9) / r["n"])
        print(f"  vs {opp:<10} {r['win_rate']*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]  (n={r['n']})")
        evals.append(r)
    (out / "bc_results.json").write_text(json.dumps({"history": hist, "evals": evals}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="bc-dev")
    ap.add_argument("--eval-battles", type=int, default=200)
    ap.add_argument("--opponents", default="random,maxdamage,heuristic")
    ap.add_argument("--loaders", type=int, default=4)
    ap.add_argument("--no-posterior", action="store_true",
                    help="ablation: zero the set-posterior features (layout unchanged)")
    ap.add_argument("--val-batches", type=int, default=8,
                    help="batches averaged per val point; 1 batch is ~+/-3 points of noise")
    ap.add_argument("--cache", default=None,
                    help="directory from tools/build_cache.py; skips re-simulation")
    ap.add_argument("--min-rating", type=int, default=0,
                    help="train only on replays rated at least this; 0 keeps every replay, "
                         "NULL ratings included, which is the historical behaviour")
    ap.add_argument("--d-emb", type=int, default=32, help="entity embedding width")
    ap.add_argument("--d-mon", type=int, default=128, help="per-Pokemon encoder width")
    ap.add_argument("--d-hidden", type=int, default=256, help="trunk and GRU width")
    ap.add_argument("--patience", type=int, default=6, help="val evals without improvement before stopping; 0 disables")
    main(ap.parse_args())
