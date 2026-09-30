"""Phase 2 step 9 gate: long random-policy run. No desync, no leaks, no invalid frames."""
from __future__ import annotations
import argparse, subprocess, time
import numpy as np
from psrl.env.vec_env import VecBattleEnv


def node_procs() -> int:
    out = subprocess.run(["pgrep", "-f", "node .*worker.js"], capture_output=True, text=True)
    return len([l for l in out.stdout.split("\n") if l.strip()])


def main(target_episodes, workers, batch):
    before = node_procs()
    env = VecBattleEnv(n_workers=workers, batch=batch, seed="soak")
    rng = np.random.default_rng(1234)
    f = env.observe()
    episodes = 0; steps = 0; decisions = 0
    bad_reward = 0; no_legal = 0; nan_obs = 0; bad_id = 0
    rewards = []
    t0 = time.perf_counter()
    while episodes < target_episodes:
        need = np.flatnonzero(f["needs"])
        a = np.full(env.n_slots, -1, dtype=np.int32)
        for i in need:
            legal = np.flatnonzero(f["mask"][i])
            if legal.size == 0:
                no_legal += 1
                a[i] = 0
            else:
                a[i] = rng.choice(legal)
        # invariants on the frame we are about to act on
        if not np.isfinite(f["scalars"][need]).all():
            nan_obs += 1
        ids = f["ids"][need]
        if ids.size and (ids.min() < 0 or ids.max() > 2000):
            bad_id += 1
        decisions += need.size
        f = env.step(a)
        steps += 1
        d = f["done"]
        if d.any():
            r = f["reward"][d]
            bad_reward += int((~np.isin(r, (-1.0, 0.0, 1.0))).sum())
            rewards.extend(r.tolist())
            episodes += int(d.sum())
    dt = time.perf_counter() - t0
    during = node_procs()
    env.close()
    time.sleep(1.0)
    after = node_procs()

    rw = np.array(rewards)
    print(f"episodes(slot-terminations): {episodes:,} in {dt:.1f}s over {steps:,} barriers")
    print(f"  {decisions/dt:,.0f} decisions/sec | {episodes/dt:.1f} slot-episodes/sec")
    print(f"  reward balance: +1 {int((rw>0).sum()):,}  -1 {int((rw<0).sum()):,}  0 {int((rw==0).sum()):,}  mean {rw.mean():+.4f}")
    print(f"  node workers: before={before} during={during} after={after}")
    print("\nINVARIANTS")
    checks = [
        ("no out-of-range reward", bad_reward == 0, bad_reward),
        ("every acting slot had >=1 legal action", no_legal == 0, no_legal),
        ("all scalars finite", nan_obs == 0, nan_obs),
        ("all ids in range", bad_id == 0, bad_id),
        ("workers cleaned up", after == before, after - before),
        ("self-play reward is zero-sum", abs(rw.mean()) < 0.02, rw.mean()),
    ]
    ok = True
    for name, passed, val in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  ({val})")
        ok &= bool(passed)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    main(a.episodes, a.workers, a.batch)
