"""Fit Bradley-Terry (Elo-scale) ratings from paired-seed match results."""
from __future__ import annotations
import argparse, json, math
import numpy as np

SCALE = 400.0 / math.log(10.0)   # Elo points per logit


def fit(agents, results, iters=20000, lr=0.05, anchor=0):
    idx = {a: i for i, a in enumerate(agents)}
    r = np.zeros(len(agents))
    # (i, j, wins_i, games) with ties counted as half a win each
    obs = []
    for m in results:
        if m["a"] == m["b"]:
            continue
        w = m["wins"] + 0.5 * m["ties"]
        obs.append((idx[m["a"]], idx[m["b"]], w, m["games"]))
    for _ in range(iters):
        g = np.zeros_like(r)
        for i, j, w, n in obs:
            p = 1.0 / (1.0 + math.exp(-(r[i] - r[j])))
            g[i] += w - n * p
            g[j] -= w - n * p
        r += lr * g / max(1, sum(n for *_, n in obs))
        r -= r[anchor]
    return r * SCALE


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--anchor", default="random")
    a = ap.parse_args()
    with open(a.results) as f:
        d = json.load(f)
    agents = sorted({m["a"] for m in d["results"]} | {m["b"] for m in d["results"]})
    anchor = agents.index(a.anchor) if a.anchor in agents else 0
    elo = fit(agents, d["results"], anchor=anchor)
    order = np.argsort(-elo)
    print(f"Bradley-Terry ratings (Elo scale, {a.anchor} anchored at 0)\n")
    for k in order:
        print(f"  {agents[k]:<12} {elo[k]:+8.1f}")
    print("\npairwise expected win rate:")
    print("            " + "".join(f"{agents[j]:>12}" for j in order))
    for i in order:
        row = "".join(f"{100/(1+10**((elo[j]-elo[i])/400)):>11.1f}%" for j in order)
        print(f"  {agents[i]:<10}{row}")
