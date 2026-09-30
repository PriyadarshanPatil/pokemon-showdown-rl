"""Step-16 gate: does the league's strength rise against frozen references, or cycle?

Win rate vs a FIXED opponent maps to an Elo gap by 400*log10(p/(1-p)), so a rising
win rate is a rising Elo. Cycling means a real decrease -- one larger than the noise
in the difference of two independent binomials.
"""
import json, math, sys

runs = sys.argv[1:]
OPPS = ("random", "maxdamage", "heuristic")

def elo(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 400 * math.log10(p / (1 - p))

series = {}   # opp -> iter -> [(p, n) per seed]
for r in runs:
    with open(f"runs/{r}/league_history.json") as f:
        hist = json.load(f)
    for e in hist:
        if "evals" not in e:
            continue
        for x in e["evals"]:
            series.setdefault(x["opponent"], {}).setdefault(e["iter"], []).append((x["win_rate"], x["n"]))

for opp in OPPS:
    pts = sorted(series.get(opp, {}).items())
    if not pts:
        continue
    print(f"\n{opp}  ({len(runs)} seeds, n={sum(n for _, n in pts[0][1]):,} pooled per checkpoint)")
    pooled = []
    for it, vals in pts:
        p = sum(pi * ni for pi, ni in vals) / sum(ni for _, ni in vals)
        n = sum(ni for _, ni in vals)
        se = math.sqrt(p * (1 - p) / n)
        pooled.append((it, p, se, [pi for pi, _ in vals]))
        spread = "  ".join(f"{pi*100:.1f}" for pi, _ in vals)
        print(f"  iter {it:5d}  {p*100:5.1f}% +/-{1.96*se*100:.1f}  elo {elo(p):+7.1f}   seeds: {spread}")
    print("  consecutive changes (significant = |delta| > 1.96*se_diff):")
    worst = None
    for (i0, p0, s0, _), (i1, p1, s1, _) in zip(pooled, pooled[1:]):
        d = p1 - p0
        sd = math.sqrt(s0**2 + s1**2)
        sig = abs(d) > 1.96 * sd
        print(f"    {i0:5d}->{i1:<5d} {d*100:+5.1f} pts (elo {elo(p1)-elo(p0):+6.1f})  "
              f"threshold +/-{1.96*sd*100:.1f}  {'SIGNIFICANT' if sig else 'noise'}")
        if sig and d < 0 and (worst is None or d < worst[0]):
            worst = (d, i0, i1)
    net = pooled[-1][1] - pooled[0][1]
    print(f"  net {net*100:+.1f} pts (elo {elo(pooled[-1][1])-elo(pooled[0][1]):+.1f})"
          f"   verdict: {'CYCLING at %d->%d' % worst[1:] if worst else 'no significant decrease'}")
