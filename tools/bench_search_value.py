#!/usr/bin/env python3
"""Phase 6 gate 3.1 — the batched-value half, to be added to tools/bench_search.js's node half.

The gate is p95 <= 2s per decision. A one-ply search values every candidate successor in one
forward pass, so the cost that matters is a single batched forward at the branching factors
bench_search.js measures: K ~ 8-13 for one opponent reply per candidate, K*M ~ 70-170 when
marginalising over replies.

Timed with the real BattlePolicy and the real layout, not a stand-in, because the trunk width is
what the layout change moved.

    uv run python tools/bench_search_value.py [--device mps] [--reps 200]
"""
import argparse
import time

import numpy as np
import torch

from psrl.nets import layout as L
from psrl.nets.model import BattlePolicy


def bench(model, dev, batch: int, reps: int, warmup: int = 20) -> tuple[float, float]:
    """Returns (p50, p95) milliseconds for one batched forward of `batch` successors."""
    ids = torch.randint(0, 50, (batch, 1, L.N_IDS), device=dev)
    scal = torch.rand(batch, 1, L.N_SCALARS, device=dev)
    mask = torch.ones(batch, 1, L.N_ACTIONS, dtype=torch.bool, device=dev)
    times = []
    with torch.no_grad():
        for i in range(warmup + reps):
            if dev.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            model(ids, scal, mask, None)
            if dev.type == "mps":
                torch.mps.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    a = np.sort(np.array(times))
    return float(a[len(a) // 2]), float(a[int(0.95 * len(a))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()

    dev = torch.device(a.device)
    model = BattlePolicy().to(dev).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"BattlePolicy {n:,} params on {a.device}, layout {L.N_IDS} ids / {L.N_SCALARS} scalars")
    print(f"{'batch':>6}  {'p50 ms':>8}  {'p95 ms':>8}   regime")
    regimes = {13: "FIXED p95 (K=13)", 9: "FIXED typical", 170: "MARGINAL p95 (K*M=170)",
               81: "MARGINAL typical", 512: "headroom"}
    out = {}
    for batch in sorted(regimes):
        p50, p95 = bench(model, dev, batch, a.reps)
        out[batch] = p95
        print(f"{batch:>6}  {p50:>8.2f}  {p95:>8.2f}   {regimes[batch]}")

    print("\nper-decision totals, node half + value half (node p95 from tools/bench_search.js):")
    for label, node_p95, batch in [("FIXED", 12.0, 13), ("MARGINAL", 162.0, 170)]:
        total = node_p95 + out[batch]
        print(f"  {label:<9} {node_p95:>6.0f}ms + {out[batch]:>6.1f}ms = {total:>7.1f}ms  "
              f"{'PASS' if total <= 2000 else 'FAIL'} (gate 2000ms)")


if __name__ == "__main__":
    main()
