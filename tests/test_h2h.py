"""Head-to-head evaluation must be symmetric (Phase 7's replacement instrument).

The gate deliberately asserts structural invariants rather than a win rate, because the bug that
would ruin this instrument is a slot or hidden-state mixup - one model silently driving both
sides, or the two sides sharing a recurrent state - and that failure looks exactly like a real
effect rather than like an error.

Two invariants:
  1. the same checkpoint on both sides scores about 50%
  2. swapping the sides gives complementary win rates, so A-vs-B and B-vs-A sum to about 100%

Both use modest battle counts, so the tolerances are set from the binomial standard error rather
than from taste: at n=300, one standard error is about 2.9 points, so +/-12 is above 4 sd and
will not flake.
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from psrl.train.eval_h2h import _load, head2head          # noqa: E402

N = 300


def _checkpoint():
    """The newest checkpoint whose width matches the current layout.

    A layout change strands narrower checkpoints, so this gate must follow the padded copy
    rather than hardcode one path - otherwise it fails on a size mismatch that says nothing
    about head-to-head symmetry, which is all this file is meant to test.
    """
    from psrl.nets import layout as L
    for name in (f"bc-v10-pad{L.N_SCALARS}", "bc-v10", f"bc-v9-pad{L.N_SCALARS}", "bc-v9"):
        p = REPO / "runs" / name / "bc.pt"
        if not p.exists():
            continue
        try:
            if int(torch.load(p, map_location="cpu").get("layout", {}).get("scalars", 0)) == L.N_SCALARS:
                return p
        except Exception:
            continue
    return None


CK = _checkpoint()


def _skip_if_no_checkpoint():
    if CK is None:
        print(f"  [SKIP] no checkpoint matching the current layout ({__import__('psrl.nets.layout', fromlist=['x']).N_SCALARS} scalars)")
        return True
    return False


def test_same_checkpoint_scores_about_even():
    if _skip_if_no_checkpoint():
        return
    dev = torch.device("cpu")
    m = _load(str(CK), dev)
    r = head2head(m, m, n_battles=N, seed="h2h-null", device="cpu")
    wr = 100 * r["win_rate"]
    print(f"    same checkpoint both sides: {wr:.2f}% over n={r['n']}")
    assert 38 <= wr <= 62, f"identical policies should be near 50%, got {wr:.2f}%"


def test_swapping_sides_is_complementary():
    if _skip_if_no_checkpoint():
        return
    dev = torch.device("cpu")
    m = _load(str(CK), dev)
    # same weights, different decoding: one side greedy, the other two-stage
    fwd = head2head(m, m, n_battles=N, seed="h2h-swap", tau_a=0.30, device="cpu")
    rev = head2head(m, m, n_battles=N, seed="h2h-swap", tau_b=0.30, device="cpu")
    total = 100 * (fwd["win_rate"] + rev["win_rate"])
    print(f"    tau as A: {100*fwd['win_rate']:.2f}%, tau as B: {100*rev['win_rate']:.2f}%, "
          f"sum {total:.2f}%")
    assert 76 <= total <= 124, f"swapped sides should sum to about 100%, got {total:.2f}%"


def test_tau_actually_changes_the_switch_rate():
    if _skip_if_no_checkpoint():
        return
    dev = torch.device("cpu")
    m = _load(str(CK), dev)
    r = head2head(m, m, n_battles=N, seed="h2h-rate", tau_a=0.30, device="cpu")
    a = 100 * r["a_switches"] / max(r["a_free"], 1)
    b = 100 * r["b_switches"] / max(r["b_free"], 1)
    print(f"    switch rate with tau {a:.2f}% vs greedy {b:.2f}%")
    assert a > b, f"the tau side must switch more often ({a:.2f}% vs {b:.2f}%)"


if __name__ == "__main__":
    # Self-running so the suite needs no test framework, matching tools/run_gates.sh.
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception:
            failed += 1
            print(f"  [FAIL] {name}")
            traceback.print_exc()
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
