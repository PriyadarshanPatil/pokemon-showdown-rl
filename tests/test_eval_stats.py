"""play_stats counts what live play showed and scripted evals never measured: voluntary
switches, and back-to-back moves with the same Pokemon on both sides that reuse the move."""
import numpy as np

from psrl.nets import layout as L
from psrl.train.eval_policy import play_stats


def _obs(own_species, foe_species, moves):
    ids = np.zeros((1, L.N_IDS), np.int64)
    scal = np.zeros((1, L.N_SCALARS), np.float32)
    ids[0, 0], ids[0, 5:9], scal[0, 3] = own_species, moves, 1        # own slot 1, active
    ids[0, L.N_MON * L.IDS_PER_MON] = foe_species                      # foe slot 1...
    scal[0, L.N_MON * L.SCAL_PER_MON + 3] = 1                          # ...active
    return ids, scal


def test_counts_repeats_and_voluntary_switches():
    prev = {"move": np.zeros(1, bool), "own": np.full(1, -1), "foe": np.full(1, -2), "id": np.full(1, -1)}
    c = dict(free=0, switches=0, pairs=0, repeats=0)
    free = np.ones((1, L.N_ACTIONS), bool)
    forced = np.zeros((1, L.N_ACTIONS), bool)
    forced[0, 8:] = True

    def step(own, foe, act, mask=free):
        ids, scal = _obs(own, foe, [11, 12, 13, 14])
        play_stats(ids, scal, mask, np.ones(1, bool), np.array([act]), prev, c)

    step(1, 2, 0)            # move 1
    step(1, 2, 0)            # move 1 again, same matchup: a repeat
    step(1, 2, 5)            # move 2 with tera, same matchup: a pair, not a repeat
    step(1, 3, 1)            # the foe changed: not a pair
    step(1, 3, 9)            # voluntary switch
    step(4, 3, 10, forced)   # forced switch: not a free decision
    step(4, 3, 0)            # first move after a switch: not a pair
    assert c == dict(free=6, switches=1, pairs=2, repeats=1), c


if __name__ == "__main__":
    # Self-running so the suite needs no test framework, matching tools/run_gates.sh.
    import sys, traceback
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
