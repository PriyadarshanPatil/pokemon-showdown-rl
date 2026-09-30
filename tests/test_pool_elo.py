"""Pool Elo must reflect true strength, not conflate it with learner improvement."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
import random
from psrl.train.league import Pool


class _FakeModel:
    def state_dict(self):
        return {}


def test_elo_gap_matches_win_rate():
    """A learner winning p of the time should settle ~400*log10(p/(1-p)) above the pool."""
    for p in (0.7, 0.5, 0.3):
        pool = Pool(dev=None, max_size=4)
        pool.add(_FakeModel(), "frozen")
        rng = random.Random(0)
        for _ in range(4000):
            pool.record(0, 1.0 if rng.random() < p else 0.0)
        gap = pool.learner_elo - pool.entries[0]["elo"]
        expected = 400 * (0 if p == 0.5 else __import__("math").log10(p / (1 - p)))
        assert abs(gap - expected) < 60, f"p={p}: gap {gap:.0f} vs expected {expected:.0f}"


def test_snapshot_inherits_learner_rating():
    """A frozen copy of the learner is exactly as strong as the learner right now."""
    pool = Pool(dev=None, max_size=4)
    pool.add(_FakeModel(), "init")
    for _ in range(200):
        pool.record(0, 1.0)
    assert pool.learner_elo > 100, "learner should have climbed"
    pool.add(_FakeModel(), "later")
    assert abs(pool.entries[-1]["elo"] - pool.learner_elo) < 1e-9


def test_zero_sum():
    """Elo is conserved: what the learner gains, the opponent loses."""
    pool = Pool(dev=None, max_size=4)
    pool.add(_FakeModel(), "a")
    before = pool.learner_elo + pool.entries[0]["elo"]
    for s in (1.0, 0.0, 0.5, 1.0):
        pool.record(0, s)
    after = pool.learner_elo + pool.entries[0]["elo"]
    assert abs(after - before) < 1e-9


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in fns:
        try:
            f(); print(f"  [PASS] {n}")
        except Exception:
            bad += 1; print(f"  [FAIL] {n}"); traceback.print_exc()
    print(f"{len(fns)-bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
