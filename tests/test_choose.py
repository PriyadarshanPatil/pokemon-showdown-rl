"""Two-stage action selection (Phase 7 step 1).

Greedy argmax splits the switch vote across up to five targets while move probability
concentrates on one, so a policy can carry 12% of its mass on switching - about the human rate -
and still argmax into a switch only 4% of the time. `BattlePolicy.choose(logits, switch_tau)`
settles switch-vs-attack on the *summed* switch mass, then stays greedy inside the winning class,
so move choice is untouched and no randomness is introduced.

Logits are built by hand here: no model, no corpus, so the arithmetic is the assertion.
-inf marks an illegal action, matching what BattlePolicy.forward writes into the logits.
"""
import torch

from psrl.nets import layout as L
from psrl.nets.model import BattlePolicy

NEG = float("-inf")


def _row(moves, switches):
    """One row of logits: 8 move/tera entries then 6 switch entries; None means illegal."""
    v = [NEG if x is None else x for x in moves] + [NEG if x is None else x for x in switches]
    assert len(v) == L.N_ACTIONS, f"built {len(v)} actions, layout says {L.N_ACTIONS}"
    return torch.tensor([v])


def _switch_mass(row):
    return float(torch.softmax(row, -1)[0, L.SWITCH_OFFSET:].sum())


def test_tau_none_is_exactly_plain_argmax():
    row = _row([2.0, 1.0, None, None, None, None, None, None], [0.5] * 5 + [None])
    assert int(BattlePolicy.choose(row)) == int(row.argmax(-1))


def test_split_switch_vote_wins_on_summed_mass():
    # one move at p~0.45 against five switches at p~0.11 each: argmax takes the move even
    # though switching collectively holds more mass. This is the bug the rule exists to fix.
    row = _row([1.6, None, None, None, None, None, None, None], [0.2] * 5 + [None])
    assert _switch_mass(row) > 0.5
    assert int(row.argmax(-1)) < L.SWITCH_OFFSET
    assert int(BattlePolicy.choose(row, 0.30)) >= L.SWITCH_OFFSET


def test_greedy_within_the_chosen_class():
    row = _row([1.6, None, None, None, None, None, None, None], [0.1, 0.9, 0.2, 0.1, 0.1, None])
    assert int(BattlePolicy.choose(row, 0.30)) == L.SWITCH_OFFSET + 1


def test_low_switch_mass_keeps_attacking():
    row = _row([3.0, None, None, None, None, None, None, None], [0.0] * 5 + [None])
    assert _switch_mass(row) < 0.30
    assert int(BattlePolicy.choose(row, 0.30)) == 0


def test_move_choice_is_untouched_when_not_switching():
    row = _row([0.5, 2.0, 1.0, None, None, None, None, None], [None] * 6)
    assert int(BattlePolicy.choose(row, 0.30)) == int(row.argmax(-1)) == 1


def test_forced_switch_has_no_legal_move():
    row = _row([None] * 8, [0.1, 0.4, None, None, None, None])
    assert int(BattlePolicy.choose(row, 0.30)) == L.SWITCH_OFFSET + 1


def test_a_trapped_pokemon_still_attacks():
    row = _row([0.3, 0.9, None, None, None, None, None, None], [None] * 6)
    assert int(BattlePolicy.choose(row, 0.30)) == 1


def test_never_picks_an_illegal_action():
    rows = [_row([1.6, None, None, None, None, None, None, None], [0.2] * 5 + [None]),
            _row([None] * 8, [0.1, 0.4, None, None, None, None]),
            _row([0.3, 0.9, None, None, None, None, None, None], [None] * 6)]
    for row in rows:
        for tau in (None, 0.30, 0.50):
            a = int(BattlePolicy.choose(row, tau))
            assert row[0, a] != NEG, f"chose illegal action {a} at tau={tau}"


def test_shape_is_preserved_on_a_batch():
    a = _row([1.6, None, None, None, None, None, None, None], [0.2] * 5 + [None])
    b = _row([3.0, None, None, None, None, None, None, None], [0.0] * 5 + [None])
    batch = torch.cat([a, b]).view(2, 1, L.N_ACTIONS)
    out = BattlePolicy.choose(batch, 0.30)
    assert tuple(out.shape) == (2, 1)
    assert int(out[0, 0]) >= L.SWITCH_OFFSET and int(out[1, 0]) == 0


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
