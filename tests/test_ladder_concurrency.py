"""Challenge mode must never have more than two battles in progress at once.

Challenges arrive as PMs, in the format observed on the main server on 2026-09-15. A
battle's room frame only arrives after the server starts it, so three challenges can all
land before any room exists.
"""
import asyncio

from psrl.ladder.run import Bot
from test_ladder_search import _Client, _Noop


def _challenge(who):
    return f"|pm| {who}| ladderbot|/challenge gen9randombattle|gen9randombattle|||"


def _accepts(raw_frames):
    client = _Client(["|updateuser| ladderbot|1|1|{}", *raw_frames])
    bot = Bot(client, _Noop(), _Noop(), "gen9randombattle", 10, ladder=False, dry_run=False)
    asyncio.run(bot._pump())
    return [m for m in client.sent if m.startswith("|/accept ")]


def test_at_most_two_battles_at_once():
    frames = [_challenge("alice"), _challenge("bob"), _challenge("carol")]
    assert _accepts(frames) == ["|/accept alice", "|/accept bob"]


def test_a_finished_battle_frees_a_slot():
    room = "battle-gen9randombattle-1"
    frames = [_challenge("alice"), _challenge("bob"), _challenge("carol"),
              f">{room}\n|init|battle\n|title|alice vs. ladderbot",
              f">{room}\n|win|alice",
              _challenge("dave")]
    assert _accepts(frames) == ["|/accept alice", "|/accept bob", "|/accept dave"]


# ---- ladder mode: --concurrency, capped at the same two battles ----

def _ladder(max_games, concurrency, raw_frames):
    client = _Client(["|updateuser| ladderbot|1|1|{}", *raw_frames])
    bot = Bot(client, _Noop(), _Noop(), "gen9randombattle", max_games, ladder=True,
              dry_run=False, concurrency=concurrency)
    asyncio.run(bot._pump())
    return sum(1 for m in client.sent if m == "|/search gen9randombattle")


def _two_rooms():
    """Two ladder games, the second starting before the first ends."""
    r1, r2 = "battle-gen9randombattle-1", "battle-gen9randombattle-2"
    return [
        '|updatesearch|{"searching":["gen9randombattle"],"games":null}',
        '|updatesearch|{"searching":[],"games":null}',
        f">{r1}\n|init|battle\n|title|opponent vs. ladderbot",
        '|updatesearch|{"searching":["gen9randombattle"],"games":null}',
        '|updatesearch|{"searching":[],"games":null}',
        f">{r2}\n|init|battle\n|title|other vs. ladderbot",
        f">{r1}\n|win|opponent",
        f">{r2}\n|win|ladderbot",
        f">{r1}\n|deinit",
        f">{r2}\n|deinit",
    ]


def test_ladder_plays_two_at_once_when_asked():
    assert _ladder(2, 2, _two_rooms()) == 2


def test_ladder_stays_one_at_a_time_by_default():
    assert _ladder(2, 1, _two_rooms()) == 1


def test_no_second_search_while_one_is_still_pending():
    # The server rejects a duplicate /search, and |updatesearch| empties before the battle
    # arrives, so a pending unmatched search must not look like a free slot.
    frames = ['|updatesearch|{"searching":["gen9randombattle"],"games":null}',
              '|updatesearch|{"searching":[],"games":null}']
    assert _ladder(4, 2, frames) == 1


def test_concurrency_is_capped_at_two():
    bot = Bot(_Client([]), _Noop(), _Noop(), "gen9randombattle", 1, ladder=True,
              dry_run=False, concurrency=9)
    assert bot.concurrency == 2


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
