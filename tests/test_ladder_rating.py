"""The ladder rating is logged after every game, including the last one.

A finished room reports both players' ratings as |raw| text, and those frames arrive after
the bot has already left the room (main server, 2026-09-15; see data/transcripts/). The
lines below are real, copied from those transcripts.
"""
import asyncio

from psrl.ladder.run import Bot
from test_ladder_search import _Client, _Noop

ROOM = "battle-gen9randombattle-1"
OURS = "ladderbot's rating: 1000 &rarr; <strong>1040</strong><br />(+40 for winning)"
THEIRS = "sadlord's rating: 1101 &rarr; <strong>1074</strong><br />(-27 for losing)"


def _play(rating_lines, max_games=1):
    frames = [f">{ROOM}\n|init|battle\n|title|sadlord vs. ladderbot",
              f">{ROOM}\n|win|ladderbot"]
    frames += [f">{ROOM}\n|raw|{line}" for line in rating_lines]
    frames.append(f">{ROOM}\n|deinit")
    client = _Client(["|updateuser| ladderbot|1|1|{}", *frames])
    bot = Bot(client, _Noop(), _Noop(), "gen9randombattle", max_games,
              ladder=False, dry_run=False)
    asyncio.run(bot._pump())
    return bot


def test_logs_our_new_rating():
    assert _play([THEIRS, OURS]).ratings == [(1000, 1040)]


def test_ignores_the_opponents_rating():
    assert _play([THEIRS]).ratings == []


def test_the_last_games_rating_is_not_lost():
    # The room finishes on the last game, so the old exit check returned before its rating
    # frame was ever read. The pump now waits for it.
    assert _play([OURS], max_games=1).ratings == [(1000, 1040)]


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
