"""A battle the server bins without a result must not stall the run.

Observed live on 2026-09-17, 39 games into a 40-game ladder run: a battle that had been open
about 45 minutes received `|expire|` and then `|deinit` and never a `|win|`. `expire` is in
run.py's _IGNORED set and the `|deinit|` handling only covered rooms already finished, so for an
active room both frames fell through, the room stayed in `self.rooms`, `finished` never advanced,
and the bot waited 45 minutes for a result that was never coming. Third stall of this family,
after db1d8eb (double search) and 4a34264 (finished rooms recreated by late frames).

Frames below are exactly what the server sent, from data/transcripts/session-1789670019.log.
"""
import asyncio

from psrl.ladder.run import Bot
from test_ladder_search import _Client, _Noop

ROOM = "battle-gen9randombattle-2683127003"


def _run(frames, max_games=1, ladder=True):
    client = _Client(["|updateuser| ladderbot|1|1|{}", *frames])
    bot = Bot(client, _Noop(), _Noop(), "gen9randombattle", max_games,
              ladder=ladder, dry_run=False)
    asyncio.run(bot._pump())
    searches = sum(1 for m in client.sent if m == "|/search gen9randombattle")
    return bot, client, searches


def _expired_game():
    return [f">{ROOM}\n|init|battle\n|title|ladderbot vs. iyglufciyrciyc",
            f">{ROOM}\n|expire|",
            f">{ROOM}\n|deinit"]


def test_an_expired_battle_is_dropped_not_held():
    bot, _, _ = _run(_expired_game())
    assert bot.rooms == {}, "the expired room must not stay open"


def test_an_expired_battle_is_not_counted_as_a_game():
    bot, _, _ = _run(_expired_game())
    assert bot.finished == 0
    assert dict(bot.record) == {}, "a voided battle is neither a win nor a loss"


def test_the_search_is_handed_back_so_a_replacement_is_played():
    # one search for the battle that expired, then another for its replacement, so
    # --max-games keeps meaning completed games
    _, _, searches = _run(_expired_game())
    assert searches == 2


def test_the_bot_leaves_the_expired_room():
    _, client, _ = _run(_expired_game())
    assert f"{ROOM}|/leave" in client.sent


def test_an_expired_finished_room_still_releases_the_final_exit():
    # the room finished normally but expired before sending its rating; `awaiting` has to
    # clear or the run can never reach its exit condition
    frames = [f">{ROOM}\n|init|battle\n|title|ladderbot vs. iyglufciyrciyc",
              f">{ROOM}\n|win|ladderbot",
              f">{ROOM}\n|expire|"]
    bot, _, _ = _run(frames, max_games=1, ladder=False)
    assert bot.finished == 1
    assert bot.awaiting == set(), "an expired room sends no rating, so it must release awaiting"


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
