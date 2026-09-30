"""Ladder mode must send one /search per game, whatever order the server's frames arrive in.

The match sequence is the order observed on the main server on 2026-09-15 (challenge game,
session transcript in data/transcripts/): three |updatesearch| frames with an empty search
list arrive ~0.1s before the matched battle's first room frame. After the bot leaves a finished
battle, that room still sends its rating (|raw|) and |deinit| frames (ladder run, same day).
"""
import asyncio

from psrl.ladder.protocol import outbound, parse_frame
from psrl.ladder.run import Bot


def _game(room):
    games = '{"%s":"[Gen 9] Random Battle"}' % room
    return [
        '|updatesearch|{"searching":["gen9randombattle"],"games":null}',
        '|updatesearch|{"searching":[],"games":null}',
        '|updatesearch|{"searching":[],"games":%s}' % games,
        '|updatesearch|{"searching":[],"games":%s}' % games,
        f">{room}\n|init|battle\n|title|opponent vs. ladderbot",
        f">{room}\n|win|opponent",
        '|updatesearch|{"searching":[],"games":null}',
        f">{room}\n|raw|ladderbot's rating: 1000 &rarr; <strong>1000</strong>",
        f">{room}\n|deinit",
    ]


class _Client:
    """Replays raw server frames and records what the bot sends. Already logged in."""

    def __init__(self, raw_frames):
        self.raw_frames = raw_frames
        self.named = asyncio.Event()
        self.named.set()
        self.username = "ladderbot"
        self.sent = []

    async def frames(self):
        for raw in self.raw_frames:
            yield parse_frame(raw)

    async def send(self, room, text):
        self.sent.append(outbound(room, text))


class _Noop:
    def drop(self, room): pass
    def reset(self, room): pass


def _searches(max_games, raw_frames):
    client = _Client(["|updateuser| ladderbot|1|1|{}", *raw_frames])
    bot = Bot(client, _Noop(), _Noop(), "gen9randombattle", max_games, ladder=True, dry_run=False)
    asyncio.run(bot._pump())
    return sum(1 for m in client.sent if m == "|/search gen9randombattle")


def test_one_search_for_one_game():
    assert _searches(1, _game("battle-gen9randombattle-1")) == 1


def test_one_search_per_game():
    frames = _game("battle-gen9randombattle-1") + _game("battle-gen9randombattle-2")
    assert _searches(2, frames) == 2


def test_searches_again_after_a_finished_room_closes():
    frames = _game("battle-gen9randombattle-1") + _game("battle-gen9randombattle-2")
    assert _searches(3, frames) == 3


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
