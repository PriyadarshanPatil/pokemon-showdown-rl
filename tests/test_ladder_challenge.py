"""Step 6 gate: play a real battle end to end against a LOCAL server.

Starts nothing itself - run a local server first:
    node pokemon-showdown start --skip-build 8010

Spawns the bot in challenge mode and a trivial opponent that challenges it and answers
every request with `/choose default`. Exercises the parts the offline gate cannot:
room routing, rqid, the challenge handshake, and win/loss detection.
"""
from __future__ import annotations

import asyncio
import json
import random
import string
import sys
from pathlib import Path

from psrl.ladder.agent import EncodeService, LadderAgent
from psrl.ladder.client import Credentials, ShowdownClient
from psrl.ladder.run import Bot

URL = "ws://localhost:8010/showdown/websocket"
REPO = Path(__file__).resolve().parents[1]
rnd = lambda p: p + "".join(random.choices(string.digits, k=5))


async def opponent(name: str, target: str, fmt: str, games: int) -> None:
    """Challenges `target` and answers every request with the auto-choice."""
    c = ShowdownClient(Credentials(name), url=URL)
    await c.connect()
    played = 0
    async for frame in c.frames():
        if c.username and played == 0:
            played = 1
            await c.send("", f"/challenge {target}, {fmt}")
        for m in frame.messages:
            if frame.room.startswith("battle-") and m.type == "request":
                if m.data.strip() and not json.loads(m.data).get("wait"):
                    await c.send(frame.room, "/choose default")
            elif frame.room.startswith("battle-") and m.type in ("win", "tie"):
                await c.close()
                return


async def run(checkpoint: str, games: int = 1) -> dict:
    bot_name, opp_name = rnd("PsrlBot"), rnd("PsrlOpp")
    svc = EncodeService()
    agent = LadderAgent(checkpoint, device="cpu")
    client = ShowdownClient(Credentials(bot_name), url=URL)
    bot = Bot(client, agent, svc, "gen9randombattle", games, ladder=False, dry_run=False)
    bot_task = asyncio.create_task(bot.run())
    try:
        await asyncio.wait_for(client.named.wait(), 30)
        await asyncio.sleep(1.0)
        opp = asyncio.create_task(opponent(opp_name, bot_name, "gen9randombattle", games))
        # Surface bot exceptions instead of letting the task swallow them - an unraised
        # error in the bot looks identical to a stalled battle.
        deadline = asyncio.get_running_loop().time() + 180
        while bot.finished < games and asyncio.get_running_loop().time() < deadline:
            if bot_task.done():
                exc = bot_task.exception()
                if exc:
                    raise exc
                break
            await asyncio.sleep(0.5)
        opp.cancel()
    finally:
        bot_task.cancel()
        await client.close()
        svc.close()
    return dict(bot.record)


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "runs" / "bc-v4" / "bc.pt")
    try:
        rec = asyncio.run(run(ck, games=int(sys.argv[2]) if len(sys.argv)>2 else 1))
    except OSError as e:
        print(f"  [SKIP] no local server on :8010 ({type(e).__name__}: {e})")
        sys.exit(0)
    except asyncio.TimeoutError:
        print("  [FAIL] connected and challenged, but the battle did not finish in time")
        sys.exit(1)
    total = sum(rec.values())
    print(f"  record: {rec}")
    print(f"  [{'PASS' if total >= 1 else 'FAIL'}] played {total} battle(s) to completion")
    sys.exit(0 if total >= 1 else 1)
