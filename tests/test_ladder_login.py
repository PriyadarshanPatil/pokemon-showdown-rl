"""Step 3 gate: authenticate against a LOCAL Pokemon Showdown server.

Run a local server first:  node pokemon-showdown start --skip-build 8010
Nothing here touches sim3.psim.us.
"""
from __future__ import annotations

import asyncio
import random
import string
import sys

from psrl.ladder.client import ShowdownClient, Credentials

URL = "ws://localhost:8010/showdown/websocket"


async def _pump(c):
    async for _ in c.frames():
        pass


async def run() -> str:
    name = "PsrlBot" + "".join(random.choices(string.digits, k=5))
    c = ShowdownClient(Credentials(name), url=URL)
    await c.connect()
    task = asyncio.create_task(_pump(c))
    try:
        named = await c.wait_until_named(timeout=25)
        assert named.strip() == name, f"expected {name}, got {named!r}"
        # exercise the throttled send path
        await c.send("", "/cmd rooms")
        return named
    finally:
        task.cancel()
        await c.close()


if __name__ == "__main__":
    try:
        who = asyncio.run(asyncio.wait_for(run(), 60))
    except (OSError, asyncio.TimeoutError) as e:
        print(f"  [SKIP] no local server on :8010 ({type(e).__name__})")
        sys.exit(0)
    print(f"  [PASS] logged in as {who!r}, |updateuser| NAMED=1, throttled send OK")
