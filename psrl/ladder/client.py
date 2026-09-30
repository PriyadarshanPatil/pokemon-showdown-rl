"""Pokemon Showdown websocket client: connect, authenticate, throttle, route rooms.

All the IO and all the failure modes live here; parsing is in protocol.py and the agent
knows nothing about the wire.

Every session writes a transcript of raw bytes in and out. A live battle can then be
replayed offline through the same encoder, which is the only practical way to debug
something that happened once on a ladder three minutes ago.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import websockets

from .protocol import parse_frame, outbound

MAIN_WS = "wss://sim3.psim.us/showdown/websocket"
LOGIN_URL = "https://play.pokemonshowdown.com/api/login"
ASSERTION_URL = "https://play.pokemonshowdown.com/api/getassertion"
# server/users.ts: THROTTLE_DELAY = 600ms for a normal account (100 trusted, 25 for
# rank '*'). A hard client-side limiter, not a best-effort sleep.
THROTTLE_SECONDS = 0.6


@dataclass
class Credentials:
    username: str
    password: str | None = None   # None -> unregistered login via getassertion
    # Assertions are bound to a server id. Main is "showdown"; a local test server uses
    # whatever Config.serverid is set to (unset by default).
    serverid: str | None = None


class ShowdownClient:
    def __init__(self, creds: Credentials, url: str = MAIN_WS,
                 throttle: float = THROTTLE_SECONDS, transcript_dir: str | Path | None = None):
        self.creds = creds
        self.url = url
        self.throttle = throttle
        self.ws: websockets.ClientConnection | None = None
        self.named = asyncio.Event()
        self.username: str | None = None
        self._last_send = 0.0
        self._send_lock = asyncio.Lock()
        self._tx = None
        if transcript_dir:
            d = Path(transcript_dir)
            d.mkdir(parents=True, exist_ok=True)
            self._tx = open(d / f"session-{int(time.time())}.log", "a", encoding="utf-8")  # noqa: SIM115

    # ---- transport ----
    async def connect(self) -> None:
        self.ws = await websockets.connect(self.url, max_size=8 << 20)
        self._record("--", f"connected to {self.url}")

    async def close(self) -> None:
        if self.ws:
            await self.ws.close()
        if self._tx:
            self._tx.close()

    def _record(self, direction: str, text: str) -> None:
        if self._tx:
            self._tx.write(f"{time.time():.3f} {direction} {text}\n")
            self._tx.flush()

    async def send(self, room: str, text: str) -> None:
        """Throttled send. Serialised so concurrent battles cannot burst past the limit."""
        async with self._send_lock:
            wait = self._last_send + self.throttle - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            msg = outbound(room, text)
            await self.ws.send(msg)
            self._last_send = time.monotonic()
            self._record(">>", msg)

    async def frames(self):
        """Yield parsed frames, handling login transparently."""
        async for raw in self.ws:
            self._record("<<", raw)
            frame = parse_frame(raw)
            for msg in frame.messages:
                if msg.type == "challstr":
                    await self._login(msg.data)
                elif msg.type == "updateuser":
                    # |updateuser|USER|NAMED|AVATAR|SETTINGS
                    user, named = msg.parts[0], msg.parts[1]
                    if named == "1":
                        self.username = user.strip()
                        self.named.set()
            yield frame

    # ---- authentication ----
    async def _login(self, challstr: str) -> None:
        async with httpx.AsyncClient(timeout=30) as http:
            if self.creds.password:
                r = await http.post(LOGIN_URL, data={
                    "name": self.creds.username, "pass": self.creds.password,
                    "challstr": challstr,
                })
            else:
                # unregistered: no password, just claim the name
                params = {"userid": _to_id(self.creds.username), "challstr": challstr}
                if self.creds.serverid:
                    params["serverid"] = self.creds.serverid
                r = await http.get(ASSERTION_URL, params=params)
            body = r.text
        if self.creds.password:
            if not body.startswith("]"):
                raise RuntimeError(f"login failed: {body[:200]}")
            assertion = json.loads(body[1:])["assertion"]
        else:
            assertion = body.strip()
            if assertion.startswith(";"):
                raise RuntimeError(f"getassertion refused (name may be registered): {assertion[:200]}")
        await self.send("", f"/trn {self.creds.username},0,{assertion}")

    async def wait_until_named(self, timeout: float = 30.0) -> str:
        await asyncio.wait_for(self.named.wait(), timeout)
        return self.username


def _to_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())
