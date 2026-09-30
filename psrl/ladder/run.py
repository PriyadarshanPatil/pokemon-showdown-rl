"""Play battles on a Pokemon Showdown server.

Challenge mode is the default; laddering requires an explicit --ladder and is bounded by
--max-games. Nothing here parses frames (protocol.py) or encodes observations
(agent.py) - this is the state machine that joins the two.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter

from .agent import EncodeService, LadderAgent
from .client import Credentials, ShowdownClient, MAIN_WS

# Room lines the tracker must not see as battle protocol, and that we never act on.
_IGNORED = {"init", "title", "j", "J", "l", "L", "c", "c:", "raw", "uhtml", "html",
            "n", "N", "b", "B", "chat", "join", "leave", "expire", "askreg",
            "updateuser", "popup", "pm", "queryresponse", "updatesearch",
            "updatechallenges", "formats", "customgroups", "challstr", "usercount"}


# Usernames on the wire carry a leading rank symbol (a space for regular users).
_RANKS = " +%@#*&~!^'"


def _strip_rank(name: str) -> str:
    return name[1:].strip() if name[:1] in _RANKS else name.strip()


# A finished ladder room reports both players' new ratings, e.g.
#   |raw|ladderbot's rating: 1000 &rarr; <strong>1040</strong><br />(+40 for winning)
_RATING = re.compile(r"^(?P<user>.+?)'s rating: (?P<old>\d+) &rarr; <strong>(?P<new>\d+)</strong>")


class BattleRoom:
    """Per-battle state. Perspective is only known once the first request arrives."""

    def __init__(self, room: str):
        self.room = room
        self.perspective: str | None = None
        self.pending: list[str] = []
        self.rqid: int | None = None
        self.result: str | None = None
        self.turns = 0


class Bot:
    def __init__(self, client: ShowdownClient, agent: LadderAgent, svc: EncodeService,
                 fmt: str, max_games: int, ladder: bool, dry_run: bool, concurrency: int = 1):
        self.c, self.agent, self.svc = client, agent, svc
        self.fmt, self.max_games, self.ladder, self.dry_run = fmt, max_games, ladder, dry_run
        # Standing rule: never more than two battles at once, whatever the caller asks for.
        self.concurrency = min(2, max(1, concurrency))
        self.rooms: dict[str, BattleRoom] = {}
        self.finished_rooms: set[str] = set()
        self.awaiting: set[str] = set()      # finished rooms whose rating has not arrived yet
        self.ratings: list[tuple[int, int]] = []
        self.record = Counter()
        self.finished = 0
        self.searching = False
        self.searches = 0
        self.accepted = 0

    # ---- battle handling ----
    async def _on_request(self, br: BattleRoom, payload: str) -> None:
        if not payload.strip():
            return
        req = json.loads(payload)
        if br.perspective is None:
            br.perspective = req["side"]["id"]
            self.svc.init_room(br.room, br.perspective)
            self.agent.reset(br.room)
        br.rqid = req.get("rqid")
        # Flush protocol lines first: the request describes the state AFTER them.
        self.svc.feed(br.room, br.pending)
        br.pending = []
        if req.get("wait"):
            return
        ids, scalars, mask = self.svc.encode(br.room, req)
        if not mask.any():
            return
        action = self.agent.act(br.room, ids, scalars, mask)
        choice = self.svc.choices[action]
        if self.dry_run:
            print(f"  [dry-run] {br.room} would choose: {choice}")
            return
        suffix = f"|{br.rqid}" if br.rqid is not None else ""
        await self.c.send(br.room, f"/choose {choice}{suffix}")

    async def _finish(self, br: BattleRoom, outcome: str) -> None:
        br.result = outcome
        self.record[outcome] += 1
        self.finished += 1
        print(f"  {br.room}: {outcome}  (W{self.record['win']}/L{self.record['loss']}/T{self.record['tie']}"
              f"  {self.finished}/{self.max_games})")
        self.svc.drop(br.room)
        self.agent.reset(br.room)
        self.rooms.pop(br.room, None)
        self.finished_rooms.add(br.room)
        self.awaiting.add(br.room)      # its rating |raw| only arrives after we leave
        await self.c.send(br.room, "/leave")

    async def _void(self, br: BattleRoom) -> None:
        """The server binned this battle with no result: |expire| then |deinit|, never a |win|.

        Seen live on 2026-09-17 on a battle that had been open ~45 minutes. Without this the room
        stays in `self.rooms`, `finished` never advances, and the run waits forever for a result
        that is not coming - the third stall of this family after `db1d8eb` and `4a34264`.
        Nothing counts toward the record, and the search is handed back so --max-games still
        means completed games.
        """
        print(f"  {br.room}: voided by the server (expired with no result)")
        self.svc.drop(br.room)
        self.agent.reset(br.room)
        self.rooms.pop(br.room, None)
        self.finished_rooms.add(br.room)
        self.searches = max(self.finished, self.searches - 1)
        self.searching = False
        await self.c.send(br.room, "/leave")

    async def _accept(self, who: str, fmt: str) -> None:
        if self.finished >= self.max_games:
            return
        if fmt and self.fmt and fmt.replace(" ", "").lower() != self.fmt.lower():
            print(f"  declining {who}: format {fmt!r} != {self.fmt!r}")
            return
        # Never more than two battles at once. Count accepts, not rooms: a battle's room
        # only arrives after the server starts it, so several challenges can land first.
        if self.accepted - self.finished >= 2:
            print(f"  declining {who}: two battles already in progress")
            return
        print(f"  accepting challenge from {who} ({fmt or 'unspecified'})")
        self.accepted += 1
        await self.c.send("", f"/accept {who}")

    def _note_rating(self, text: str) -> bool:
        """Record our own new ladder rating. Both players' lines arrive, so filter by name."""
        m = _RATING.match(text)
        if not m or m.group("user").strip().lower() != (self.c.username or "").strip().lower():
            return False
        old, new = int(m.group("old")), int(m.group("new"))
        self.ratings.append((old, new))
        print(f"  rating: {old} -> {new} ({new - old:+d})")
        return True

    # ---- main loop ----
    async def run(self) -> None:
        await self.c.connect()
        pump = asyncio.create_task(self._pump())
        try:
            await self.c.wait_until_named()
            print(f"  logged in as {self.c.username!r}")
            if not self.ladder:
                # /blockchallenges BLOCKS them; the inverse is /unblockchallenges.
                # Default is already unblocked, but be explicit in case the account is not.
                await self.c.send("", "/unblockchallenges")
            await pump
        finally:
            pump.cancel()

    async def _search_if_idle(self) -> None:
        # Must be named first: a guest cannot ladder, and _pump sees frames before
        # /trn completes, which otherwise fires a useless (and repeated) search.
        if not self.c.named.is_set():
            return
        # |updatesearch| empties "searching" ~0.1s before the matched battle's first frame,
        # so count instead. `pending` is the searches the server has not matched yet: the
        # server rejects a second /search while one is still pending, and two live battles
        # are the cap. At concurrency 1 this is exactly the old "searches == finished and no
        # rooms" rule, so default behaviour is unchanged.
        outstanding = self.searches - self.finished     # pending searches + live battles
        pending = outstanding - len(self.rooms)
        if (self.ladder and not self.searching and pending == 0
                and outstanding < self.concurrency and self.searches < self.max_games):
            self.searching = True
            self.searches += 1
            await self.c.send("", f"/search {self.fmt}")

    async def _pump(self) -> None:
        async for frame in self.c.frames():
            if self.finished >= self.max_games and not self.rooms and not self.awaiting:
                print(f"  done: {dict(self.record)}")
                return
            room = frame.room
            for msg in frame.messages:
                t = msg.type
                if room.startswith("battle-"):
                    if room in self.finished_rooms:
                        # Its rating |raw| and |deinit| still arrive after /leave; recreating
                        # the room here would block the next search and the final exit.
                        # The opponent's rating line can arrive first, so only our own line
                        # closes the room out; |deinit| is the backstop if ours never comes.
                        if t == "raw" and self._note_rating(msg.data):
                            self.awaiting.discard(room)
                        elif t in ("deinit", "expire"):
                            # an expiring room never sends a rating, so this has to release
                            # `awaiting` or the run cannot reach its final exit
                            self.awaiting.discard(room)
                        continue
                    br = self.rooms.setdefault(room, BattleRoom(room))
                    if t == "request":
                        await self._on_request(br, msg.data)
                    elif t == "error":
                        # [Invalid choice] / [Unavailable choice]: the server follows with
                        # a fresh |request|, which we will answer normally.
                        print(f"  {room} error: {msg.data[:120]}")
                    elif t in ("win", "tie"):
                        me = (self.c.username or "").strip().lower()
                        won = t == "win" and msg.parts and msg.parts[0].strip().lower() == me
                        await self._finish(br, "win" if won else ("tie" if t == "tie" else "loss"))
                        self.searching = False
                        await self._search_if_idle()
                    elif t in ("expire", "deinit"):
                        # `expire` is in _IGNORED, so before this both frames fell through and
                        # the room was never closed. Active rooms must be voided explicitly.
                        await self._void(br)
                        await self._search_if_idle()
                    elif t not in _IGNORED:
                        if t == "turn":
                            br.turns = int(msg.parts[0]) if msg.parts else br.turns
                        br.pending.append(msg.raw)
                elif t == "pm" and not self.ladder:
                    # Challenges arrive as PMs on this server build:
                    #   |pm| SENDER| RECEIVER|/challenge FORMAT|FORMAT|||
                    # (|updatechallenges| is handled below for builds that send it.)
                    if len(msg.parts) >= 3 and msg.parts[2].startswith("/challenge"):
                        fmt = msg.parts[3] if len(msg.parts) > 3 else ""
                        sender = _strip_rank(msg.parts[0])
                        if sender.lower() != (self.c.username or "").strip().lower():
                            await self._accept(sender, fmt)
                elif t == "updatechallenges" and not self.ladder:
                    data = json.loads(msg.data or "{}")
                    for who, fmt in (data.get("challengesFrom") or {}).items():
                        await self._accept(who, fmt)
                elif t == "updatesearch":
                    data = json.loads(msg.data or "{}")
                    self.searching = bool(data.get("searching"))
            if self.ladder:
                await self._search_if_idle()


async def main(a) -> None:
    # Results have to reach the log as they happen: the stop procedure watches for a result
    # line, and redirected stdout is block-buffered, which held a whole run's output until exit.
    sys.stdout.reconfigure(line_buffering=True)
    svc = EncodeService()
    agent = LadderAgent(a.checkpoint, device=a.device, greedy=not a.sample,
                        switch_tau=a.switch_tau)
    client = ShowdownClient(
        Credentials(a.username, a.password or None, serverid=a.serverid or None),
        url=a.url, transcript_dir=a.transcripts)
    bot = Bot(client, agent, svc, a.format, a.max_games, a.ladder, a.dry_run, a.concurrency)
    try:
        await bot.run()
    finally:
        await client.close()
        svc.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", default="")
    ap.add_argument("--serverid", default="")
    ap.add_argument("--url", default=MAIN_WS)
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--max-games", type=int, default=1)
    ap.add_argument("--ladder", action="store_true",
                    help="ladder instead of accepting challenges (explicit opt-in)")
    ap.add_argument("--dry-run", action="store_true", help="compute choices, never send")
    ap.add_argument("--sample", action="store_true", help="sample instead of argmax")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="ladder games to play at once; capped at 2")
    ap.add_argument("--switch-tau", type=float, default=None,
                    help="switch when the summed switch probability exceeds this; 0.30 matches "
                         "the human rate. Default is plain argmax, which discards that mass")
    asyncio.run(main(ap.parse_args()))
