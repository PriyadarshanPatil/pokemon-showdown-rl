"""Pokemon Showdown wire-frame parsing. Pure functions, no IO.

Server -> client frames look like::

    >ROOMID
    MESSAGE
    MESSAGE

`>ROOMID` and its newline are omitted for the lobby/global room. A MESSAGE cannot start
with `>`, so the presence of a room header is unambiguous. Messages beginning with `|`
are `|TYPE|DATA`; anything else is raw text meant for display.

Client -> server messages are ``ROOMID|TEXT`` with ROOMID optionally blank.
"""
from __future__ import annotations

from typing import NamedTuple

GLOBAL_ROOM = ""


class Message(NamedTuple):
    type: str          # "" for raw display text
    parts: list[str]   # fields after the type
    raw: str

    @property
    def data(self) -> str:
        return "|".join(self.parts)


class Frame(NamedTuple):
    room: str
    messages: list[Message]


def parse_message(line: str) -> Message:
    if not line.startswith("|"):
        return Message("", [line], line)
    # "|TYPE|A|B" -> type TYPE, parts [A, B]; a bare "|" is the spacer message
    fields = line[1:].split("|")
    return Message(fields[0], fields[1:], line)


def parse_frame(text: str) -> Frame:
    """One server frame -> (room, messages). Empty lines are dropped per the spec."""
    lines = text.split("\n")
    room = GLOBAL_ROOM
    if lines and lines[0].startswith(">"):
        room = lines[0][1:].strip()
        lines = lines[1:]
    return Frame(room, [parse_message(l) for l in lines if l != ""])


def outbound(room: str, text: str) -> str:
    """Client -> server: ROOMID|TEXT (ROOMID may be blank for global commands)."""
    return f"{room}|{text}"
