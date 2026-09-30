"""Table-driven parser tests, using the PROTOCOL.md examples verbatim."""
from psrl.ladder.protocol import GLOBAL_ROOM, parse_frame, parse_message, outbound


def test_examples_from_protocol_md():
    frame = parse_frame(
        "|j| Some dude\n"
        "|c|@Moderator|hi!\n"
        "|c| Some dude|you suck and i hate you!\n"
        "Some dude was banned by Moderator.\n"
        "|l| Some dude\n"
        "|b|battle-ou-12| Cool guy|@Moderator"
    )
    assert frame.room == GLOBAL_ROOM
    assert [m.type for m in frame.messages] == ["j", "c", "c", "", "l", "b"]
    assert frame.messages[1].parts == ["@Moderator", "hi!"]
    # raw display text keeps its whole line and has an empty type
    assert frame.messages[3].type == ""
    assert frame.messages[3].parts == ["Some dude was banned by Moderator."]
    assert frame.messages[5].parts == ["battle-ou-12", " Cool guy", "@Moderator"]


def test_room_header_and_blank_lines():
    f = parse_frame(">battle-gen9randombattle-42\n|turn|3\n\n|upkeep")
    assert f.room == "battle-gen9randombattle-42"
    assert [m.type for m in f.messages] == ["turn", "upkeep"]


def test_bare_spacer_and_split_markers():
    assert parse_message("|").type == ""
    assert parse_message("|").parts == []
    # request JSON contains no pipes at the top level, so parts[0] is the whole payload
    m = parse_message('|request|{"active":[{"moves":[]}],"rqid":7}')
    assert m.type == "request"
    assert m.data == '{"active":[{"moves":[]}],"rqid":7}'


def test_challstr_keeps_its_pipes():
    # challstr famously contains | characters; data must rejoin them losslessly
    m = parse_message("|challstr|4|abcdef0123|deadbeef")
    assert m.type == "challstr"
    assert m.data == "4|abcdef0123|deadbeef"


def test_outbound():
    assert outbound("", "/trn Bot,0,ASSERT") == "|/trn Bot,0,ASSERT"
    assert outbound("battle-x-1", "/choose move 1|7") == "battle-x-1|/choose move 1|7"


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
