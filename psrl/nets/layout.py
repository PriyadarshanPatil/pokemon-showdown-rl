"""Observation layout, mirrored from node/src/encode.js.

If encode.js changes, this must change with it. `assert_matches` is called by the model
against what the worker actually reports, so a drift fails loudly instead of silently
feeding the wrong embedding table.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLES = json.loads((REPO / "node" / "src" / "id_tables.json").read_text())

N_MON = 6
N_SIDE = 2
IDS_PER_MON = 9          # species, item, ability, teraType, status, move1..4
SCAL_PER_MON = 17        # 6 flags + 7 boosts + 4 pp
N_IDS = N_SIDE * N_MON * IDS_PER_MON + 2
N_SCALARS = 476   # 310 base + 32 move feats + 6 matchup + 12 posterior + 19 last turn + 60 stats
                  # + 4 damage + 24 tier-1 (6 incoming, 7 KO flags, 1 outspeed, 6 hazard, 4 accuracy)
                  # + 9 tier-2 (6 locked-move incoming, 1 choice prob, 1 gated on a repeat,
                  #   1 inferred-item confidence)
N_ACTIONS = 14
# Action layout, mirroring node/src/actions.js: 0-3 moves, 4-7 move+tera, 8-13 switches.
# Both sides must agree; node/src/encode.js warns that the mapping breaks silently otherwise.
TERA_OFFSET = 4
SWITCH_OFFSET = 8

# offset within a Pokemon's 9 ids -> which embedding table it indexes
MON_FIELDS = [
    ("species", 0), ("items", 1), ("abilities", 2), ("types", 3), ("statuses", 4),
    ("moves", 5), ("moves", 6), ("moves", 7), ("moves", 8),
]
FIELD_TABLES = ["species", "items", "abilities", "types", "statuses", "moves",
                "weathers", "terrains"]
SIZES = {k: len(TABLES[k]) for k in FIELD_TABLES}


def assert_matches(n_ids: int, n_scalars: int, n_actions: int) -> None:
    if (n_ids, n_scalars, n_actions) != (N_IDS, N_SCALARS, N_ACTIONS):
        raise RuntimeError(
            f"layout drift: worker reports ids={n_ids} scalars={n_scalars} "
            f"actions={n_actions}, python expects {N_IDS}/{N_SCALARS}/{N_ACTIONS}. "
            f"Regenerate id tables and update psrl/nets/layout.py."
        )
