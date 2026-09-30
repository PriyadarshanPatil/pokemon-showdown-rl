"""Step 5 gate: the ladder path must produce byte-identical observations.

Feeds each fixture's server-shaped inputs (protocol lines + |request|) through
encode_service and requires the result to match, exactly, what the audited local encode
path produced for the same battle. Hermetic - no network.

If this passes, "the ladder bot runs the same agent we measured offline" is a verified
claim. If it fails, we learn that here instead of on the ladder.

Build the fixture first:  node tools/export_ladder_fixture.js 200
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from psrl.ladder.agent import EncodeService

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "data" / "ladder_fixture.jsonl"


def main(limit: int | None = None) -> int:
    if not FIXTURE.exists():
        print(f"  [SKIP] no fixture at {FIXTURE}; run tools/export_ladder_fixture.js")
        return 0
    svc = EncodeService()
    battles = decisions = 0
    bad_ids = bad_scal = bad_mask = 0
    worst = 0.0
    first_fail = None
    try:
        with open(FIXTURE) as fh:
            for line in fh:
                rec = json.loads(line)
                svc.init_room(rec["room"], rec["perspective"])
                for t, step in enumerate(rec["steps"]):
                    svc.feed(rec["room"], step["lines"])
                    ids, scal, mask = svc.encode(rec["room"], step["request"])
                    exp_ids = np.asarray(step["ids"], dtype=np.int64)
                    exp_scal = np.asarray(step["scalars"], dtype=np.float32)
                    exp_mask = np.asarray(step["mask"], dtype=bool)
                    di = not np.array_equal(ids, exp_ids)
                    dm = not np.array_equal(mask, exp_mask)
                    delta = float(np.abs(scal - exp_scal).max()) if scal.size else 0.0
                    ds = delta > 0.0
                    worst = max(worst, delta)
                    bad_ids += di; bad_mask += dm; bad_scal += ds
                    if (di or dm or ds) and first_fail is None:
                        first_fail = (rec["room"], t, int(di), int(dm), delta)
                    decisions += 1
                svc.drop(rec["room"])
                battles += 1
                if limit and battles >= limit:
                    break
    finally:
        svc.close()

    print(f"  {battles} battles, {decisions:,} decision points")
    print(f"  id mismatches:     {bad_ids}")
    print(f"  mask mismatches:   {bad_mask}")
    print(f"  scalar mismatches: {bad_scal}   (max abs delta {worst:g})")
    if first_fail:
        print(f"  first failure: room={first_fail[0]} step={first_fail[1]} "
              f"ids={first_fail[2]} mask={first_fail[3]} scalarDelta={first_fail[4]:g}")
    ok = not (bad_ids or bad_mask or bad_scal)
    print(f"  [{'PASS' if ok else 'FAIL'}] ladder observations are byte-identical to the local path")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
