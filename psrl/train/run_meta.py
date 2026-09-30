"""Run traceability: an untraceable run is not a result (CLAUDE.md)."""
from __future__ import annotations

import json, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def write_run_meta(out: Path, a) -> None:
    """Record commit SHA, working-tree cleanliness, full config and start time."""
    def git(*c):
        return subprocess.run(("git", *c), cwd=REPO, capture_output=True, text=True).stdout.strip()
    (out / "run_meta.json").write_text(json.dumps(
        {"commit": git("rev-parse", "HEAD"),
         "dirty": bool(git("status", "--porcelain")),
         "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "config": vars(a)}, indent=1))
