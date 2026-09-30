#!/usr/bin/env python3
"""Pad a narrower checkpoint to the current scalar layout so it loads unchanged.

New scalars are always appended after the existing ones, so they land at the end of the trunk's
input (psrl/nets/model.py: [mons, field, global scalars]). Zero weights for them leave the
network's outputs on the original inputs exactly as they were - measured at 0.0 max difference
over 1,694 decisions when 360 was padded to 443.

The source width is read from the checkpoint's own `layout` field, because more than one older
width is now in circulation: 360 pre-Phase-6, 443 from Phase 6 onward. Do not try to derive it
from `trunk.0.weight`, whose columns are [per-mon features, field embeddings, global scalars] -
only the scalar tail grows. OLD_SCALARS is the fallback for checkpoints written before the
`layout` field existed.

    uv run python tools/pad_checkpoint.py runs/bc-v4/bc.pt runs/bc-v4-pad.pt
"""
import sys
from pathlib import Path

import torch

from psrl.nets import layout as L

OLD_SCALARS = 360      # fallback only: checkpoints predating the `layout` field


def main(src: str, dst: str) -> None:
    out = Path(dst)
    if out.exists():
        sys.exit(f"{out} exists; not overwriting")
    ck = torch.load(src, map_location="cpu")
    old = int(ck.get("layout", {}).get("scalars", OLD_SCALARS))
    grow = L.N_SCALARS - old
    if grow < 0:
        sys.exit(f"{src} has {old} scalars, wider than the current layout's {L.N_SCALARS}; refusing")
    if grow == 0:
        sys.exit(f"{src} already has {old} scalars, matching the current layout; nothing to pad")
    w = ck["model"]["trunk.0.weight"]
    ck["model"]["trunk.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], grow)], dim=1)
    ck["layout"] = {"ids": L.N_IDS, "scalars": L.N_SCALARS, "actions": L.N_ACTIONS}
    ck["padded_from"] = src
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    print(f"{src} -> {out}: {old} -> {L.N_SCALARS} scalars, trunk.0.weight "
          f"{tuple(w.shape)} -> {tuple(ck['model']['trunk.0.weight'].shape)} (+{grow} columns)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
