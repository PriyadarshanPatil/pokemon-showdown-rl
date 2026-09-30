"""Policy/value network: entity embeddings -> per-Pokemon encoder -> GRU -> masked heads.

The GRU matters: from one player's view the state is genuinely non-Markov (unrevealed
sets, sleep counters, choice lock), so the policy needs memory of the episode so far.

The value head is ASYMMETRIC-ready: pass `critic_extra` during training to give the critic
information the actor never sees (centralised training, decentralised execution). With
`critic_extra=None` it degrades to a symmetric critic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layout as L

NEG_INF = -1e9


class MonEncoder(nn.Module):
    """Embeds one Pokemon's 9 categorical ids + its 17 scalars into a vector."""

    def __init__(self, d_emb: int = 32, d_out: int = 128):
        super().__init__()
        self.species = nn.Embedding(L.SIZES["species"], d_emb)
        self.item = nn.Embedding(L.SIZES["items"], d_emb // 2)
        self.ability = nn.Embedding(L.SIZES["abilities"], d_emb // 2)
        self.tera = nn.Embedding(L.SIZES["types"], d_emb // 4)
        self.status = nn.Embedding(L.SIZES["statuses"], d_emb // 4)
        self.move = nn.Embedding(L.SIZES["moves"], d_emb)
        d_in = d_emb + d_emb // 2 + d_emb // 2 + d_emb // 4 + d_emb // 4 + 4 * d_emb + L.SCAL_PER_MON
        self.proj = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out))

    def forward(self, ids: torch.Tensor, scal: torch.Tensor) -> torch.Tensor:
        # ids: (..., 9)  scal: (..., 17)
        parts = [
            self.species(ids[..., 0]), self.item(ids[..., 1]), self.ability(ids[..., 2]),
            self.tera(ids[..., 3]), self.status(ids[..., 4]),
            self.move(ids[..., 5]), self.move(ids[..., 6]),
            self.move(ids[..., 7]), self.move(ids[..., 8]),
            scal,
        ]
        return self.proj(torch.cat(parts, dim=-1))


class BattlePolicy(nn.Module):
    def __init__(self, d_emb: int = 32, d_mon: int = 128, d_hidden: int = 256,
                 critic_extra_dim: int = 0):
        super().__init__()
        self.mon = MonEncoder(d_emb, d_mon)
        self.weather = nn.Embedding(L.SIZES["weathers"], 16)
        self.terrain = nn.Embedding(L.SIZES["terrains"], 16)
        n_global_scalars = L.N_SCALARS - L.N_SIDE * L.N_MON * L.SCAL_PER_MON
        d_trunk = L.N_SIDE * L.N_MON * d_mon + 32 + n_global_scalars
        self.trunk = nn.Sequential(nn.Linear(d_trunk, d_hidden), nn.ReLU(),
                                   nn.Linear(d_hidden, d_hidden), nn.ReLU())
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.pi = nn.Linear(d_hidden, L.N_ACTIONS)
        self.critic_extra_dim = critic_extra_dim
        self.v = nn.Sequential(nn.Linear(d_hidden + critic_extra_dim, d_hidden), nn.ReLU(),
                               nn.Linear(d_hidden, 1))
        self.d_hidden = d_hidden
        self.n_global_scalars = n_global_scalars

    def _trunk(self, ids: torch.Tensor, scal: torch.Tensor) -> torch.Tensor:
        """ids (B,T,110) int64, scal (B,T,310) -> (B,T,d_hidden)"""
        B, T = ids.shape[:2]
        n_mon = L.N_SIDE * L.N_MON
        mon_ids = ids[..., : n_mon * L.IDS_PER_MON].reshape(B, T, n_mon, L.IDS_PER_MON)
        mon_scal = scal[..., : n_mon * L.SCAL_PER_MON].reshape(B, T, n_mon, L.SCAL_PER_MON)
        mons = self.mon(mon_ids, mon_scal).reshape(B, T, n_mon * self.mon.proj[-1].out_features)
        field = torch.cat([self.weather(ids[..., -2]), self.terrain(ids[..., -1])], dim=-1)
        gscal = scal[..., n_mon * L.SCAL_PER_MON:]
        return self.trunk(torch.cat([mons, field, gscal], dim=-1))

    def forward(self, ids, scal, mask, hidden=None, critic_extra=None):
        """ids (B,T,110) int64; scal (B,T,310); mask (B,T,14) bool."""
        x = self._trunk(ids, scal)
        core, hidden = self.gru(x, hidden)
        logits = self.pi(core)
        logits = logits.masked_fill(~mask, NEG_INF)
        vin = core if critic_extra is None else torch.cat([core, critic_extra], dim=-1)
        value = self.v(vin).squeeze(-1)
        return logits, value, hidden

    @staticmethod
    def dist(logits: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=logits)

    @staticmethod
    def choose(logits: torch.Tensor, switch_tau: float | None = None) -> torch.Tensor:
        """Greedy action per row; with `switch_tau`, settle switch-vs-attack on summed mass first.

        Argmax splits the switch vote across up to five targets while move probability
        concentrates on one, so a policy carrying 12% mass on switching - bc-v9, against humans'
        13.5% in the same states - argmaxes into a switch only 4% of the time. Deciding the class
        from the summed mass spends that mass instead of discarding it, and stays greedy inside
        the chosen class, so move choice is untouched and no randomness is introduced.

        `forward` already fills illegal actions with -inf, so this softmax mass is legal-only.
        """
        if switch_tau is None:
            return logits.argmax(-1)
        p = torch.softmax(logits, dim=-1)
        moves, switches = p[..., :L.SWITCH_OFFSET], p[..., L.SWITCH_OFFSET:]
        # a forced switch leaves no legal move; a trapped Pokemon leaves no legal switch
        take_switch = (switches.amax(-1) > 0) & (
            (moves.amax(-1) <= 0) | (switches.sum(-1) > switch_tau))
        return torch.where(take_switch, switches.argmax(-1) + L.SWITCH_OFFSET, moves.argmax(-1))

    @staticmethod
    def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Entropy over legal actions only; fully-masked rows contribute 0."""
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp() * mask
        ent = -(p * logp.masked_fill(~mask, 0.0)).sum(-1)
        return ent


def load_policy(path, dev, **overrides) -> "BattlePolicy":
    """Build a BattlePolicy at the width its checkpoint was trained at, then load it.

    Every loader used to construct `BattlePolicy()` with defaults, which silently assumed one
    width for all time. Checkpoints written before bc.py recorded `dims` carry none, and the
    default 32/128/256 is exactly what they were - so omitting the key is not a missing value,
    it is the right answer.
    """
    m, _dims, _ck = load_policy_dims(path, dev, **overrides)
    return m


def load_policy_dims(path, dev, **overrides):
    """As load_policy, but also returns the dims and the raw checkpoint.

    league.py and ppo.py build scratch modules (a frozen snapshot holder, a KL reference) that
    must match the learner's width, and constructing those at the default width is how a wide
    checkpoint silently fails to load.
    """
    ck = torch.load(path, map_location=dev)
    dims = dict(ck.get("dims") or {})
    dims.update(overrides)
    m = BattlePolicy(**dims).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, dims, ck
