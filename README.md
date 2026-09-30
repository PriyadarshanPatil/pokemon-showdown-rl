# psrl — reinforcement learning on Pokémon Showdown

Behavioural cloning and self-play RL for Gen 9 Random Battle, trained on a corpus of 964,244
human replays and evaluated head-to-head and on the live Showdown ladder.

The simulator is the real one: battles run inside a
[pokemon-showdown](https://github.com/smogon/pokemon-showdown) checkout, driven synchronously so a batch of
battles steps deterministically from a seed. Python owns training, Node owns the simulator and
the observation encoder, and they talk over length-prefixed binary frames with no JSON in the
hot loop.

## What it found

The headline result is negative, and it is the point of the repo.

| model | vs `bc-v12` | |
|---|---|---|
| **`bc-v12`** — replays ≥1800, 4.1M params | reference | best |
| `bc-v13` — ≥1800, 9.0M params | −0.74 pts | capacity exhausted |
| `bc-v11` — all data, 1.14M params | −5.24 pts | weaker demonstrators |
| `league-v3` — self-play from `bc-v12` | −6.56 pts | self-play degrades it |
| `bc-v14` — ≥2000, 4.1M params | −8.25 pts | too little data |

`bc-v12` is a local optimum: more capacity, stronger-but-fewer demonstrators, weaker-but-more
numerous ones, and self-play all lose to it. Three axes closed:

- **Capacity** is matched to data. 1.14M underfits (train accuracy *below* val), 4.1M balances,
  9.0M overfits and loses.
- **Demonstrator quality** helps until it costs volume. ≥1800 gained +4.94 points over all-data;
  ≥2000 lost 8.25, because it halves the corpus.
- **Self-play is closed as constructed.** PPO gained +0.37 and +0.44; league self-play lost 6.56.
  The mechanism: self-play against an opponent that never pivots never punishes staying in, so
  switch mass drifts away from the 13.49% human anchor. The moment training leaves the replay
  corpus the opponent stops being human — and the ladder is made of humans.

There is **no demonstrated path to 1800 Elo** here. Imitation caps near its demonstrators,
realistically 1400–1500. Getting past them needs an opponent distribution that *stays* human.

Two instruments proved actively misleading and are documented as such: the scripted baselines
went 0 for 6 at ranking checkpoints, and league's own `pool init:NN%` reported +108 Elo where
head-to-head measured −46. Judge with `eval_h2h`, both orientations, n≈2000, and check the
mirror gap.

## Setup

Requires Python ≥3.13, Node ≥22.5 (for `node:sqlite`), [uv](https://docs.astral.sh/uv/), and a
built `pokemon-showdown` checkout **beside this repo**:

```sh
git clone https://github.com/smogon/pokemon-showdown
cd pokemon-showdown && npm install && node build && cd -

git clone https://github.com/PriyadarshanPatil/pokemon-showdown-rl
cd pokemon-showdown-rl && uv sync
```

`node/src/ps_dir.js` resolves the checkout: `$PS_DIR` when set, otherwise the sibling directory.
There are no npm dependencies — the Node side uses only builtins and the simulator.

The gate suite runs immediately after this and needs nothing else. Training does not: the replay
corpus and all checkpoints are gitignored, so see the note at the top of the next section.

## Verify

Every gate has a recorded expected value, and the suite is the definition of "working".

```sh
npm test                 # all gates (tools/run_gates.sh)
npm run test:encode      # observation golden + opponent-information leakage audit
npm run test:baselines   # scripted ladder: expect 50.0 / 1.5 / 2.0 / 50.0 / 39.3 / 50.0
uv run ruff check .
```

`verify_baselines.js` prints those six numbers but never exits non-zero, so a green suite alone
does not prove they held — read them.

## Train and evaluate

**None of this runs from a fresh clone without building the corpus first.** No checkpoints and no
replay data are committed — `runs/*.pt` is hundreds of MB and `data/replays.db` is 3.7 GB. Nothing
is missing from the repo; both are regenerable, and the crawler is polite and rate-limited, which
is why it is slow:

```sh
uv run python tools/crawl_replays.py index      # then: sample, fetch — ~2 days for 964k replays
```

The `runs/bc-v*.pt` paths below are the author's, kept so the commands match what produced the
published numbers. Substitute your own checkpoint names.

```sh
# Materialise encoded sequences once (~40 min, ~45 GiB at this filter), then train from cache.
# build_cache prints the directory it wrote; its suffix is the encoder fingerprint.
uv run python tools/build_cache.py --min-rating 1800 --loaders 8
uv run python -m psrl.train.bc --name bc-v12 --min-rating 1800 --seed 0 \
    --cache data/cache/r1800-<fingerprint>

# Head-to-head, both orientations on identical seeds. This is the only trustworthy comparison.
uv run python -m psrl.train.eval_h2h --checkpoint-a runs/bc-v12/bc.pt \
    --checkpoint-b runs/bc-v11/bc.pt --name v12-vs-v11 --battles 2000

# Self-play (both reproduce the documented non-results).
uv run python -m psrl.train.ppo --init runs/bc-v12/bc.pt --name ppo-v12 --seed 0
uv run python -m psrl.train.league --init runs/bc-v12/bc.pt --name league-v3 --seed 0

# Live ladder. Needs a Showdown account; put PS_USERNAME and PS_PASSWORD in a .env you create
# yourself (.env is gitignored and no credentials are committed). Never pass them inline.
set -a; source .env; set +a
uv run python -m psrl.ladder.run --checkpoint runs/bc-v12/bc.pt \
    --username "$PS_USERNAME" --password "$PS_PASSWORD" --ladder --max-games 100
```

## Layout

```
psrl/nets/       policy/value network (GRU over a non-Markov observation) and its layout
psrl/env/        binary bridge to the Node worker, and the vectorised env over several workers
psrl/data/       replay loader, sequence cache and the cache fingerprint
psrl/train/      bc, ppo, league, eval_h2h, eval_policy
psrl/ladder/     live Showdown client, protocol parser and agent
node/src/        synchronous simulator, observation encoder, set and item posteriors
node/test/       verification gates with recorded expected values
tools/           corpus crawler, cache builder, benchmarks, diagnostics
```

The per-phase research notes are not published. Code comments cite them by filename
(`docs/PHASE7.md`, `docs/CODEBASE_NOTES.md`) because the published code is byte-identical to the
working code; those files are the author's working record and stay private. Every number quoted
above, and every gate's expected value, is in this README or in the gate output itself.

Observation: 110 categorical ids + 476 scalars + a 14-action legality mask, per player, emitted
by `node/src/encode.js` and mirrored in `psrl/nets/layout.py`. The two must agree; the worker
asserts it at startup. The encoder is audited for leakage — every opponent fact it emits must be
derivable from that player's own protocol channel, and `verify_encode.js` checks 196,740 of them
per run and can be made to fail on a planted leak.

## Reproducibility

Every run records its commit SHA, full config and seed to `run_meta.json`. Results are reported
across seeds with the spread; single-seed numbers are not results. Checkpoints, replay databases,
run directories and the sequence cache are gitignored and regenerable — the corpus with
`tools/crawl_replays.py`, the cache with `tools/build_cache.py`.

**Some code here looks removable and is not.** Several files have no callers and two functions
look like near-duplicates, but each is load-bearing for a recorded gate value or an open issue —
`effectiveness()` exists twice on purpose, and the two versions rank moves differently. Run the
full gate suite before and after any cleanup, and check the six baseline percentages by eye,
because `verify_baselines.js` prints them but never exits non-zero.

## Contributing

`main` is protected: it takes no direct pushes, so changes arrive by pull request. Fork or clone
freely, branch, and open a PR. Two things a reviewer will check first, because they are the ways
this codebase breaks quietly:

- The gate suite passes, and the six baseline percentages are unchanged at
  50.0 / 1.5 / 2.0 / 50.0 / 39.3 / 50.0. `verify_baselines.js` prints them but never exits
  non-zero, so a green suite alone proves nothing about them.
- New observation scalars are **appended**, never inserted, and no existing scalar's value
  changes. Padded checkpoints reinterpret any column that moves, so an insertion silently
  invalidates every earlier model.

## License

MIT. Pokémon Showdown is a separate project under its own license; this repo contains none of its
code and expects a local checkout.
