#!/bin/bash
# Full verification suite. Every gate has a recorded expected value in docs/PHASE*.md.
set -u
cd "$(dirname "$0")/.."
fail=0
run() { echo; echo "### $1"; shift; "$@" 2>&1 | tail -"${TAIL:-6}"; [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "  ^^ FAILED"; fail=1; }; }
TAIL=5  run "Phase 0: reproducibility"      node tools/bench_sim.js repro gen9randombattle
TAIL=3  run "Step 5: sync == BattleStream (gen9)"  node node/test/verify_sync.js 200 gen9randombattle
TAIL=2  run "Step 5: sync == BattleStream (gen1)"  node node/test/verify_sync.js 150 gen1randombattle
TAIL=5  run "Step 6: action mask (gen9)"    node node/test/verify_actions.js 250 gen9randombattle
TAIL=4  run "Step 6: action mask (gen1)"    node node/test/verify_actions.js 200 gen1randombattle
TAIL=3  run "Step 7: encode golden + leakage" node node/test/verify_encode.js 60
TAIL=6  run "Step 13: set posterior"        node node/test/verify_posterior.js 200
TAIL=4  run "Step 2.9: damage estimate"     node node/test/verify_damage.js
TAIL=7  run "Step 10: baseline ladder"      node node/test/verify_baselines.js 200
TAIL=8  run "Step 9: env soak invariants"   uv run python tools/soak_env.py --episodes 2000
TAIL=8  run "Ladder: protocol parsing"      uv run python tests/test_protocol.py
TAIL=5  run "League: pool Elo"              uv run python tests/test_pool_elo.py
TAIL=6  run "Ladder: observation equivalence" uv run python tests/test_ladder_equivalence.py
TAIL=3  run "Ladder: one search per game"   uv run python tests/test_ladder_search.py
TAIL=7  run "Ladder: two-battle cap"        uv run python tests/test_ladder_concurrency.py
TAIL=4  run "Ladder: rating logging"        uv run python tests/test_ladder_rating.py
TAIL=6  run "Ladder: expired battles"       uv run python tests/test_ladder_void.py
TAIL=3  run "Eval: repeat/switch counters"  uv run python tests/test_eval_stats.py
TAIL=3  run "Phase 7: two-stage selection"  uv run python tests/test_choose.py
TAIL=8  run "Phase 7: head-to-head symmetry" uv run python tests/test_h2h.py
# These two need a local server: node pokemon-showdown start --skip-build 8010
# They SKIP cleanly when it is absent so the suite stays runnable offline.
TAIL=3  run "Ladder: local login"           uv run python tests/test_ladder_login.py
# The checkpoint path must match the CURRENT scalar layout; a layout change strands it silently
# until this gate runs. See docs/CODEBASE_NOTES.md.
TAIL=4  run "Ladder: local battle"          uv run python tests/test_ladder_challenge.py runs/bc-v10-pad476/bc.pt
echo; echo "==== $( [ $fail -eq 0 ] && echo 'ALL GATES PASS' || echo 'SOME GATES FAILED' ) ===="
exit $fail
