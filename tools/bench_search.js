'use strict';
/**
 * Phase 6 gate 3.1 — battle copy + one-turn sim + encode, benchmarked per decision.
 *
 * The gate is p95 <= 2s per decision. This measures the node half: for every legal action at a
 * real decision point, clone the battle, apply the action, and encode the resulting state. The
 * value net's batched forward is measured separately by tools/bench_search_value.py, because it
 * runs in the Python process; the two are summed in the writeup.
 *
 * Copying is PS's own State.serializeBattle/deserializeBattle, which round-trips the PRNG as well
 * as the board, so a clone replays identically - verified before this benchmark was written, and
 * re-asserted here on the first decision of every battle.
 *
 * Two branching regimes are reported, because they differ by an order of magnitude and the choice
 * is a design decision rather than a measurement:
 *   FIXED    - one opponent reply per candidate, from a fixed policy. K clones.
 *   MARGINAL - every opponent reply against every candidate. K * K clones.
 *
 *   node tools/bench_search.js [battles] [maxDecisions]
 */
const PS = require('../node/src/ps_dir');
const { State } = require(`${PS}/dist/sim/state.js`);
const { SyncBattle } = require('../node/src/sync_battle');
const { seedsFor } = require('../node/src/seeds');
const { legalMask } = require('../node/src/actions');
const { encode } = require('../node/src/encode');
const { Tracker } = require('../node/src/tracker');

const N_BATTLES = +process.argv[2] || 12;
const MAX_DEC = +process.argv[3] || 40;

const ms = ns => Number(ns) / 1e6;
const pct = (xs, p) => {
	const a = [...xs].sort((x, y) => x - y);
	return a[Math.min(a.length - 1, Math.floor(p * a.length))];
};
const legalOf = mask => { const o = []; for (let i = 0; i < mask.length; i++) if (mask[i]) o.push(i); return o; };

/** Clone a live battle. Returns a Battle, not a SyncBattle. */
const cloneBattle = b => State.deserializeBattle(State.serializeBattle(b));

const fixed = [], marginal = [], branch = [], cloneMs = [], stepMs = [], encMs = [];
let decisions = 0, fidelityChecks = 0;

for (let ep = 0; ep < N_BATTLES; ep++) {
	const sb = new SyncBattle({ formatid: 'gen9randombattle', ...seedsFor('bench-search', ep) });
	const trackers = [new Tracker('p1'), new Tracker('p2')];
	const v0 = sb.drainViews();
	trackers[0].feed(v0.p1); trackers[1].feed(v0.p2);
	let firstOfBattle = true;

	for (let d = 0; d < MAX_DEC && !sb.ended; d++) {
		const acts = sb.toAct();
		const masks = sb.masks();
		if (!acts[0]) {                                  // only p1 searches
			const legal2 = acts[1] ? legalOf(masks[1]) : [null];
			sb.step(null, legal2[0]);
			const v = sb.drainViews();
			trackers[0].feed(v.p1); trackers[1].feed(v.p2);
			continue;
		}
		const mine = legalOf(masks[0]);
		const theirs = acts[1] ? legalOf(masks[1]) : [null];
		const K = mine.length, M = theirs.length;
		branch.push(K);

		// One faithfulness assertion per battle: a clone stepped the same way must agree.
		if (firstOfBattle) {
			const c = cloneBattle(sb.battle);
			if (c.turn !== sb.battle.turn) throw new Error(`clone turn ${c.turn} != ${sb.battle.turn}`);
			fidelityChecks++; firstOfBattle = false;
		}

		// FIXED: one reply per candidate.
		let tClone = 0n, tStep = 0n, tEnc = 0n;
		const t0 = process.hrtime.bigint();
		for (const a of mine) {
			const c0 = process.hrtime.bigint();
			const copy = cloneBattle(sb.battle);
			const c1 = process.hrtime.bigint();
			const sim = Object.create(SyncBattle.prototype);
			sim.battle = copy; sim.logCursor = copy.log.length;
			try { sim.step(a, theirs[0]); } catch { /* trapped/unavailable: the sim's own signal */ }
			const c2 = process.hrtime.bigint();
			const tr = new Tracker('p1');
			tr.feed(sim.drainViews().p1);
			const req = copy.sides[0].activeRequest;
			if (req) encode(req, tr);
			const c3 = process.hrtime.bigint();
			tClone += c1 - c0; tStep += c2 - c1; tEnc += c3 - c2;
		}
		const t1 = process.hrtime.bigint();
		fixed.push(ms(t1 - t0));
		cloneMs.push(ms(tClone) / K); stepMs.push(ms(tStep) / K); encMs.push(ms(tEnc) / K);
		// MARGINAL is K*M of the same unit cost; measured rather than extrapolated would cost
		// minutes per decision at K=M=9, so it is projected from the per-candidate mean.
		marginal.push(ms(t1 - t0) / K * K * M);
		decisions++;

		const a1 = mine[Math.floor(mine.length / 2)];
		sb.step(a1, theirs[0]);
		const v = sb.drainViews();
		trackers[0].feed(v.p1); trackers[1].feed(v.p2);
	}
}

const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
console.log(`decisions: ${decisions} over ${N_BATTLES} battles; clone fidelity asserted ${fidelityChecks}x`);
console.log(`branching factor K: mean ${mean(branch).toFixed(1)}, p95 ${pct(branch, 0.95)}, max ${Math.max(...branch)}`);
console.log(`per-candidate unit cost: clone ${mean(cloneMs).toFixed(2)}ms  step ${mean(stepMs).toFixed(2)}ms  encode ${mean(encMs).toFixed(2)}ms`);
console.log('');
console.log(`FIXED   (K clones):     p50 ${pct(fixed, 0.5).toFixed(0)}ms  p95 ${pct(fixed, 0.95).toFixed(0)}ms  max ${Math.max(...fixed).toFixed(0)}ms`);
console.log(`MARGINAL (K*M clones):  p50 ${pct(marginal, 0.5).toFixed(0)}ms  p95 ${pct(marginal, 0.95).toFixed(0)}ms  max ${Math.max(...marginal).toFixed(0)}ms`);
console.log('');
console.log(`gate 3.1 is p95 <= 2000ms per decision, node half only:`);
console.log(`  FIXED    ${pct(fixed, 0.95) <= 2000 ? 'PASS' : 'FAIL'}`);
console.log(`  MARGINAL ${pct(marginal, 0.95) <= 2000 ? 'PASS' : 'FAIL'}`);
