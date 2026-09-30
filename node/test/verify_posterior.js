'use strict';
/**
 * Phase 3 step 13 gate. Using the simulator's ground-truth opponent team (available to
 * the TEST only, never to the actor), check that as moves are revealed through p1's
 * channel the posterior:
 *   - never eliminates the true set (support stays correct)
 *   - concentrates on it (P(true) rises)
 *   - loses entropy
 */
const { toID } = require('../src/tracker');
const { posterior, speciesSets } = require('../src/set_posterior');
const { playBattle, randomPick } = require('./harness');

const N = parseInt(process.argv[2] || '300');
/** Indices of sets whose movepool covers all of the Pokemon's real moves. */
function truthSupport(speciesId, realMoves) {
	const sets = speciesSets(speciesId);
	const out = [];
	sets.forEach((s, i) => { if (realMoves.every(m => s.movepool.includes(m))) out.push(i); });
	return out;
}

// P(true) and entropy bucketed by how many opposing moves have been revealed
const byK = new Map();
let eliminated = 0, observations = 0, noSets = 0, noTruth = 0;

for (let ep = 0; ep < N; ep++) {
	const r = playBattle({
		tag: 'post', ep, rngSeed: ep * 22695477, pick: randomPick,
		onDecision: (i, ctx) => {
			if (i !== 0) return;
			const tr = ctx.tracker;
			const foeActive = tr.activeMon('p2');
			if (!foeActive || !foeActive.species) return;
			// ground truth, available to the TEST only - never to the actor
			const real = ctx.battle.battle.sides[1].pokemon
				.find(p => p.species.id === foeActive.species || p.baseSpecies.id === foeActive.species);
			const sets = speciesSets(foeActive.species);
            if (!sets.length) { noSets++; return; }
			if (!real) return;
			const realMoves = real.moveSlots.map(m => toID(m.id));
			const truth = truthSupport(foeActive.species, realMoves);
			if (!truth.length) { noTruth++; return; }
			const revealed = foeActive.moves.slice();
			const p = posterior(foeActive.species, { moves: revealed });
			const pTrue = truth.reduce((acc, k) => acc + (p.probs[k] || 0), 0);
			if (pTrue <= 0) eliminated++;
			observations++;
			const k = Math.min(4, revealed.length);
			if (!byK.has(k)) byK.set(k, { n: 0, pTrue: 0, H: 0, collapsed: 0 });
			const acc = byK.get(k);
			acc.n++; acc.pTrue += pTrue; acc.H += p.entropy;
			if (pTrue > 0.99) acc.collapsed++;
		},
	});
	r.battle.destroy();
}

console.log(`${N} battles | ${observations} posterior evaluations`);
console.log(`  species with no sets.json entry: ${noSets} | real moves outside every movepool: ${noTruth}\n`);
console.log('revealed moves |    n   | P(true set) | entropy | collapsed to truth');
const ks = [...byK.keys()].sort();
const rows = ks.map(k => { const a = byK.get(k); return { k, n: a.n, p: a.pTrue / a.n, H: a.H / a.n, c: a.collapsed / a.n }; });
for (const r of rows) {
	console.log(`      ${r.k}        | ${String(r.n).padStart(6)} |    ${r.p.toFixed(3)}    |  ${r.H.toFixed(3)}  |  ${(r.c * 100).toFixed(1)}%`);
}
const mono = (get) => rows.every((r, i) => i === 0 || get(r) >= get(rows[i - 1]) - 1e-9);
const antimono = (get) => rows.every((r, i) => i === 0 || get(r) <= get(rows[i - 1]) + 1e-9);
console.log('\nGATES');
const checks = [
	['true set never eliminated', eliminated === 0, eliminated],
	['P(true set) rises with evidence', mono(r => r.p), rows.map(r => r.p.toFixed(3)).join(' -> ')],
	['entropy falls with evidence', antimono(r => r.H), rows.map(r => r.H.toFixed(3)).join(' -> ')],
	['fully-revealed collapses to truth', (rows[rows.length - 1]?.c ?? 0) > 0.9, rows[rows.length - 1]?.c],
];
let ok = true;
for (const [name, passed, val] of checks) { console.log(`  [${passed ? 'PASS' : 'FAIL'}] ${name}  (${val})`); ok &= !!passed; }
process.exitCode = ok ? 0 : 1;
