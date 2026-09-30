'use strict';
/** Phase 2 step 10 gate: the baseline ladder must be monotone and random-vs-random ~50%. */
const { policies } = require('../src/baselines');
const { playBattle } = require('./harness');
const { mulberry32, seedsFor } = require('../src/seeds');

const N = parseInt(process.argv[2] || '400');
/** One battle between two scripted policies; returns +1 if `a` won. */
function play(a, b, ep, swap) {
	const pol = swap ? [b, a] : [a, b];
	const rngs = [mulberry32(ep * 7919 + 1), mulberry32(ep * 104729 + 3)];
	const r = playBattle({
		tag: 'bl', ep, turnCap: 300,
		pick: (i, ctx) => policies[pol[i]](ctx.request, ctx.tracker, ctx.mask, rngs[i]),
	});
	const res = r.result;
	r.battle.destroy();
	return swap ? -res : res;
}

const names = (process.env.AGENTS || 'random,maxdamage,heuristic').split(',');
const results = [];
console.log(`${N} paired battles per matchup (each seed played from both sides)\n`);
const wilson = (w, n) => { const z = 1.96, p = w / n, d = 1 + z * z / n;
	const c = (p + z * z / (2 * n)) / d, m = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
	return [c - m, c + m]; };

for (let i = 0; i < names.length; i++) {
	for (let j = i; j < names.length; j++) {
		let wins = 0, ties = 0, games = 0;
		for (let ep = 0; ep < N; ep++) {
			for (const swap of [false, true]) {          // paired seeds: play both sides
				const r = play(names[i], names[j], ep, swap);
				games++;
				if (r > 0) wins++; else if (r === 0) ties++;
			}
		}
		const wr = wins / games;
		const [lo, hi] = wilson(wins, games);
		console.log(`${names[i].padEnd(10)} vs ${names[j].padEnd(10)}  ${(wr * 100).toFixed(1)}%  [${(lo * 100).toFixed(1)}, ${(hi * 100).toFixed(1)}]  (${games} games, ${ties} ties)`);
		results.push({ a: names[i], b: names[j], wins, ties, games });
	}
}

if (process.env.OUT) {
	require('fs').writeFileSync(process.env.OUT, JSON.stringify({ n: N, results }, null, 1));
	console.log(`\nresults -> ${process.env.OUT}`);
}
