'use strict';
/**
 * Empirical randbats table: P(item | species, moveset) plus the 0-IV stat rule, written to
 * node/src/item_table.json.
 *
 * Why sample rather than port. Items come from getPriorityItem/getItem in the generator
 * (data/random-battles/gen9/teams.ts, ~350 lines) branching on a MoveCounter, the chosen ability,
 * teraType, isLead and two randomChance(1,2) coin flips; the 0-IV rule is a separate predicate at
 * teams.ts:1583-1599. Porting either is a large surface for silent drift. Sampling the real
 * generator is exact by construction: the counts ARE its distribution, coin flips included.
 *
 * sets.json carries neither items nor IVs, and the foe's item is revealed in only ~16% of live
 * decisions - always as "gone". So before a reveal there is nothing else to go on.
 *
 * Keyed on the exact 4-move set the item is pinned 96% of the time; on species alone, 75%. At
 * encode time the moveset is partial, so item_posterior.js sums the rows whose moveset contains
 * the revealed moves - the same "condition on a closed enumerable pool" move set_posterior.js
 * already makes, but empirical rather than hypergeometric.
 *
 * Converged at 200k teams: 2,744 movesets over all 509 species, 0.26 MB, ~50s.
 *   node tools/build_item_table.js [teams] [seed]
 */
const fs = require('fs');
const path = require('path');

const PS = require('../node/src/ps_dir');
const { Teams, Dex } = require(`${PS}/dist/sim`);
const { toID } = require('../node/src/tracker');
const { setsKey } = require('../node/src/set_posterior');

const N_TEAMS = +process.argv[2] || 200000;
// PS accepts a gen5 seed as four comma-separated numbers; it must start with a digit.
const SEED = process.argv[3] || '2026,9,22,1';
const OUT = path.join(__dirname, '..', 'node', 'src', 'item_table.json');

// species -> movesetSignature -> {n, atk0, spe0, items:Map}
const tbl = new Map();
let mons = 0;

const t0 = Date.now();
const gen = Teams.getGenerator('gen9randombattle', SEED);
for (let i = 0; i < N_TEAMS; i++) {
	for (const m of gen.getTeam()) {
		if (!m.ivs) throw new Error('generated set has no ivs - the generator changed shape.');
		const sp = setsKey(toID(m.speciesId || m.species));
		const sig = m.moves.map(toID).sort().join('|');
		if (!tbl.has(sp)) tbl.set(sp, new Map());
		const byMoves = tbl.get(sp);
		if (!byMoves.has(sig)) byMoves.set(sig, { n: 0, atk0: 0, spe0: 0, items: new Map() });
		const r = byMoves.get(sig);
		r.n++;
		if (m.ivs.atk === 0) r.atk0++;
		if (m.ivs.spe === 0) r.spe0++;
		const it = toID(m.item);
		r.items.set(it, (r.items.get(it) || 0) + 1);
		mons++;
	}
}

// The 0-IV rule reads only the moveset, so within one moveset row it must be all-or-nothing.
// A split row would mean the rule depends on something we are not keying on.
let split = 0;
const sets = {};
let movesets = 0, thin = 0;
for (const [sp, byMoves] of [...tbl].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
	sets[sp] = {};
	for (const [sig, r] of [...byMoves].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
		if ((r.atk0 && r.atk0 !== r.n) || (r.spe0 && r.spe0 !== r.n)) split++;
		sets[sp][sig] = {
			n: r.n,
			atk0: r.atk0 / r.n,
			spe0: r.spe0 / r.n,
			items: Object.fromEntries([...r.items].sort((a, b) => b[1] - a[1])),
		};
		movesets++;
		if (r.n < 5) thin++;
	}
}

const idTables = require('../node/src/id_tables.json');
const missing = new Set();
for (const byMoves of Object.values(sets)) {
	for (const r of Object.values(byMoves)) {
		for (const it of Object.keys(r.items)) if (it && !(it in idTables.items)) missing.add(it);
	}
}
if (missing.size) {
	// Do NOT regenerate id_tables.json to fix this. It is frozen on purpose (docs/CODEBASE_NOTES.md,
	// "Do not delete"): rebuilding it from a newer checkout shifts every ability id after
	// `auraguard` and silently reindexes the embeddings of every trained checkpoint.
	throw new Error(`items absent from node/src/id_tables.json: ${[...missing].join(', ')}. ` +
		`id_tables.json is frozen - do not regenerate it. Either PS_DIR points at a newer checkout ` +
		`than the tables were built from, or these items are genuinely new and need handling by hand.`);
}

fs.writeFileSync(OUT, JSON.stringify({
	_meta: {
		gen: 9, teams: N_TEAMS, seed: SEED, mons,
		psDir: path.basename(PS), psVersion: Dex.version || null,
		generated: 'tools/build_item_table.js',
	},
	sets,
}));
const mb = (fs.statSync(OUT).size / 1048576).toFixed(2);
console.log(`${N_TEAMS} teams, ${mons} mons, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
console.log(`species=${Object.keys(sets).length} movesets=${movesets} thin(<5 samples)=${thin} ` +
	`split-IV rows=${split}`);
console.log(`wrote ${path.relative(process.cwd(), OUT)} (${mb} MB), seed "${SEED}"`);
