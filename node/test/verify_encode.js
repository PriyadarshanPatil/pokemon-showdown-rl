'use strict';
/**
 * Phase 1 step 7 gates.
 *
 * 1. STABILITY - feature vectors must be byte-identical across runs for fixed seeds.
 *    Run with --update to regenerate the golden file after an intentional layout change.
 * 2. LEAKAGE - every opponent fact the encoder emits must literally have appeared in
 *    this player's own protocol channel. This is the check that keeps the actor honest;
 *    it is deliberately hard to fool, because it re-derives the allowed vocabulary from
 *    the raw text rather than trusting the tracker.
 */
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const { toID } = require('../src/tracker');
const { playBattle, randomPick } = require('./harness');
const { encode, N_IDS, N_SCALARS, N_BASE_SCALARS, IDS_PER_MON, SCAL_PER_MON, N_MON, TABLES } = require('../src/encode');
const { statAt } = require('../src/damage');
const PS = require('../src/ps_dir');
const dex = require(`${PS}/dist/sim`).Dex.forGen(9);

const UPDATE = process.argv.includes('--update');
const { foePosterior } = require('../src/item_posterior');
const LEGACY_STATS = process.env.PSRL_LEGACY_STATS === '1';
// Writes a true, unrevealed foe Pokemon's stat into the observation, to prove the audit catches it.
const PLANT = process.argv.includes('--plant-leak');
const STATS = ['atk', 'def', 'spa', 'spd', 'spe'];
const FOE_TURN = N_BASE_SCALARS + 9;                  // foe's last turn: move slot x4, streak, lost, healed, protect, switched
const FOE_STATS = N_BASE_SCALARS + 19 + N_MON * 5;    // after both last-turn blocks, matchup turns and own stats
const N = parseInt(process.argv[2] || '60');
const GOLDEN = path.join(__dirname, 'golden_encode.json');
const UNKNOWN = TABLES._meta.UNKNOWN, NONE = TABLES._meta.NONE;

// reverse maps, for turning an emitted id back into its name
const rev = {};
for (const t of ['species', 'moves', 'items', 'abilities', 'types', 'statuses']) {
	rev[t] = {};
	for (const [k, v] of Object.entries(TABLES[t])) rev[t][v] = k;
}
const FOE_BASE = N_MON * IDS_PER_MON;   // opponent block starts after own 6 mons
const hashes = [];
let leaks = 0, checks = 0;
const leakEx = [];

for (let ep = 0; ep < N; ep++) {
	const seenText = [];                       // every p1 line this player has received
	const h = crypto.createHash('sha256');
	const r = playBattle({
		tag: 'encode', ep, rngSeed: ep * 40503, pick: randomPick, turnCap: 1e9,
		onStep: views => seenText.push(...views.p1),
		onDecision: (i, ctx) => {
			if (i !== 0) return;
			const obs = encode(ctx.request, ctx.tracker);
			h.update(Buffer.from(obs.ids.buffer, obs.ids.byteOffset, obs.ids.byteLength));
			h.update(Buffer.from(obs.scalars.buffer, obs.scalars.byteOffset, obs.scalars.byteLength));

			// --- leakage audit on the opponent block ---
			// Rebuild the vocabulary this player could possibly have seen by tokenising
			// each protocol field the way the tracker does. Independent of the tracker, so
			// a tracker bug cannot hide a leak.
			const seenVocab = new Set();
			for (const line of seenText) {
				for (const field of String(line).split('|')) {
					for (const part of field.split(',')) {
						const id = toID(part.replace(/^\[?(move|ability|item|from|of|silent)\]?:?\s*/i, ''));
						if (id) seenVocab.add(id);
					}
					const id2 = toID(field);
					if (id2) seenVocab.add(id2);
				}
			}
			// a cosmetic forme on screen (Florges-Blue) is encoded as its base species
			for (const id of [...seenVocab]) {
				const sp = dex.species.get(id);
				if (sp.exists && sp.isCosmeticForme) seenVocab.add(toID(sp.baseSpecies));
			}
			for (let k = 0; k < N_MON; k++) {
				const base = FOE_BASE + k * IDS_PER_MON;
				const fields = [['species', 0], ['items', 1], ['abilities', 2], ['types', 3],
					['moves', 5], ['moves', 6], ['moves', 7], ['moves', 8]];
				for (const [tbl, off] of fields) {
					const id = obs.ids[base + off];
					if (id === UNKNOWN || id === NONE) continue;
					checks++;
					const name = rev[tbl][id];
					if (!name) continue;
					if (!seenVocab.has(name)) {
						// An unrevealed item is now INFERRED from the randbats table rather than left
						// UNKNOWN, so it will not appear in the seen text. Accept it only when it is
						// exactly what the public inference yields from this slot's species and moves -
						// both of which this same loop audits - so the item stays a pure function of
						// audited-public inputs. That is a tighter test than vocabulary membership.
						let excused = false;
						if (tbl === 'items') {
							const spName = rev.species[obs.ids[base]];
							const mvs = [5, 6, 7, 8]
								.map(o => obs.ids[base + o])
								.filter(i => i !== UNKNOWN && i !== NONE)
								.map(i => toID(rev.moves[i] || ''))
								.filter(Boolean);
							const inferred = spName ? foePosterior(toID(spName), mvs).top : '';
							excused = !!inferred && toID(name) === inferred;
						}
						if (!excused) {
							leaks++;
							if (leakEx.length < 5) leakEx.push(`ep${ep} foe slot ${k} ${tbl}="${name}" unseen`);
						}
					}
				}
			}

			// --- Phase 6 scalars about the foe, re-derived from the raw text ---
			const leak = msg => { leaks++; if (leakEx.length < 5) leakEx.push(`ep${ep} ${msg}`); };
			const text = seenText.map(String);
			const turnAt = text.flatMap((x, i) => (x.startsWith('|turn|') ? [i] : []));
			const lastTurn = turnAt.length
				? text.slice(turnAt.length > 1 ? turnAt[turnAt.length - 2] + 1 : 0, turnAt[turnAt.length - 1]) : [];
			const foeMoves = lastTurn.filter(x => x.startsWith('|move|p2a: ') &&
				(!x.includes('|[from]') || x.includes('|[from]lockedmove')));
			const lastFoeMove = foeMoves.length ? toID(foeMoves[foeMoves.length - 1].split('|')[3]) : '';
			const active = [...Array(N_MON).keys()].find(j => obs.scalars[(N_MON + j) * SCAL_PER_MON + 3] === 1);
			const levels = {}, levelOf = {};
			for (const x of text) {
				let m = /^\|(?:switch|drag|replace|detailschange)\|p2a: ([^|]*)\|([^,|]+)(?:[^|]*?\bL(\d+))?/.exec(x);
				if (m) {
					levelOf[m[1]] = m[3] ? +m[3] : 100;
					levels[toID(m[2])] = levelOf[m[1]];
					const sp = dex.species.get(toID(m[2]));
					if (sp.exists && sp.isCosmeticForme) levels[toID(sp.baseSpecies)] = levelOf[m[1]];
					continue;
				}
				// a forme change (Morpeko-Hangry, Minior-Meteor) keeps the Pokemon's level
				m = /^\|-formechange\|p2a: ([^|]*)\|([^,|]+)/.exec(x);
				if (m && m[1] in levelOf) levels[toID(m[2])] = levelOf[m[1]];
			}
			if (PLANT) {
				const j = [...Array(N_MON).keys()].find(j => obs.ids[FOE_BASE + j * IDS_PER_MON] === UNKNOWN);
				const hidden = ctx.battle.battle.sides[1].pokemon.find(p => !seenVocab.has(p.species.id));
				if (j !== undefined && hidden) obs.scalars[FOE_STATS + j * STATS.length + 1] = hidden.storedStats.def / 500;
			}
			for (let k = 0; k < 4; k++) {
				if (!obs.scalars[FOE_TURN + k]) continue;
				checks++;
				const name = active === undefined ? '' : rev.moves[obs.ids[FOE_BASE + active * IDS_PER_MON + 5 + k]];
				if (name !== lastFoeMove) leak(`foe last-move slot ${k}="${name}", text says "${lastFoeMove}"`);
			}
			const protect = lastTurn.some(x => x.startsWith('|-singleturn|p2a: ') && /\|(move: )?Protect$/.test(x)) ? 1 : 0;
			const switched = lastTurn.some(x => x.startsWith('|switch|p2a: ') || x.startsWith('|drag|p2a: ')) ? 1 : 0;
			checks += 2;
			if (obs.scalars[FOE_TURN + 7] !== protect) leak(`foe protect flag ${obs.scalars[FOE_TURN + 7]}, text says ${protect}`);
			if (obs.scalars[FOE_TURN + 8] !== switched) leak(`foe switched flag ${obs.scalars[FOE_TURN + 8]}, text says ${switched}`);
			// Stats must follow from a species and level named in the text: the emitted species when
			// its id is known, otherwise any foe species on screen (cosmetic formes such as
			// Florges-Blue are missing from the id table but still named in the switch line).
			// Atk and Spe now follow the randbats 0-IV rule, which is a function of the slot's
			// revealed moves - themselves audited by the loop above - so re-derive them the same
			// way rather than assuming 31 IVs. LEGACY_STATS turns the correction off in the
			// encoder, so honour it here too.
			const slotMoves = j => [5, 6, 7, 8]
				.map(o => obs.ids[FOE_BASE + j * IDS_PER_MON + o])
				.filter(i => i !== UNKNOWN && i !== NONE)
				.map(i => toID(rev.moves[i] || ''))
				.filter(Boolean);
			const statsOf = (id, lv, mvs) => {
				const q = LEGACY_STATS ? null : foePosterior(id, mvs);
				return STATS.map(k => {
					const iv = !q ? 31 : k === 'atk' ? q.atkIv : k === 'spe' ? q.speIv : 31;
					const ev = !q ? 85 : k === 'atk' ? q.atkEv : k === 'spe' ? q.speEv : 85;
					return statAt(dex.species.get(id).baseStats[k], lv, iv, ev) / 500;
				});
			};
			const visibleFor = mvs => Object.entries(levels).filter(([id]) => dex.species.get(id).exists)
				.map(([id, lv]) => statsOf(id, lv, mvs));
			const zeros = STATS.map(() => 0);
			for (let j = 0; j < N_MON; j++) {
				const sid = obs.ids[FOE_BASE + j * IDS_PER_MON];
				const got = STATS.map((_, s) => obs.scalars[FOE_STATS + j * STATS.length + s]);
				const near = want => want.every((w, s) => Math.abs(got[s] - w) <= 1e-5);
				checks += STATS.length;
				const mvs = slotMoves(j);
				const ok = sid === UNKNOWN ? near(zeros) || visibleFor(mvs).some(near)
					: near(sid === NONE ? zeros
						: statsOf(rev.species[sid], levels[rev.species[sid]] || 100, mvs));
				if (!ok) leak(`foe slot ${j} stats [${got.map(x => x.toFixed(3))}] follow from no species on screen`);
			}
		},
	});
	r.battle.destroy();
	hashes.push(h.digest('hex').slice(0, 16));
}

console.log(`layout: ids=${N_IDS} scalars=${N_SCALARS}`);
console.log(`leakage audit: ${checks} opponent facts checked, ${leaks} leaks`);
if (leakEx.length) console.log('  ' + leakEx.join('\n  '));

if (PLANT) {
	console.log(`planted leak ${leaks ? 'CAUGHT' : 'MISSED'}`);
	process.exitCode = leaks ? 0 : 1;
} else if (UPDATE) {
	fs.writeFileSync(GOLDEN, JSON.stringify({ N_IDS, N_SCALARS, n: N, hashes }, null, 1));
	console.log(`golden file written (${N} episodes)`);
} else if (fs.existsSync(GOLDEN)) { (() => {
	const g = JSON.parse(fs.readFileSync(GOLDEN, 'utf8'));
	const layoutOk = g.N_IDS === N_IDS && g.N_SCALARS === N_SCALARS;
	if (g.hashes.length !== hashes.length) {
		// Comparing a different episode count is a usage error, not a stability failure.
		console.log(`stability: SKIPPED - golden has ${g.hashes.length} episodes, ran ${hashes.length}. ` +
			`Re-run with ${g.hashes.length}, or --update to re-baseline.`);
		process.exitCode = leaks ? 1 : 0;
		return;
	}
	const same = g.hashes.every((x, i) => x === hashes[i]);
	console.log(`stability: layout ${layoutOk ? 'ok' : 'CHANGED'}, hashes ${same ? 'identical' : 'DIFFER'}`);
	if (!same) {
		const i = g.hashes.findIndex((x, k) => x !== hashes[k]);
        console.log(`  first difference at episode ${i}: golden ${g.hashes[i]} vs now ${hashes[i]}`);
	}
	process.exitCode = (layoutOk && same && !leaks) ? 0 : 1;
})(); } else {
	console.log('no golden file; run with --update to create one');
	process.exitCode = leaks ? 1 : 0;
}
