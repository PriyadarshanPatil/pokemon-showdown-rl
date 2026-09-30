'use strict';
/**
 * Exact-support posterior over an opponent Pokemon's randbats set.
 *
 * gen9 randbats draws from a CLOSED, enumerable pool: data/random-battles/gen9/sets.json
 * lists every species' candidate sets and their movepools. So instead of learning a team
 * predictor, we can just condition on what has been revealed. This is the cheap
 * information win Metamon needs a 7.6k-line learned decoder for.
 *
 * The likelihood is an approximation: the generator is role-driven, not a uniform draw of
 * 4 moves from the movepool, so P(moves | set) is modelled as the hypergeometric
 * probability that a uniform 4-subset contains the revealed moves. Support (which sets
 * are possible at all) is exact; the weighting within the support is not.
 */
const PS = require('./ps_dir');
const SETS = require(`${PS}/dist/data/random-battles/gen9/sets.json`);
const { Dex } = require(`${PS}/dist/sim`);
const { toID } = require('./tracker');
const dex = Dex.forGen(9);

/**
 * Cosmetic formes (Gastrodon-East, Florges-Orange, Alcremie-*) are absent from sets.json
 * because the generator keys on the base species. Resolve to the base before lookup.
 */
function setsKey(speciesId) {
	if (SETS[speciesId]) return speciesId;
	const sp = dex.species.get(speciesId);
	if (sp && sp.exists) {
		const base = toID(sp.baseSpecies);
		if (SETS[base]) return base;
	}
	return speciesId;
}

const N_TEAM_MOVES = 4;
const cache = new Map();

function speciesSets(speciesId) {
	if (cache.has(speciesId)) return cache.get(speciesId);
	const entry = SETS[setsKey(speciesId)];
	const out = entry ? entry.sets.map((s, i) => ({
		index: i,
		role: s.role,
		level: entry.level,
		movepool: (s.movepool || []).map(toID),
		abilities: (s.abilities || []).map(toID),
		teraTypes: (s.teraTypes || []).map(toID),
	})) : [];
	cache.set(speciesId, out);
	return out;
}

/** log C(n, k) */
function logChoose(n, k) {
	if (k < 0 || k > n) return -Infinity;
	let r = 0;
	for (let i = 0; i < k; i++) r += Math.log(n - i) - Math.log(i + 1);
	return r;
}

/**
 * @param {string} speciesId
 * @param {object} ev revealed evidence: {moves: id[], ability?, item?, teraType?}
 * @returns {{sets: object[], probs: number[], entropy: number}}
 */
function posterior(speciesId, ev = {}) {
	const sets = speciesSets(speciesId);
	if (!sets.length) return { sets: [], probs: [], entropy: 0 };
	const moves = (ev.moves || []).map(toID).filter(Boolean);
	const logw = sets.map(s => {
		for (const m of moves) if (!s.movepool.includes(m)) return -Infinity;   // exact elimination
		if (ev.ability && s.abilities.length && !s.abilities.includes(toID(ev.ability))) return -Infinity;
		if (ev.teraType && s.teraTypes.length && !s.teraTypes.includes(toID(ev.teraType))) return -Infinity;
		const P = s.movepool.length;
		const k = Math.min(N_TEAM_MOVES, P);
		// P(a uniform k-subset of the movepool contains all revealed moves)
		return logChoose(P - moves.length, k - moves.length) - logChoose(P, k);
	});
	const max = Math.max(...logw);
	if (!isFinite(max)) {
		// evidence contradicts every set (data drift): fall back to uniform
		const u = 1 / sets.length;
		return { sets, probs: sets.map(() => u), entropy: Math.log(sets.length), contradiction: true };
	}
	const w = logw.map(x => Math.exp(x - max));
	const z = w.reduce((a, b) => a + b, 0);
	const probs = w.map(x => x / z);
	const entropy = -probs.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0);
	return { sets, probs, entropy };
}

/** P(this Pokemon knows move X) over the unrevealed slots, marginalised across sets. */
function moveMarginals(speciesId, ev = {}) {
	const { sets, probs } = posterior(speciesId, ev);
	const revealed = new Set((ev.moves || []).map(toID));
	const out = new Map();
	for (const m of revealed) out.set(m, 1);
	sets.forEach((s, i) => {
		if (!probs[i]) return;
		const unknownSlots = Math.max(0, Math.min(N_TEAM_MOVES, s.movepool.length) - revealed.size);
		const candidates = s.movepool.filter(m => !revealed.has(m));
		if (!candidates.length || !unknownSlots) return;
		const per = unknownSlots / candidates.length;   // uniform over remaining pool
		for (const m of candidates) out.set(m, (out.get(m) || 0) + probs[i] * per);
	});
	return out;
}

module.exports = { posterior, moveMarginals, speciesSets, setsKey, N_TEAM_MOVES };
