'use strict';
/**
 * What the empirical randbats table says about a foe we have only partly seen: its held item, and
 * whether its set was rolled with 0 Atk or 0 Spe.
 *
 * Both are invisible to sets.json, which carries only role, movepool, abilities and teraTypes. The
 * item is revealed in ~16% of live decisions and essentially always as "gone", so before a reveal
 * there is nothing else to go on - which is why damage.js used to list the foe's item among the
 * things it ignores. The 0-IV rule (teams.ts:1583-1599) zeroes Atk EVs *and* IVs on any set whose
 * moves are all non-physical, and Spe on Gyro Ball / Trick Room sets; the old encoder assumed 31
 * IVs throughout and so overstated a special attacker's Attack by about 25%.
 *
 * tools/build_item_table.js samples the real generator, so these counts are its true distribution
 * rather than a model of it - including its randomChance coin flips. Conditioning is the same move
 * set_posterior.js makes: keep the rows consistent with what has been revealed, where a moveset
 * row is consistent when it contains every revealed move.
 *
 * Measured on 24,000 held-out mons (table seed "2026,9,22,1", test seed "7,7,7,7"): top-1 item
 * accuracy 96.4% with all four moves seen, 88.6% with two, 75.0% with none, and P(choice item)
 * calibrated to within 0.2 points at every level. The 0-Atk rule fires on 31.3% of all randbats
 * Pokemon and is predicted exactly with four moves seen, 93.3% with none.
 */
const { toID } = require('./tracker');
const { setsKey } = require('./set_posterior');

const TABLE = require('./item_table.json').sets;

const CHOICE = new Set(['choiceband', 'choicespecs', 'choicescarf']);
const EMPTY = {
	probs: new Map(), top: '', topP: 0, choiceP: 0,
	atk0P: 0, spe0P: 0, atkIv: 31, atkEv: 85, speIv: 31, speEv: 85,
};

const cache = new Map();

/**
 * @param {string} speciesId
 * @param {string[]} revealed  move ids already seen from this Pokemon
 * @returns {{probs: Map<string, number>, top: string, topP: number, choiceP: number,
 *            atk0P: number, spe0P: number,
 *            atkIv: number, atkEv: number, speIv: number, speEv: number}}
 *   `atkIv`/`atkEv`/`speIv`/`speEv` are posterior expectations, ready for damage.js's statAt.
 *   Unknown species give EMPTY: no item, and the 31/85 the generator uses by default.
 */
function foePosterior(speciesId, revealed = []) {
	const moves = [...new Set((revealed || []).map(toID).filter(Boolean))].sort();
	const key = `${speciesId}\u0000${moves.join('|')}`;
	const hit = cache.get(key);
	if (hit) return hit;

	const byMoves = TABLE[setsKey(toID(speciesId))];
	if (!byMoves) return EMPTY;

	const counts = new Map();
	let total = 0, atk0 = 0, spe0 = 0;
	const add = filter => {
		for (const sig of Object.keys(byMoves)) {
			if (filter) {
				const have = new Set(sig.split('|'));
				let ok = true;
				for (const m of moves) if (!have.has(m)) { ok = false; break; }
				if (!ok) continue;
			}
			const r = byMoves[sig];
			total += r.n;
			atk0 += r.atk0 * r.n;
			spe0 += r.spe0 * r.n;
			for (const [item, c] of Object.entries(r.items)) counts.set(item, (counts.get(item) || 0) + c);
		}
	};
	add(moves.length > 0);
	// Nothing consistent means the revealed moves are off-table (Transform, an illusion, or a
	// sets.json revision newer than the table). Fall back to the species marginal.
	if (!total) add(false);
	if (!total) return EMPTY;

	const probs = new Map();
	let top = '', topP = 0, choiceP = 0;
	for (const [item, c] of counts) {
		const p = c / total;
		probs.set(item, p);
		if (p > topP) { topP = p; top = item; }
		if (CHOICE.has(item)) choiceP += p;
	}
	const atk0P = atk0 / total, spe0P = spe0 / total;
	const out = {
		probs, top, topP, choiceP, atk0P, spe0P,
		// posterior expectations, so a half-certain read lands between the two stat lines
		atkIv: 31 * (1 - atk0P), atkEv: 85 * (1 - atk0P),
		speIv: 31 * (1 - spe0P), speEv: 85 * (1 - spe0P),
	};
	cache.set(key, out);
	return out;
}

/**
 * Expected damage multiplier from the attacker's held item, over the posterior.
 * Choice Band and Choice Specs are +50% to the matching category; Life Orb is +30% to both.
 */
function itemAtkMult(probs, phys) {
	let m = 0, seen = 0;
	for (const [item, p] of probs) {
		seen += p;
		if (item === 'lifeorb') m += p * 1.3;
		else if (item === (phys ? 'choiceband' : 'choicespecs')) m += p * 1.5;
		else m += p;
	}
	return seen ? m / seen : 1;
}

/** Expected special-defence multiplier from the defender's item: Assault Vest is +50% SpD. */
function itemSpDMult(probs) {
	let m = 0, seen = 0;
	for (const [item, p] of probs) {
		seen += p;
		m += p * (item === 'assaultvest' ? 1.5 : 1);
	}
	return seen ? m / seen : 1;
}

/** Expected speed multiplier from the held item: Choice Scarf is +50% Spe. */
function itemSpeMult(probs) {
	let m = 0, seen = 0;
	for (const [item, p] of probs) {
		seen += p;
		m += p * (item === 'choicescarf' ? 1.5 : 1);
	}
	return seen ? m / seen : 1;
}

module.exports = { foePosterior, itemAtkMult, itemSpDMult, itemSpeMult, CHOICE };
