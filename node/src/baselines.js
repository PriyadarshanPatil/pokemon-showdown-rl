'use strict';
/**
 * Scripted opponents. These are the evaluation ladder and the bootstrap opponents for
 * league play. Each policy is (request, tracker, mask, rng) -> action index.
 *
 * Design lifted from Metamon's heuristic baselines (MIT, UT Robot Perception and
 * Learning Lab); reimplemented here against our request/tracker interface rather than
 * poke-env's Player API.
 */
const PS = require('./ps_dir');
const { Dex } = require(`${PS}/dist/sim`);
const { toID } = require('./tracker');
const { N_MOVES, TERA_OFFSET, SWITCH_OFFSET } = require('./actions');

const dex = Dex.forGen(9);
const legalOf = mask => { const o = []; for (let i = 0; i < mask.length; i++) if (mask[i]) o.push(i); return o; };
const speciesTypes = id => { const s = dex.species.get(id); return s && s.exists ? s.types : []; };

function effectiveness(moveType, defTypes) {
	if (!moveType || !defTypes.length) return 1;
	let mult = 1;
	for (const t of defTypes) {
		if (dex.getImmunity(moveType, t) === false) return 0;
		mult *= Math.pow(2, dex.getEffectiveness(moveType, t));
	}
	return mult;
}

/** Rough expected damage of a move slot, used only for ranking. */
function moveScore(request, tracker, slot) {
	const act = request.active && request.active[0];
	const md = act && act.moves && act.moves[slot];
	if (!md) return -Infinity;
	const mv = dex.moves.get(md.id);
	if (!mv || !mv.exists) return 0;
	const me = request.side.pokemon.find(p => p.active);
	const foe = tracker.activeMon(tracker.foe);
	const defTypes = foe ? speciesTypes(foe.species) : [];
	if (mv.category === 'Status') return 1;                 // ranked below any damaging hit
	const eff = effectiveness(mv.type, defTypes);
	if (eff === 0) return 0;
	const myTypes = me ? speciesTypes(toID(String(me.details).split(', ')[0])) : [];
	const stab = myTypes.includes(mv.type) ? 1.5 : 1;
	const phys = mv.category === 'Physical';
	const atk = me && me.stats ? (phys ? me.stats.atk : me.stats.spa) : 100;
	const fs = foe ? dex.species.get(foe.species) : null;
	const def = fs && fs.exists ? (phys ? fs.baseStats.def : fs.baseStats.spd) : 100;
	const bp = mv.basePower || (mv.multihit ? 25 : 60);
	const acc = mv.accuracy === true ? 1 : (mv.accuracy || 100) / 100;
	return bp * eff * stab * acc * (atk / Math.max(1, def)) * (mv.multihit ? 3 : 1) + 10;
}

/** How badly the active Pokemon is threatened by the opposing active Pokemon. */
function matchupScore(tracker, sideId) {
	const mine = tracker.activeMon(sideId);
	const foe = tracker.activeMon(sideId === 'p1' ? 'p2' : 'p1');
	if (!mine || !foe) return 0;
	const myTypes = speciesTypes(mine.species), foeTypes = speciesTypes(foe.species);
	let best = 0, worst = 0;
	for (const t of myTypes) best = Math.max(best, effectiveness(t, foeTypes));
	for (const t of foeTypes) worst = Math.max(worst, effectiveness(t, myTypes));
	return best - worst;
}

const policies = {
	random(request, tracker, mask, rng) {
		const legal = legalOf(mask);
		return legal[Math.floor(rng() * legal.length)];
	},

	maxdamage(request, tracker, mask, rng) {
		const legal = legalOf(mask);
		const moves = legal.filter(a => a < TERA_OFFSET);
		if (!moves.length) return legal[Math.floor(rng() * legal.length)];   // forced switch
		let best = moves[0], bestScore = -Infinity;
		for (const a of moves) {
			const s = moveScore(request, tracker, a);
			if (s > bestScore) { bestScore = s; best = a; }
		}
		return best;
	},

	/** Max damage, but pivot out of a clearly losing type matchup. */
	heuristic(request, tracker, mask, rng) {
		const legal = legalOf(mask);
		const moves = legal.filter(a => a < TERA_OFFSET);
		const switches = legal.filter(a => a >= SWITCH_OFFSET);
		if (!moves.length) {
			if (!switches.length) return legal[0];
			// pick the switch-in with the best type matchup against the foe's active
			const foe = tracker.activeMon(tracker.foe);
			const foeTypes = foe ? speciesTypes(foe.species) : [];
			let best = switches[0], bestScore = -Infinity;
			for (const a of switches) {
				const p = request.side.pokemon[a - SWITCH_OFFSET];
				if (!p) continue;
				const t = speciesTypes(toID(String(p.details).split(', ')[0]));
				let off = 0, def = 0;
				for (const x of t) off = Math.max(off, effectiveness(x, foeTypes));
				for (const x of foeTypes) def = Math.max(def, effectiveness(x, t));
				const s = off - def;
				if (s > bestScore) { bestScore = s; best = a; }
			}
			return best;
		}
		const mu = matchupScore(tracker, tracker.me);
		const me = request.side.pokemon.find(p => p.active);
		const hp = me ? (() => { const m = /^(\d+)\/(\d+)/.exec(me.condition || ''); return m ? +m[1] / +m[2] : 1; })() : 1;
		if (switches.length && mu < -1.5 && hp > 0.2 && rng() < 0.8) {
			return policies.heuristic({ ...request, active: null }, tracker, mask.map((v, i) => v && i >= SWITCH_OFFSET), rng);
		}
		return policies.maxdamage(request, tracker, mask, rng);
	},

	/**
	 * Punishes a foe that never pivots (Phase 7 step 2). Two exploits, both worthless against an
	 * opponent who leaves: take a free favourable matchup whenever the bench offers a clearly
	 * better one, and use a self-boosting status move while nothing threatens us. A pivoting
	 * opponent denies both by switching out, so this baseline can separate a switching policy
	 * from a static one - which random/maxdamage/heuristic demonstrably cannot.
	 *
	 * Deliberately self-contained rather than extending `heuristic`, whose win rates are
	 * recorded gate values in docs/PHASE2.md and must not move.
	 */
	punisher(request, tracker, mask, rng) {
		const legal = legalOf(mask);
		const moves = legal.filter(a => a < TERA_OFFSET);
		const switches = legal.filter(a => a >= SWITCH_OFFSET);
		if (!moves.length) return policies.heuristic(request, tracker, mask, rng);
		const me = request.side.pokemon.find(p => p.active);
		const m = me ? /^(\d+)\/(\d+)/.exec(me.condition || '') : null;
		const hp = m ? +m[1] / +m[2] : 1;
		const mu = matchupScore(tracker, tracker.me);

		// 1) free switch: a bench mon is clearly better here and the foe will not run from it
		const foe = tracker.activeMon(tracker.foe);
		const foeTypes = foe ? speciesTypes(foe.species) : [];
		let bestSwitch = -1, bestScore = mu + 1;          // demand a clear margin over staying
		for (const a of switches) {
			const p = request.side.pokemon[a - SWITCH_OFFSET];
			if (!p) continue;
			const t = speciesTypes(toID(String(p.details).split(', ')[0]));
			let off = 0, def = 0;
			for (const x of t) off = Math.max(off, effectiveness(x, foeTypes));
			for (const x of foeTypes) def = Math.max(def, effectiveness(x, t));
			if (off - def > bestScore) { bestScore = off - def; bestSwitch = a; }
		}
		if (bestSwitch >= 0 && hp > 0.25) return bestSwitch;

		// 2) set up while safe: boosting is a free turn against a Pokemon that stays in
		const act = request.active && request.active[0];
		if (act && mu >= 0 && hp >= 0.6) {
			const mine = tracker.activeMon(tracker.me);
			const boosts = (mine && mine.boosts) || {};
			if (!Object.keys(boosts).some(k => boosts[k] >= 2)) {
				for (const a of moves) {
					const md = act.moves[a];
					const mv = md && dex.moves.get(md.id);
					if (mv && mv.exists && mv.category === 'Status' && mv.target === 'self'
						&& mv.boosts && Object.keys(mv.boosts).some(k => mv.boosts[k] > 0)) return a;
				}
			}
		}
		return policies.maxdamage(request, tracker, mask, rng);
	},
};

module.exports = { policies, moveScore, matchupScore, effectiveness };
