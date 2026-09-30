'use strict';
/**
 * Observation encoder. Actor features come from the player's own ChoiceRequest plus a
 * Tracker fed only that player's protocol channel — never from the Battle object.
 *
 * Own Pokemon are emitted in REQUEST ORDER, i.e. team slots 1-6, which is exactly what
 * switch action indices 8-13 address. Changing one without the other silently breaks
 * the mapping.
 */
const T = require('./id_tables.json');
const { toID, EMPTY_BOOSTS } = require('./tracker');
const { damageFraction, statAt } = require('./damage');
const PS = require('./ps_dir');
const { Dex } = require(`${PS}/dist/sim`);
const { posterior, moveMarginals, speciesSets } = require('./set_posterior');
const { foePosterior, itemAtkMult, itemSpDMult, itemSpeMult } = require('./item_posterior');
const dex = Dex.forGen(9);

const N_MON = 6, N_SIDE = 2, N_MOVE = 4;
const IDS_PER_MON = 9;                       // species,item,ability,tera,status,move1..4
const SCAL_PER_MON = 17;                     // 6 flags + 7 boosts + 4 pp
const BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const SIDE_CONDS = Object.keys(T.sideConditions).filter(k => k && k !== '<unknown>').sort();
const VOLATILES = Object.keys(T.volatiles).filter(k => k && k !== '<unknown>').sort();
const PSEUDO = Object.keys(T.pseudoWeathers).filter(k => k && k !== '<unknown>').sort();

// Derived combat features. Without these the network has to rediscover the type chart
// and every move's base power from embeddings alone, which is why BC lost to a scripted
// max-damage bot (27% vs 39% agreement with human actions).
const MOVE_FEATS = 8;                    // per move slot of OUR active
const MATCHUP_FEATS = 6;
const ROLES = ['AV Pivot', 'Bulky Attacker', 'Bulky Setup', 'Bulky Support', 'Fast Attacker',
	'Fast Bulky Setup', 'Fast Support', 'Setup Sweeper', 'Tera Blast user', 'Wallbreaker'];
const POSTERIOR_FEATS = 2 + ROLES.length;   // entropy, P(SE move) + role distribution

// Phase 6 inputs, appended after the original scalars so those stay byte-identical: the last
// completed turn per side, every known Pokemon's stats, and an estimated damage per move.
const LAST_TURN_FEATS = 2 * (N_MOVE + 5) + 1;   // per side: move slot, streak, lost, healed, protect, switched; + matchup turns
const STATS = ['atk', 'def', 'spa', 'spd', 'spe'];

// Phase 7 step 5: what the FOE can do to US. Every switching decision needs it and the
// observation never carried it - the model saw four outgoing damage numbers and nothing
// incoming, so "this mon is about to be KO'd, get out" was not representable at all.
// Everything here is indexed by team slot 0-5, which is exactly what switch actions 8-13
// address; a bench-only ordering would shift meaning with whichever mon is active.
const TIER1_FEATS = N_MON        // worst risk-weighted incoming hit on each of our slots
	+ 1 + N_MON                  // KO flags: ours on the foe now, the foe's on each of our slots
	+ 1                          // do we outspeed the foe's active, after boosts and paralysis
	+ N_MON                      // hazard chip each slot takes on switch-in
	+ N_MOVE;                    // per-move accuracy, kept separate so the raw damage values
	                             // stay byte-identical and padded checkpoints remain valid
// Phase 7 step 6: what the foe is HOLDING, and whether it is locked into one move. sets.json
// carries no item and the foe's is revealed in only ~16% of decisions - always as "gone" - so
// Choice Band, Choice Specs and Life Orb were invisible to every damage number the model saw.
// item_posterior.js reads them off an empirical table instead (75% accurate with nothing
// revealed, 96% once all four moves are out).
const TIER2_FEATS = N_MON        // incoming damage if the foe is locked into the move it just used
	+ 1                          // P(the foe's active holds a Choice item)
	+ 1                          // the same, gated on it having actually repeated a move
	+ 1;                         // confidence in the inferred item, so the id can be discounted
const SPIKE_FRAC = [0, 1 / 8, 1 / 6, 1 / 4];

const N_IDS = N_SIDE * N_MON * IDS_PER_MON + 2;
const N_BASE_SCALARS = N_SIDE * N_MON * SCAL_PER_MON
	+ N_SIDE * SIDE_CONDS.length
	+ N_SIDE * VOLATILES.length
	+ PSEUDO.length + 6
	+ N_MOVE * MOVE_FEATS + MATCHUP_FEATS + POSTERIOR_FEATS;
const N_SCALARS = N_BASE_SCALARS + LAST_TURN_FEATS + N_SIDE * N_MON * STATS.length + N_MOVE
	+ TIER1_FEATS + TIER2_FEATS;

// Ablation switch. Zeroes the 12 posterior features while keeping N_SCALARS identical,
// so the network shape and existing checkpoints are unaffected and the comparison is
// clean. Set PSRL_NO_POSTERIOR=1 in the worker's environment.
const NO_POSTERIOR = process.env.PSRL_NO_POSTERIOR === '1';
// Ablation switch for the Phase 6 inputs: zeroes every scalar after the original ones.
const BASE_INPUTS_ONLY = process.env.PSRL_BASE_INPUTS_ONLY === '1';
// Ablation switch for the step 6 corrections: restores the pre-step-6 observation at identical
// width - 31 IVs / 85 EVs throughout, no weather or terrain in the damage estimate, and no
// inferred item. CODEBASE_NOTES forbids changing an existing scalar's VALUE precisely because it
// invalidates a padded control; this flag hands that control back. Set PSRL_LEGACY_STATS=1.
const LEGACY_STATS = process.env.PSRL_LEGACY_STATS === '1';

// What the randbats table says about a foe: its item, and whether its set was rolled 0 Atk / 0 Spe.
const NEUTRAL_FOE = foePosterior('');
// Inference reads exactly the moves the encoder ENCODES - the first N_MOVE - so the observation
// stays a pure function of what it also shows the model. The tracker can hold more than four ids
// for one mon (Copycat, Mimic, Dancer), and those are not set moves anyway.
const foeInfo = m =>
	(LEGACY_STATS || !m || !m.species
		? NEUTRAL_FOE : foePosterior(m.species, (m.moves || []).slice(0, N_MOVE)));

const speciesTypes = id => { const sp = dex.species.get(id); return sp && sp.exists ? sp.types : []; };
/** Damage multiplier of `moveType` into `defTypes`; 0 for immunity. */
function effMult(moveType, defTypes) {
	if (!moveType || !defTypes.length) return 1;
	let m = 1;
	for (const t of defTypes) {
		if (dex.getImmunity(moveType, t) === false) return 0;
		m *= Math.pow(2, dex.getEffectiveness(moveType, t));
	}
	return m;
}
// log2 multiplier squashed to [-1, 1]: x4 -> +1, x1 -> 0, x1/4 -> -1
const effScalar = m => (m <= 0 ? 0 : Math.max(-1, Math.min(1, Math.log2(m) / 2)));

const look = (tbl, key) => (key ? (tbl[key] ?? T._meta.UNKNOWN) : T._meta.NONE);
/** Cosmetic formes (Florges-Blue) are missing from the id table and share their base species' data. */
const speciesKey = id => {
	if (!id || id in T.species) return id;
	const sp = dex.species.get(id);
	return sp.exists && sp.isCosmeticForme ? toID(sp.baseSpecies) : id;
};
const clamp01 = x => (x < 0 ? 0 : x > 1 ? 1 : x);

/**
 * @param {object} request  this player's ChoiceRequest (authoritative for own side)
 * @param {Tracker} tracker fed with this player's channel only
 */
function encode(request, tracker) {
	const ids = new Int32Array(N_IDS);
	const sc = new Float32Array(N_SCALARS);
	let ip = 0, sp = 0;

	const me = tracker.me, foe = tracker.foe;

	// ---- own side, in team-slot order (matches switch action indices) ----
	const reqMons = (request && request.side && request.side.pokemon) || [];
	for (let i = 0; i < N_MON; i++) {
		const rm = reqMons[i];
		const name = rm ? (rm.ident.split(': ')[1] || '') : '';
		const tm = name ? tracker.sides[me].mons.get(name) : null;
		if (!rm) { ip += IDS_PER_MON; sp += SCAL_PER_MON; continue; }

		const details = String(rm.details || '').split(', ');
		ids[ip++] = look(T.species, speciesKey(toID(details[0])));
		ids[ip++] = look(T.items, toID(rm.item));
		ids[ip++] = look(T.abilities, toID(rm.ability || rm.baseAbility));
		ids[ip++] = look(T.types, toID(rm.teraType));
		const cond = String(rm.condition || '');
		const st = /\b(brn|par|slp|frz|psn|tox)\b/.exec(cond);
		ids[ip++] = look(T.statuses, st ? st[1] : '');
		const mv = rm.moves || [];
		for (let k = 0; k < N_MOVE; k++) ids[ip++] = look(T.moves, mv[k] ? toID(mv[k]) : '');

		const hpm = /^(\d+)\/(\d+)/.exec(cond);
		const hp = hpm ? parseInt(hpm[1]) / Math.max(1, parseInt(hpm[2])) : (cond.endsWith('fnt') ? 0 : 1);
		let lvl = 100;
		for (const x of details.slice(1)) if (/^L\d+$/.test(x)) lvl = parseInt(x.slice(1));
		sc[sp++] = clamp01(hp);
		sc[sp++] = lvl / 100;
		sc[sp++] = cond.endsWith('fnt') ? 1 : 0;
		sc[sp++] = rm.active ? 1 : 0;
		sc[sp++] = rm.terastallized ? 1 : 0;
		sc[sp++] = 1;                                   // own HP is always exactly known
		for (const b of BOOSTS) sc[sp++] = (tm ? tm.boosts[b] : 0) / 6;
		// PP fractions are only present for the active Pokemon's move request
		const act = request.active && request.active[0];
		for (let k = 0; k < N_MOVE; k++) {
			const m = rm.active && act && act.moves && act.moves[k];
			sc[sp++] = m && m.maxpp ? clamp01(m.pp / m.maxpp) : (rm.active ? 0 : 1);
		}
	}

	// ---- opponent side, in order of first reveal, padded with UNKNOWN ----
	const foeMons = tracker.revealed(foe);
	for (let i = 0; i < N_MON; i++) {
		const m = foeMons[i];
		if (!m) {
			// unrevealed: species/item/ability/moves unknown, everything else neutral
			ids[ip++] = T._meta.UNKNOWN;
			ids[ip++] = T._meta.UNKNOWN;
			ids[ip++] = T._meta.UNKNOWN;
			ids[ip++] = T._meta.UNKNOWN;
			ids[ip++] = T._meta.NONE;
			for (let k = 0; k < N_MOVE; k++) ids[ip++] = T._meta.UNKNOWN;
			sc[sp++] = 1; sc[sp++] = 1; sc[sp++] = 0; sc[sp++] = 0; sc[sp++] = 0; sc[sp++] = 0;
			for (let b = 0; b < BOOSTS.length; b++) sc[sp++] = 0;
			for (let k = 0; k < N_MOVE; k++) sc[sp++] = 1;
			continue;
		}
		ids[ip++] = look(T.species, speciesKey(m.species));
		// an unrevealed item falls back to the table's best guess rather than UNKNOWN; the
		// confidence scalar in the Tier-2 block tells the model how much to trust it
		const qm = foeInfo(m);
		ids[ip++] = m.itemKnown ? look(T.items, m.item)
			: (qm.top ? look(T.items, qm.top) : T._meta.UNKNOWN);
		ids[ip++] = m.abilityKnown ? look(T.abilities, m.ability) : T._meta.UNKNOWN;
		ids[ip++] = m.teraType ? look(T.types, m.teraType) : T._meta.UNKNOWN;
		ids[ip++] = look(T.statuses, m.status);
		for (let k = 0; k < N_MOVE; k++) {
			ids[ip++] = m.moves[k] ? look(T.moves, m.moves[k]) : T._meta.UNKNOWN;
		}
		sc[sp++] = clamp01(m.hp / Math.max(1, m.maxhp));
		sc[sp++] = m.level / 100;
		sc[sp++] = m.fainted ? 1 : 0;
		sc[sp++] = tracker.sides[foe].activeName === m.name ? 1 : 0;
		sc[sp++] = m.terastallized ? 1 : 0;
		sc[sp++] = m.hpKnownExact ? 1 : 0;
		for (const b of BOOSTS) sc[sp++] = m.boosts[b] / 6;
		for (let k = 0; k < N_MOVE; k++) sc[sp++] = 1;   // foe PP is never visible
	}

	// ---- side conditions (own, then foe) ----
	for (const sid of [me, foe]) {
		const c = tracker.sides[sid].conditions;
		for (const k of SIDE_CONDS) sc[sp++] = Math.min(3, c[k] || 0) / 3;
	}
	// ---- volatiles on each active Pokemon ----
	for (const sid of [me, foe]) {
		const a = tracker.activeMon(sid);
		for (const v of VOLATILES) sc[sp++] = a && a.volatiles[v] ? 1 : 0;
	}
	// ---- field ----
	for (const k of PSEUDO) sc[sp++] = tracker.pseudo[k] ? 1 : 0;
	ids[ip++] = look(T.weathers, tracker.weather);
	ids[ip++] = look(T.terrains, tracker.terrain);
	sc[sp++] = Math.min(1, tracker.turn / 100);
	sc[sp++] = Math.min(1, tracker.weatherTurns / 8);
	sc[sp++] = Math.min(1, tracker.terrainTurns / 8);
	sc[sp++] = tracker.sides[me].faintedCount / N_MON;
	sc[sp++] = tracker.sides[foe].faintedCount / N_MON;
	sc[sp++] = foeMons.length / N_MON;

	// ---- derived combat features (own request + tracker only; no Battle access) ----
	const act = request.active && request.active[0];
	const meMon = reqMons.find(p => p && p.active);
	const foeMon = tracker.activeMon(foe);
	const foeTypes = foeMon && foeMon.species ? speciesTypes(foeMon.species) : [];
	const myTypes = meMon ? speciesTypes(toID(String(meMon.details).split(', ')[0])) : [];

	let bestOff = 0;
	for (let k = 0; k < N_MOVE; k++) {
		const md = act && act.moves && act.moves[k];
		if (!md) { sp += MOVE_FEATS; continue; }
		const mv = dex.moves.get(md.id);
		const known = mv && mv.exists;
		const mult = known && mv.category !== 'Status' ? effMult(mv.type, foeTypes) : 1;
		if (known && mv.category !== 'Status') bestOff = Math.max(bestOff, mult);
		sc[sp++] = known ? Math.min(1, (mv.basePower || 0) / 150) : 0;
		sc[sp++] = known && mv.category === 'Physical' ? 1 : 0;
		sc[sp++] = known && mv.category === 'Special' ? 1 : 0;
		sc[sp++] = foeTypes.length ? effScalar(mult) : 0;
		sc[sp++] = known && mv.category !== 'Status' && mult === 0 ? 1 : 0;
		sc[sp++] = known && myTypes.includes(mv.type) ? 1 : 0;
		sc[sp++] = known ? (mv.accuracy === true ? 1 : (mv.accuracy || 100) / 100) : 1;
		sc[sp++] = known ? Math.max(-1, Math.min(1, (mv.priority || 0) / 5)) : 0;
	}

	// ---- matchup summary ----
	let worstDef = 0;
	for (const t of foeTypes) worstDef = Math.max(worstDef, effMult(t, myTypes));
	const hpOf = c => { const m = /^(\d+)\/(\d+)/.exec(String(c || '')); return m ? +m[1] / Math.max(1, +m[2]) : 0; };
	const myHp = meMon ? hpOf(meMon.condition) : 0;
	const foeHp = foeMon ? clamp01(foeMon.hp / Math.max(1, foeMon.maxhp)) : 1;
	const mySpe = meMon && meMon.stats ? meMon.stats.spe : 0;
	const foeSp = foeMon && foeMon.species ? dex.species.get(foeMon.species) : null;
	const foeSpe = foeSp && foeSp.exists ? foeSp.baseStats.spe : 0;
	sc[sp++] = foeTypes.length ? effScalar(bestOff) : 0;
	sc[sp++] = foeTypes.length ? effScalar(worstDef) : 0;
	sc[sp++] = mySpe && foeSpe ? Math.max(-1, Math.min(1, Math.log2(mySpe / (foeSpe * 2)) / 2)) : 0;
	sc[sp++] = myHp;
	sc[sp++] = foeHp;
	sc[sp++] = (tracker.sides[foe].faintedCount - tracker.sides[me].faintedCount) / N_MON;

	// ---- opponent set posterior (conditioned only on what has been revealed) ----
	if (!NO_POSTERIOR && foeMon && foeMon.species && speciesSets(foeMon.species).length) {
		const ev = { moves: foeMon.moves, teraType: foeMon.terastallized || undefined };
		const post = posterior(foeMon.species, ev);
		const nSets = post.sets.length;
		sc[sp++] = nSets > 1 ? post.entropy / Math.log(nSets) : 0;
		// P(the foe holds a move that is super-effective on our active)
		const marg = moveMarginals(foeMon.species, ev);
		let pSE = 0;
		for (const [mid, w] of marg) {
			const mv = dex.moves.get(mid);
			if (!mv || !mv.exists || mv.category === 'Status') continue;
			if (effMult(mv.type, myTypes) > 1) pSE = Math.max(pSE, Math.min(1, w));
		}
		sc[sp++] = pSE;
		const roleW = new Array(ROLES.length).fill(0);
		post.sets.forEach((st, i) => {
			const r = ROLES.indexOf(st.role);
			if (r >= 0) roleW[r] += post.probs[i];
		});
		for (const w of roleW) sc[sp++] = w;
	} else {
		sc[sp++] = 1;                        // maximal uncertainty when nothing is known
		sc[sp++] = 0;
		for (let k = 0; k < ROLES.length; k++) sc[sp++] = 0;
	}

	// ---- last completed turn per side (tracker.last) ----
	const ownSlots = act && act.moves ? act.moves.map(m => toID(m.id)) : [];
	for (const [sid, slots] of [[me, ownSlots], [foe, foeMon ? foeMon.moves : []]]) {
		const t = tracker.last[sid];
		const onField = !!t.mon && t.mon === tracker.sides[sid].activeName;
		const slot = onField ? slots.indexOf(t.move) : -1;
		for (let k = 0; k < N_MOVE; k++) sc[sp++] = k === slot ? 1 : 0;
		sc[sp++] = onField ? Math.min(1, t.streak / 5) : 0;
		sc[sp++] = Math.min(1, t.dmg);
		sc[sp++] = Math.min(1, t.heal);
		sc[sp++] = t.protect;
		sc[sp++] = t.switched;
	}
	sc[sp++] = Math.min(1, tracker.matchupTurns / 10);

	// ---- stats: own team exact from the request, revealed foes from species and level ----
	for (let i = 0; i < N_MON; i++) {
		const s = reqMons[i] && reqMons[i].stats;
		for (const k of STATS) sc[sp++] = s ? s[k] / 500 : 0;
	}
	for (let i = 0; i < N_MON; i++) {
		const s = foeMons[i] && foeMons[i].species ? dex.species.get(foeMons[i].species) : null;
		const q = foeInfo(foeMons[i]);
		for (const k of STATS) {
			if (!(s && s.exists)) { sc[sp++] = 0; continue; }
			const iv = k === 'atk' ? q.atkIv : k === 'spe' ? q.speIv : 31;
			const ev = k === 'atk' ? q.atkEv : k === 'spe' ? q.speEv : 85;
			sc[sp++] = statAt(s.baseStats[k], foeMons[i].level, iv, ev) / 500;
		}
	}

	// ---- estimated damage of each of our active's moves to the foe's active, / its max HP ----
	const meSpecies = meMon ? dex.species.get(toID(String(meMon.details).split(', ')[0])) : null;
	const foeSpecies = foeMon && foeMon.species ? dex.species.get(foeMon.species) : null;
	const meTracked = meMon ? tracker.sides[me].mons.get(meMon.ident.split(': ')[1]) : null;
	// the foe's ability when a player could know it: revealed in the battle, or the only ability
	// randbats gives that species (92% of live decisions)
	let foeAbility = '';
	if (foeMon && foeMon.species) {
		if (foeMon.abilityKnown) foeAbility = foeMon.ability;
		else {
			const pool = [...new Set(speciesSets(foeMon.species).flatMap(s => s.abilities))];
			if (pool.length === 1) foeAbility = pool[0];
		}
	}
	const qFoe = foeInfo(foeMon);
	const field = LEGACY_STATS ? { weather: '', terrain: '' }
		: { weather: tracker.weather, terrain: tracker.terrain };
	// a known item is used directly by damage.js, so the inferred multipliers only stand in for
	// an unknown one
	const foeSpD = foeMon && foeMon.itemKnown ? 1 : itemSpDMult(qFoe.probs);
	const foeMultPhys = itemAtkMult(qFoe.probs, true);
	const foeMultSpec = itemAtkMult(qFoe.probs, false);

	let bestMine = 0;
	for (let k = 0; k < N_MOVE; k++) {
		const md = act && act.moves && act.moves[k];
		let d = 0;
		if (md && meSpecies && meSpecies.exists && meMon.stats && foeSpecies && foeSpecies.exists) {
			const lv = /\bL(\d+)\b/.exec(meMon.details);
			const st = /\b(brn|par|slp|frz|psn|tox)\b/.exec(String(meMon.condition || ''));
			d = damageFraction(dex.moves.get(md.id),
				{ species: meSpecies, level: lv ? +lv[1] : 100, stats: meMon.stats,
					boosts: meTracked ? meTracked.boosts : EMPTY_BOOSTS(), tera: meMon.terastallized || '',
					status: st ? st[1] : '', ability: toID(meMon.baseAbility || meMon.ability),
					item: toID(meMon.item) },
				{ species: foeSpecies, level: foeMon.level, boosts: foeMon.boosts,
					tera: foeMon.terastallized ? dex.types.get(foeMon.terastallized).name : '',
					ability: foeAbility, hp: foeMon.maxhp ? foeMon.hp / foeMon.maxhp : 1,
					spdMult: foeSpD }, field);
		}
		if (d > bestMine) bestMine = d;
		sc[sp++] = Math.min(1, d);
	}

	// ---- Tier-1: incoming damage, KO flags, turn order, hazards, accuracy ----
	const conds = tracker.sides[me].conditions || {};
	const rocks = conds.stealthrock ? 1 : 0;
	const spikes = Math.min(3, conds.spikes || 0);
	const boostMul = s => (s >= 0 ? (2 + s) / 2 : 2 / (2 - s));

	// The foe as the attacker. Randbats fixes 85 EVs and 31 IVs EXCEPT on the 31% of sets whose
	// moves are all non-physical, which get 0 Atk, and Gyro Ball / Trick Room sets, which get
	// 0 Spe; foeInfo carries the posterior expectation of each.
	let foeAtk = null;
	if (foeSpecies && foeSpecies.exists && foeMon) {
		const bs = foeSpecies.baseStats;
		foeAtk = { species: foeSpecies, level: foeMon.level,
			stats: { atk: statAt(bs.atk, foeMon.level, qFoe.atkIv, qFoe.atkEv),
				def: statAt(bs.def, foeMon.level),
				spa: statAt(bs.spa, foeMon.level), spd: statAt(bs.spd, foeMon.level),
				spe: statAt(bs.spe, foeMon.level, qFoe.speIv, qFoe.speEv) },
			boosts: foeMon.boosts, status: foeMon.status || '',
			tera: foeMon.terastallized ? dex.types.get(foeMon.terastallized).name : '',
			ability: foeAbility, item: foeMon.itemKnown ? toID(foeMon.item) : '' };
	}
	// Its move distribution, from the same evidence the posterior block above uses. Revealed
	// moves carry weight 1, so for a fully revealed set this is a plain worst case.
	const foeMoves = [];
	if (foeAtk && speciesSets(foeMon.species).length) {
		const marg = moveMarginals(foeMon.species,
			{ moves: foeMon.moves, teraType: foeMon.terastallized || undefined });
		for (const [mid, w] of marg) {
			const mv = dex.moves.get(mid);
			if (mv && mv.exists && mv.category !== 'Status') foeMoves.push([mv, Math.min(1, w)]);
		}
	}

	const incoming = new Array(N_MON).fill(0);
	const hazard = new Array(N_MON).fill(0);
	const defs = new Array(N_MON).fill(null);   // reused by the locked-move pass below
	for (let i = 0; i < N_MON; i++) {
		const rm = reqMons[i];
		if (!rm) continue;
		const sp0 = dex.species.get(speciesKey(toID(String(rm.details).split(', ')[0])));
		if (!sp0 || !sp0.exists) continue;
		const types = sp0.types || [];
		const abil = toID(rm.ability || rm.baseAbility);
		if (toID(rm.item) !== 'heavydutyboots') {          // Boots ignore both hazard kinds
			let h = rocks ? 0.125 * effMult('Rock', types) : 0;
			if (spikes && !types.includes('Flying') && abil !== 'levitate') h += SPIKE_FRAC[spikes];
			hazard[i] = Math.min(1, h);
		}
		if (!foeAtk || !foeMoves.length) continue;
		const lv = /\bL(\d+)\b/.exec(String(rm.details));
		const nm = rm.ident ? rm.ident.split(': ')[1] : '';
		const tm = nm ? tracker.sides[me].mons.get(nm) : null;
		const def = { species: sp0, level: lv ? +lv[1] : 100,
			boosts: tm ? tm.boosts : EMPTY_BOOSTS(), tera: rm.terastallized || '',
			ability: abil, hp: hpOf(rm.condition) };
		defs[i] = def;
		let worst = 0;
		for (const [mv, w] of foeMoves) {
			foeAtk.itemMult = mv.category === 'Physical' ? foeMultPhys : foeMultSpec;
			const d = damageFraction(mv, foeAtk, def, field) * w;   // risk-weighted: unlikely moves count less
			if (d > worst) worst = d;
		}
		incoming[i] = Math.min(1, worst);
	}

	for (let i = 0; i < N_MON; i++) sc[sp++] = incoming[i];
	sc[sp++] = bestMine >= foeHp ? 1 : 0;
	for (let i = 0; i < N_MON; i++) {
		const hp = reqMons[i] ? hpOf(reqMons[i].condition) : 0;
		sc[sp++] = hp > 0 && incoming[i] >= hp ? 1 : 0;
	}
	const myPar = /\bpar\b/.test(String((meMon && meMon.condition) || ''));
	const mySpeEff = mySpe * boostMul(meTracked ? meTracked.boosts.spe : 0) * (myPar ? 0.5 : 1);
	const foeSpeEff = foeSp && foeSp.exists && foeMon
		? statAt(foeSp.baseStats.spe, foeMon.level, qFoe.speIv, qFoe.speEv) * boostMul(foeMon.boosts.spe)
			* (foeMon.status === 'par' ? 0.5 : 1)
			* (foeMon.itemKnown ? 1 : itemSpeMult(qFoe.probs))
		: 0;
	sc[sp++] = mySpeEff && foeSpeEff ? (mySpeEff > foeSpeEff ? 1 : 0) : 0;
	for (let i = 0; i < N_MON; i++) sc[sp++] = hazard[i];
	for (let k = 0; k < N_MOVE; k++) {
		const md = act && act.moves && act.moves[k];
		const mv = md && dex.moves.get(md.id);
		sc[sp++] = mv && mv.exists ? (mv.accuracy === true ? 1 : (mv.accuracy || 100) / 100) : 0;
	}

	// ---- Tier-2: the foe's inferred item, and the move it may be locked into ----
	// A Choice item forces the foe to repeat its last move, so the risk to each of our slots
	// collapses from "worst over the whole movepool" to that one move. That is the difference
	// between a safe switch-in and a bad one, and the Tier-1 worst case cannot express it.
	const lastFoe = tracker.last[foe];
	const foeRepeating = !!(lastFoe && lastFoe.mon && lastFoe.mon === tracker.sides[foe].activeName);
	const lockMv = foeRepeating && lastFoe.move ? dex.moves.get(lastFoe.move) : null;
	for (let i = 0; i < N_MON; i++) {
		let d = 0;
		if (foeAtk && defs[i] && lockMv && lockMv.exists && lockMv.category !== 'Status') {
			foeAtk.itemMult = lockMv.category === 'Physical' ? foeMultPhys : foeMultSpec;
			d = damageFraction(lockMv, foeAtk, defs[i], field);
		}
		sc[sp++] = Math.min(1, d);
	}
	sc[sp++] = qFoe.choiceP;
	sc[sp++] = foeRepeating && lastFoe.streak >= 2 ? qFoe.choiceP : 0;
	sc[sp++] = foeMon && foeMon.itemKnown ? 1 : qFoe.topP;
	// Under the legacy flag the whole Tier-2 block reads zero, so the observation becomes exactly
	// the old 467 scalars padded to 476 - which is precisely what a padded pre-step-6 checkpoint
	// was trained to see.
	if (LEGACY_STATS) sc.fill(0, sp - TIER2_FEATS, sp);

	if (BASE_INPUTS_ONLY) sc.fill(0, N_BASE_SCALARS);

	if (ip !== N_IDS) throw new Error(`id layout drift: wrote ${ip}, expected ${N_IDS}`);
	if (sp !== N_SCALARS) throw new Error(`scalar layout drift: wrote ${sp}, expected ${N_SCALARS}`);
	return { ids, scalars: sc };
}

module.exports = { encode, N_IDS, N_SCALARS, N_BASE_SCALARS, IDS_PER_MON, SCAL_PER_MON, N_MON, TABLES: T };
