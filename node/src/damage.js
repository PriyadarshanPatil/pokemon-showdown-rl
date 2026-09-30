'use strict';
/**
 * Rough damage estimate for one move, from what the player can see: its own request (exact stats,
 * ability and item), the foe's species and level, both sides' boosts, types and tera, its own
 * status, and the foe's ability when that is visible - either revealed in the battle, or the only
 * ability randbats gives that species (together, 92% of live decisions).
 *
 * Randbats gives every set 85 EVs and 31 IVs, EXCEPT that a set whose moves are all non-physical
 * gets 0 Atk EVs and IVs, and a Gyro Ball or Trick Room set gets 0 Spe (teams.ts:1588-1599). So
 * statAt takes them explicitly and the caller supplies what the set posterior implies.
 *
 * Weather and terrain arrive in `field`. The attacker's unheld-but-inferred item arrives as
 * `me.itemMult` and the defender's as `foe.spdMult` - posterior-expected multipliers from
 * item_posterior.js, used only when the real item is not known.
 *
 * Ignored: crits and the damage roll's spread. Also ignored because they need state we do not
 * track: Slow Start's first five turns and Stakeout's switch-in timing.
 */
const PS = require('./ps_dir');
const { Dex } = require(`${PS}/dist/sim`);
const dex = Dex.forGen(9);

const statAt = (base, level, iv = 31, ev = 85) =>
	Math.floor((2 * base + iv + Math.floor(ev / 4)) * level / 100) + 5;
const hpAt = (base, level) => (base === 1 ? 1 : Math.floor((2 * base + 31 + 21) * level / 100) + level + 10);
const boost = s => (s >= 0 ? (2 + s) / 2 : 2 / (2 - s));

// held item -> the type it boosts by 1.2 (the multipliers live in handlers, not in the dex data)
const TYPE_ITEM = { silverpowder: 'Bug', charcoal: 'Fire', mysticwater: 'Water', miracleseed: 'Grass',
	magnet: 'Electric', nevermeltice: 'Ice', blackbelt: 'Fighting', poisonbarb: 'Poison',
	softsand: 'Ground', sharpbeak: 'Flying', twistedspoon: 'Psychic', silkscarf: 'Normal',
	hardstone: 'Rock', spelltag: 'Ghost', dragonfang: 'Dragon', blackglasses: 'Dark',
	metalcoat: 'Steel' };
// foe ability -> the move type it absorbs outright
const ABSORB = { levitate: 'Ground', eartheater: 'Ground', voltabsorb: 'Electric',
	lightningrod: 'Electric', motordrive: 'Electric', waterabsorb: 'Water', stormdrain: 'Water',
	dryskin: 'Water', flashfire: 'Fire', wellbakedbody: 'Fire', sapsipper: 'Grass' };
// weather -> [boosted type, weakened type]
const WEATHER = { raindance: ['Water', 'Fire'], primordialsea: ['Water', 'Fire'],
	sunnyday: ['Fire', 'Water'], desolateland: ['Fire', 'Water'] };
// terrain -> the move type it boosts by 1.3 for a grounded attacker
const TERRAIN_BOOST = { electricterrain: 'Electric', grassyterrain: 'Grass', psychicterrain: 'Psychic' };
const GRASSY_HALVED = new Set(['earthquake', 'bulldoze', 'magnitude']);
const EMPTY_FIELD = { weather: '', terrain: '' };

/** Grounded for terrain purposes: not Flying, no Levitate, no Air Balloon. */
const grounded = (types, ability, item) =>
	!types.includes('Flying') && ability !== 'levitate' && item !== 'airballoon';

function effectiveness(id, type, defTypes) {
	let m = 1;
	for (const t of defTypes) {
		if (!dex.getImmunity(type, t) && !(id === 'thousandarrows' && t === 'Flying')) return 0;
		let e = dex.getEffectiveness(type, t);
		if (id === 'freezedry' && t === 'Water') e = 1;
		if (id === 'flyingpress') e += dex.getEffectiveness('Flying', t);
		m *= 2 ** e;
	}
	return m;
}

function basePower(move, me, foe) {
	const w = foe.species.weightkg;
	switch (move.id) {
	case 'grassknot': case 'lowkick':
		return w >= 200 ? 120 : w >= 100 ? 100 : w >= 50 ? 80 : w >= 25 ? 60 : w >= 10 ? 40 : 20;
	case 'heavyslam': case 'heatcrash': {
		const r = me.species.weightkg / w;
		return r >= 5 ? 120 : r >= 4 ? 100 : r >= 3 ? 80 : r >= 2 ? 60 : 40;
	}
	default: return move.basePower;
	}
}

/** Items such as Light Ball and Soul Dew only work for the species the dex lists. */
const itemFitsUser = (item, species) => {
	const u = dex.items.get(item).itemUser;
	return !u || u.includes(species.name);
};

/** What our own ability, item and status multiply the damage by. */
function ownMod(move, me, type, phys, eff) {
	let m = 1;
	const f = move.flags || {};
	switch (me.ability) {
	case 'technician': if (move.basePower && move.basePower <= 60) m *= 1.5; break;
	case 'reckless': if (move.recoil || move.hasCrashDamage || move.mindBlownRecoil) m *= 1.2; break;
	case 'tintedlens': if (eff > 0 && eff < 1) m *= 2; break;
	case 'sheerforce': if (move.secondary || (move.secondaries || []).length) m *= 1.3; break;
	case 'toughclaws': if (f.contact) m *= 1.3; break;
	case 'ironfist': if (f.punch) m *= 1.2; break;
	case 'sharpness': if (f.slicing) m *= 1.5; break;
	case 'guts': if (me.status) m *= 1.5; break;
	}
	if (me.item === 'lifeorb') m *= 1.3;
	else if (me.item === (phys ? 'choiceband' : 'choicespecs')) m *= 1.5;
	else if (me.item === (phys ? 'muscleband' : 'wiseglasses')) m *= 1.1;
	else if (me.item === 'expertbelt' && eff > 1) m *= 1.2;
	else if (TYPE_ITEM[me.item] === type) m *= 1.2;
	else if (me.item === 'souldew' && (type === 'Psychic' || type === 'Dragon')
		&& itemFitsUser('souldew', me.species)) m *= 1.2;
	else if (me.item === 'lightball' && itemFitsUser('lightball', me.species)) m *= 2;
	// burn halves physical damage; with Guts the status is already counted as a bonus
	if (phys && me.status === 'brn' && me.ability !== 'guts' && move.id !== 'facade') m *= 0.5;
	return m;
}

/** What the foe's visible ability multiplies our damage by; 0 when it absorbs the type. */
function foeMod(move, foe, type, phys, eff) {
	const a = foe.ability;
	if (!a) return 1;
	const f = move.flags || {};
	if (ABSORB[a] === type && move.id !== 'thousandarrows') return 0;
	if (a === 'bulletproof' && f.bullet) return 0;
	if (a === 'soundproof' && f.sound) return 0;
	if (a === 'wonderguard' && eff <= 1) return 0;
	let m = 1;
	if ((a === 'multiscale' || a === 'shadowshield') && foe.hp >= 1) m *= 0.5;
	if (a === 'thickfat' && (type === 'Fire' || type === 'Ice')) m *= 0.5;
	if (a === 'heatproof' && type === 'Fire') m *= 0.5;
	if (a === 'fluffy') { if (f.contact) m *= 0.5; if (type === 'Fire') m *= 2; }
	if (a === 'icescales' && !phys) m *= 0.5;
	if (a === 'furcoat' && phys) m *= 0.5;
	if (a === 'purifyingsalt' && type === 'Ghost') m *= 0.5;
	if (a === 'punkrock' && f.sound) m *= 0.5;
	if ((a === 'filter' || a === 'solidrock' || a === 'prismarmor') && eff > 1) m *= 0.75;
	if (a === 'tabletsofruin' && phys) m *= 0.75;     // lowers the Attack of every other Pokemon
	if (a === 'vesselofruin' && !phys) m *= 0.75;     // ...and this one the Special Attack
	return m;
}

/** What the active weather and terrain multiply the damage by. */
function fieldMod(move, me, foe, type, field, defTypes) {
	let m = 1;
	const w = WEATHER[field.weather];
	if (w) {
		if (type === w[0]) m *= 1.5;
		else if (type === w[1]) m *= 0.5;
	}
	const t = field.terrain;
	if (t) {
		// the boost follows the attacker's feet, the reductions the defender's
		if (TERRAIN_BOOST[t] === type && grounded(me.species.types, me.ability, me.item)) m *= 1.3;
		const foeDown = grounded(defTypes, foe.ability, '');
		if (t === 'grassyterrain' && GRASSY_HALVED.has(move.id) && foeDown) m *= 0.5;
		if (t === 'mistyterrain' && type === 'Dragon' && foeDown) m *= 0.5;
	}
	return m;
}

/**
 * @param move  dex move
 * @param me    {species, level, stats, boosts, tera, status, ability, item}  tera: type name or ''
 *              optional `itemMult`: expected item multiplier, used only when `item` is unknown
 * @param foe   {species, level, boosts, tera, ability, hp}                   ability: '' if unknown
 *              optional `ivs`/`evs` (partial), and `spdMult` for an inferred Assault Vest
 * @param field {weather, terrain}  ids as tracker.js records them; both default to absent
 * @returns estimated damage as a fraction of the foe's max HP (uncapped)
 */
function damageFraction(move, me, foe, field = EMPTY_FIELD) {
	if (!move.exists || move.category === 'Status' || move.ohko) return 0;
	const foeHp = hpAt(foe.species.baseStats.hp, foe.level);
	if (move.damage === 'level') return me.level / foeHp;
	if (typeof move.damage === 'number') return move.damage / foeHp;
	const bp = basePower(move, me, foe);
	if (!bp) return 0;
	let type = move.type, phys = move.category === 'Physical';
	if (move.id === 'terablast' && me.tera) { type = me.tera; phys = me.stats.atk > me.stats.spa; }
	const defTypes = foe.tera ? [foe.tera] : foe.species.types;
	const eff = effectiveness(move.id, type, defTypes);
	const fIv = k => (foe.ivs && foe.ivs[k] !== undefined ? foe.ivs[k] : 31);
	const fEv = k => (foe.evs && foe.evs[k] !== undefined ? foe.evs[k] : 85);
	const foeStat = k => statAt(foe.species.baseStats[k], foe.level, fIv(k), fEv(k)) * boost(foe.boosts[k]);
	const power = me.ability === 'hugepower' || me.ability === 'purepower' ? 2 : 1;
	const A = move.overrideOffensivePokemon === 'target' ? foeStat('atk')
		: move.overrideOffensiveStat === 'def' ? me.stats.def * boost(me.boosts.def)
		: phys ? me.stats.atk * boost(me.boosts.atk) * power : me.stats.spa * boost(me.boosts.spa);
	const useDef = phys || move.overrideDefensiveStat === 'def';
	let D = foeStat(useDef ? 'def' : 'spd');
	// Sandstorm gives Rock types +50% SpD, Snow gives Ice types +50% Def
	if (!useDef && field.weather === 'sandstorm' && defTypes.includes('Rock')) D *= 1.5;
	if (useDef && (field.weather === 'snowscape' || field.weather === 'hail')
		&& defTypes.includes('Ice')) D *= 1.5;
	if (!useDef && foe.spdMult) D *= foe.spdMult;
	const base = Math.floor(Math.floor(Math.floor(2 * me.level / 5 + 2) * bp * A / D) / 50) + 2;
	const own = me.species.types;
	const stabBase = me.ability === 'adaptability' ? 2 : 1.5;
	const stab = me.tera === type && own.includes(type) ? stabBase + 0.5
		: own.includes(type) || me.tera === type ? stabBase : 1;
	const hits = Array.isArray(move.multihit) ? 3.1 : move.multihit || 1;   // 2-5 hits average 3.1
	// an inferred item only stands in when the real one is unknown, or it would double-count
	const itemMult = me.item ? 1 : (me.itemMult || 1);
	return base * 0.925 * stab * eff * hits * ownMod(move, me, type, phys, eff)
		* foeMod(move, foe, type, phys, eff) * fieldMod(move, me, foe, type, field, defTypes)
		* itemMult / foeHp;
}

module.exports = { damageFraction, statAt, hpAt };
