'use strict';
/**
 * Ability and item effects in the damage estimate (Phase 6 step 2.9).
 *
 * Each case compares two estimates that differ in exactly one field, so the assertions are ratios
 * and never hard-code an absolute damage number.
 */
const { damageFraction } = require('../src/damage');
const PS = require('../src/ps_dir');
const dex = require(`${PS}/dist/sim`).Dex.forGen(9);

const ZERO = { atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0 };
const me = (over = {}) => ({ species: dex.species.get('garchomp'), level: 80,
	stats: { atk: 250, def: 200, spa: 200, spd: 200, spe: 220 },
	boosts: { ...ZERO }, tera: '', status: '', ability: '', item: '', ...over });
const foe = (over = {}) => ({ species: dex.species.get('blissey'), level: 80,
	boosts: { ...ZERO }, tera: '', ability: '', hp: 1, ...over });
const est = (moveId, m, f) => damageFraction(dex.moves.get(moveId), m, f);

const cases = [
	// foe abilities
	['Levitate absorbs Ground', () => est('earthquake', me(), foe({ ability: 'levitate' })) === 0
		&& est('earthquake', me(), foe()) > 0],
	['Storm Drain absorbs Water', () => est('surf', me(), foe({ ability: 'stormdrain' })) === 0],
	['Thousand Arrows ignores Levitate', () => est('thousandarrows', me(), foe({ ability: 'levitate' })) > 0],
	['Multiscale halves at full HP only', () => {
		const full = est('earthquake', me(), foe({ ability: 'multiscale', hp: 1 }));
		const hurt = est('earthquake', me(), foe({ ability: 'multiscale', hp: 0.9 }));
		return Math.abs(full / hurt - 0.5) < 1e-9;
	}],
	['Thick Fat halves Fire', () => {
		const r = est('flamethrower', me(), foe({ ability: 'thickfat' })) / est('flamethrower', me(), foe());
		return Math.abs(r - 0.5) < 1e-9;
	}],
	['Tablets of Ruin cuts physical by a quarter', () => {
		const r = est('earthquake', me(), foe({ ability: 'tabletsofruin' })) / est('earthquake', me(), foe());
		return Math.abs(r - 0.75) < 1e-9;
	}],
	['Filter softens super-effective hits', () => {
		const f = foe({ species: dex.species.get('skarmory') });   // Steel/Flying: Fire is 2x
		const r = est('flamethrower', me(), { ...f, ability: 'filter' }) / est('flamethrower', me(), f);
		return Math.abs(r - 0.75) < 1e-9;
	}],
	// own abilities
	['Tinted Lens doubles a resisted hit', () => {
		const f = foe({ species: dex.species.get('skarmory') });   // Steel/Flying: Normal is 0.5x
		const r = est('bodyslam', me({ ability: 'tintedlens' }), f) / est('bodyslam', me(), f);
		return Math.abs(r - 2) < 1e-9;
	}],
	['Technician boosts weak moves only', () => {
		const weak = est('bulletpunch', me({ ability: 'technician' }), foe()) / est('bulletpunch', me(), foe());
		const strong = est('earthquake', me({ ability: 'technician' }), foe()) / est('earthquake', me(), foe());
		return Math.abs(weak - 1.5) < 1e-9 && Math.abs(strong - 1) < 1e-9;
	}],
	['Huge Power doubles physical attack', () => {
		const r = est('earthquake', me({ ability: 'hugepower' }), foe()) / est('earthquake', me(), foe());
		return r > 1.9 && r < 2.1;      // the stat is doubled before the formula's flooring
	}],
	// own items
	['Life Orb adds 30%', () => {
		const r = est('earthquake', me({ item: 'lifeorb' }), foe()) / est('earthquake', me(), foe());
		return Math.abs(r - 1.3) < 1e-9;
	}],
	['Choice Specs boosts special, not physical', () => {
		const sp = est('flamethrower', me({ item: 'choicespecs' }), foe()) / est('flamethrower', me(), foe());
		const ph = est('earthquake', me({ item: 'choicespecs' }), foe()) / est('earthquake', me(), foe());
		return Math.abs(sp - 1.5) < 1e-9 && Math.abs(ph - 1) < 1e-9;
	}],
	['type-boosting items match the move type', () => {
		const hit = est('earthquake', me({ item: 'softsand' }), foe()) / est('earthquake', me(), foe());
		const miss = est('flamethrower', me({ item: 'softsand' }), foe()) / est('flamethrower', me(), foe());
		return Math.abs(hit - 1.2) < 1e-9 && Math.abs(miss - 1) < 1e-9;
	}],
	['Light Ball only helps its species', () => {
		const wrong = est('earthquake', me({ item: 'lightball' }), foe()) / est('earthquake', me(), foe());
		const pika = me({ species: dex.species.get('pikachu'), item: 'lightball' });
		const right = est('earthquake', pika, foe()) / est('earthquake', me({ species: dex.species.get('pikachu') }), foe());
		return Math.abs(wrong - 1) < 1e-9 && Math.abs(right - 2) < 1e-9;
	}],
	['burn halves physical damage, but not with Guts', () => {
		const burned = est('earthquake', me({ status: 'brn' }), foe()) / est('earthquake', me(), foe());
		const guts = est('earthquake', me({ status: 'brn', ability: 'guts' }), foe()) / est('earthquake', me(), foe());
		return Math.abs(burned - 0.5) < 1e-9 && Math.abs(guts - 1.5) < 1e-9;
	}],
	['an unknown foe ability changes nothing', () => est('earthquake', me(), foe({ ability: '' }))
		=== est('earthquake', me(), foe())],
];

let failed = 0;
for (const [name, fn] of cases) {
	let ok = false;
	try { ok = !!fn(); } catch (e) { console.log(`  [FAIL] ${name}: ${e.message}`); failed++; continue; }
	console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}`);
	if (!ok) failed++;
}
console.log(`${cases.length - failed}/${cases.length} passed`);
process.exitCode = failed ? 1 : 0;
