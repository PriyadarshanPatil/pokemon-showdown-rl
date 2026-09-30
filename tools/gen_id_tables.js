'use strict';
/**
 * Generates the shared id tables consumed by BOTH node/src/encode.js and the Python
 * side. Single source of truth: if these drift, action and observation indices stop
 * meaning the same thing.
 */
const fs = require('fs'), path = require('path');
const PS = require('../node/src/ps_dir');
const { Dex } = require(`${PS}/dist/sim`);

const GEN = 9;
const dex = Dex.forGen(GEN);

// id 0 is reserved for "none", id 1 for "unknown" (unrevealed opponent info)
const NONE = 0, UNKNOWN = 1;
function table(ids) {
	const sorted = Array.from(new Set(ids)).sort();
	const map = { '': NONE, '<unknown>': UNKNOWN };
	sorted.forEach((k, i) => { map[k] = i + 2; });
	return map;
}

const species = table(dex.species.all().filter(s => s.exists && s.num > 0).map(s => s.id));
const moves = table(dex.moves.all().filter(m => m.exists).map(m => m.id));
const items = table(dex.items.all().filter(i => i.exists).map(i => i.id));
const abilities = table(dex.abilities.all().filter(a => a.exists).map(a => a.id));
const types = table(dex.types.all().map(t => t.id));
const statuses = table(['brn', 'par', 'slp', 'frz', 'psn', 'tox']);

// Volatiles / side conditions / field effects we encode as fixed slots.
const volatiles = table([
	'confusion', 'substitute', 'leechseed', 'taunt', 'encore', 'disable', 'yawn', 'attract',
	'curse', 'nightmare', 'partiallytrapped', 'aquaring', 'ingrain', 'magnetrise', 'protect',
	'endure', 'destinybond', 'focusenergy', 'foresight', 'torment', 'saltcure', 'flashfire',
	'perishsong', 'slowstart', 'dynamax', 'gastroacid', 'imprison', 'lockedmove', 'mustrecharge',
	'powertrick', 'roost', 'smackdown', 'stockpile', 'telekinesis', 'throatchop', 'glaiverush',
]);
const sideConditions = table([
	'stealthrock', 'spikes', 'toxicspikes', 'stickyweb', 'reflect', 'lightscreen', 'auroraveil',
	'safeguard', 'mist', 'tailwind', 'luckychant',
]);
const weathers = table(['sunnyday', 'raindance', 'sandstorm', 'snowscape', 'hail', 'desolateland', 'primordialsea', 'deltastream']);
const terrains = table(['electricterrain', 'grassyterrain', 'mistyterrain', 'psychicterrain']);
const pseudoWeathers = table(['trickroom', 'magicroom', 'wonderroom', 'gravity', 'iondeluge', 'fairylock']);

const out = {
	_meta: {
		gen: GEN,
		psVersion: require('child_process')
			.execSync(`git -C ${PS} rev-parse HEAD`, { encoding: 'utf8' }).trim(),
		generated: 'tools/gen_id_tables.js',
		NONE, UNKNOWN,
	},
	species, moves, items, abilities, types, statuses,
	volatiles, sideConditions, weathers, terrains, pseudoWeathers,
	sizes: {
		species: Object.keys(species).length, moves: Object.keys(moves).length,
		items: Object.keys(items).length, abilities: Object.keys(abilities).length,
		types: Object.keys(types).length, statuses: Object.keys(statuses).length,
		volatiles: Object.keys(volatiles).length, sideConditions: Object.keys(sideConditions).length,
		weathers: Object.keys(weathers).length, terrains: Object.keys(terrains).length,
		pseudoWeathers: Object.keys(pseudoWeathers).length,
	},
};
const dest = path.join(__dirname, '..', 'node', 'src', 'id_tables.json');
fs.writeFileSync(dest, JSON.stringify(out));
console.log('wrote', dest);
console.log(JSON.stringify(out.sizes, null, 1));
