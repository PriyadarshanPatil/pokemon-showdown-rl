'use strict';
/**
 * Action space for gen9 singles (superset; older gens simply mask more).
 *
 *   0-3    move 1..4
 *   4-7    move 1..4 + terastallize
 *   8-13   switch to team slot 1..6
 *
 * Absolute switch slots are safe here because randbats has no Team Preview, so a
 * Pokemon's team slot is fixed for the whole battle. Observation encoding MUST use
 * the same slot order or the indices mean nothing.
 */
const N_MOVES = 4;
const TERA_OFFSET = 4;
const SWITCH_OFFSET = 8;
const N_SWITCH = 6;
const N_ACTIONS = 14;

const fainted = p => !p || (p.condition || '').endsWith(' fnt');

/** Boolean legality mask of length N_ACTIONS for one side's ChoiceRequest. */
function legalMask(request) {
	const mask = new Array(N_ACTIONS).fill(false);
	if (!request || request.wait) return mask;
	if (request.teamPreview) {
		throw new Error('teamPreview is not reachable in gen9randombattle; unsupported');
	}

	const pokemon = request.side.pokemon;

	if (request.forceSwitch) {
		// Only slot 0 (singles). Revival Blessing inverts the fainted test.
		const reviving = !!pokemon[0]?.reviving;
		for (let j = 0; j < N_SWITCH; j++) {
			const p = pokemon[j];
			if (!p || p.active) continue;
			if (fainted(p) === reviving) mask[SWITCH_OFFSET + j] = true;
		}
		return mask;
	}

	const active = request.active[0];
	// `moves` can be shorter than 4, and is length 1 when Struggle is forced.
	const moves = active.moves || [];
	for (let i = 0; i < moves.length && i < N_MOVES; i++) {
		if (moves[i].disabled) continue;
		mask[i] = true;
		if (active.canTerastallize) mask[TERA_OFFSET + i] = true;
	}
	if (!active.trapped) {
		for (let j = 0; j < N_SWITCH; j++) {
			const p = pokemon[j];
			if (!p || p.active || fainted(p)) continue;
			mask[SWITCH_OFFSET + j] = true;
		}
	}
	return mask;
}

/** Action index -> Showdown choice string. */
function toChoice(idx) {
	if (idx < TERA_OFFSET) return `move ${idx + 1}`;
	if (idx < SWITCH_OFFSET) return `move ${idx - TERA_OFFSET + 1} terastallize`;
	return `switch ${idx - SWITCH_OFFSET + 1}`;
}

/**
 * Showdown choice string -> action index. Used to label human replay actions.
 * Returns -1 for choices outside this space (pass/default/shift/gimmicks).
 */
function fromChoice(choice, request) {
	const s = choice.trim().toLowerCase();
	if (s.startsWith('switch ')) {
		const rest = s.slice(7).trim();
		let slot = parseInt(rest);
		if (isNaN(slot)) {
			// name form: first non-active pokemon whose species/nickname matches
			const pokemon = request.side.pokemon;
			slot = pokemon.findIndex(p => p && !p.active &&
				(p.ident.split(': ')[1] || '').toLowerCase().replace(/[^a-z0-9]/g, '') ===
				rest.replace(/[^a-z0-9]/g, '')) + 1;
			if (!slot) return -1;
		}
		return SWITCH_OFFSET + slot - 1;
	}
	if (!s.startsWith('move ')) return -1;
	let rest = s.slice(5).trim();
	let tera = false;
	for (const suffix of [' terastallize', ' tera']) {
		if (rest.endsWith(suffix)) { tera = true; rest = rest.slice(0, -suffix.length).trim(); }
	}
	if (/ (mega|megax|megay|zmove|ultra|dynamax|max)$/.test(rest)) return -1;
	let slot = parseInt(rest);
	if (isNaN(slot)) {
		const moves = request.active?.[0]?.moves || [];
		const norm = x => String(x).toLowerCase().replace(/[^a-z0-9]/g, '');
		slot = moves.findIndex(m => norm(m.id) === norm(rest) || norm(m.move) === norm(rest)) + 1;
		if (!slot) return -1;
	}
	if (slot < 1 || slot > N_MOVES) return -1;
	return (tera ? TERA_OFFSET : 0) + slot - 1;
}

module.exports = { N_ACTIONS, N_MOVES, N_SWITCH, TERA_OFFSET, SWITCH_OFFSET, legalMask, toChoice, fromChoice };
