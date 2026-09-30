'use strict';
/** Deterministic seeding and small PRNGs, shared by workers, tests and tools. */
const crypto = require('crypto');

/** Salted pseudonym for a username. Must match tools/crawl_replays.py. */
const SALT = 'psrl-v1';

/**
 * The three seeds a reproducible battle needs. Battle#getTeam falls back to a random
 * seed when PlayerOptions.seed is absent, so team seeds must always be explicit.
 */
function seedsFor(tag, ep) {
	const h = crypto.createHash('blake2b512').update(`${tag}:${ep}`).digest('hex');
	return {
		seed: `sodium,${h.slice(0, 32)}`,
		p1seed: `sodium,${h.slice(32, 64)}`,
		p2seed: `sodium,${h.slice(64, 96)}`,
	};
}

/** Deterministic PRNG so scripted policies in tests are reproducible. */
function mulberry32(a) {
	return () => {
		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul(a ^ a >>> 15, 1 | a);
		t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
		return ((t ^ t >>> 14) >>> 0) / 4294967296;
	};
}

const userHash = name =>
	'u' + crypto.createHash('blake2b512').update(SALT + name.trim().toLowerCase()).digest('hex').slice(0, 16);

module.exports = { SALT, seedsFor, mulberry32, userHash };
