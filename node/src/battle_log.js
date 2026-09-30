'use strict';
/** Protocol-log normalisation shared by every replay-validation path. */
const PS = require('./ps_dir');
const { State } = require(`${PS}/dist/sim/state`);
const { extractChannelMessages } = require(`${PS}/dist/sim/battle`);

/** Server-room chrome the simulator never emits; not part of a reconstruction check. */
const DROP = /^\|(j|J|l|L|c|c:|raw|inactive|inactiveoff|uhtml|html|n|N|chat|-message|player|win|tie|expire|message|badge|error|askreg|debug|seed)\b/;
const DROP_BARE = /^\|\|/;

/**
 * Mechanically meaningful events. Cross-version protocol cosmetics ([of], [silent],
 * init ordering) are not evidence that a reconstruction is wrong, so the semantic
 * comparison keeps only what determines the course of the battle.
 */
const KEEP_SEM = /^\|(move|switch|drag|faint|-damage|-heal|-status|-curestatus|-boost|-unboost|-weather|-fieldstart|-fieldend|-sidestart|-sideend|-terastallize|replace|detailschange|-formechange|turn|upkeep)\b/;

/**
 * @param {string[]} lines
 * @param {boolean} flatten  battle.log carries raw |split|pN markers; a stored replay log
 *                           is the already-flattened omniscient channel (server getLog(-1)).
 */
function normalize(lines, flatten = false) {
	const src = flatten ? extractChannelMessages(lines.join('\n'), [-1])[-1] : lines;
	return State.normalizeLog(src).filter(l => l && !DROP.test(l) && !DROP_BARE.test(l)).join('\n');
}

/** Strip bracket annotations whose formatting drifted between simulator versions. */
function skeleton(text) {
	return text.split('\n').filter(l => KEEP_SEM.test(l))
		.map(l => l.split('|').filter(x => !x.startsWith('[')).join('|')).join('\n');
}

/** Consistent username substitution; applied to inputlog AND log so comparison is unaffected. */
function redact(text, names, hash) {
	for (const n of names) if (n) text = text.split(n).join(hash(n));
	return text;
}

module.exports = { DROP, DROP_BARE, KEEP_SEM, normalize, skeleton, redact };
