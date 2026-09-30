'use strict';
/**
 * Resolves the pokemon-showdown checkout that every simulator-facing module loads from.
 *
 * $PS_DIR wins when set and is never silently replaced: pointing at the wrong checkout
 * must fail here rather than 60 battles into verify_encode.js (docs/CODEBASE_NOTES.md).
 * Unset, this resolves to a sibling checkout beside the repo, which is the canonical one
 * `node/test/golden_encode.json` reproduces against.
 */
const fs = require('fs');
const path = require('path');

const SIBLING = path.resolve(__dirname, '..', '..', '..', 'pokemon-showdown');
const built = (p) => fs.existsSync(path.join(p, 'dist', 'sim'));

function resolve() {
	const explicit = process.env.PS_DIR;
	if (explicit) {
		if (built(explicit)) return path.resolve(explicit);
		throw new Error(`PS_DIR=${explicit} has no dist/sim. Build it there with \`node build\`, ` +
			`or unset PS_DIR to use ${SIBLING}.`);
	}
	if (built(SIBLING)) return SIBLING;
	throw new Error(
		`pokemon-showdown not found at ${SIBLING}. Clone and build it beside this repo:\n` +
		'  git clone https://github.com/smogon/pokemon-showdown\n' +
		'  cd pokemon-showdown && npm install && node build\n' +
		'or point PS_DIR at an existing built checkout.');
}

module.exports = resolve();
