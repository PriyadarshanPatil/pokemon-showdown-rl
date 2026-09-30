'use strict';
/**
 * Replay reconstruction: inputlog -> a fully re-simulated Battle.
 *
 * gen9 randbats replays carry an inputlog (seed, both team seeds, and every choice
 * verbatim), so battles can be replayed exactly rather than inferred from the protocol
 * log. Team generation reads data/random-battles/gen9/sets.json, which upstream changes
 * roughly monthly, so the era-correct revision is swapped in per replay.
 *
 * The swap works through a Module._load intercept rather than a rebuild because
 * RandomTeams does `this.randomSets = require('./sets.json')` in its CONSTRUCTOR - so the
 * intercept only has to be installed before a Battle is created, not before the sim is
 * required.
 */
const path = require('path');
const fs = require('fs');
const zlib = require('zlib');
const Module = require('module');

const PS = require('./ps_dir');
const REPO = path.join(__dirname, '..', '..');
const VDIR = path.join(REPO, 'data', 'sets_versions');
const DB_PATH = path.join(REPO, 'data', 'replays.db');
/** First upload time whose replays this simulator version can reproduce (see docs/PHASE_A.md). */
const ERA_START = 1751328000;   // 2025-07-01

let currentSets = null;
const origLoad = Module._load;
Module._load = function (request, parent) {
	if (request === './sets.json' && currentSets && parent && parent.filename &&
		parent.filename.includes(path.join('random-battles', 'gen9'))) {
		return currentSets;
	}
	return origLoad.apply(this, arguments);
};

const { Battle } = require(`${PS}/dist/sim`);
const { extractChannelMessages } = require(`${PS}/dist/sim/battle`);

let _versions = null;
function versions() {
	if (!_versions) {
		_versions = JSON.parse(fs.readFileSync(path.join(VDIR, 'manifest.json'), 'utf8'))
			.map(m => ({ ...m, data: JSON.parse(fs.readFileSync(path.join(VDIR, m.file), 'utf8')) }))
			.sort((a, b) => a.epoch - b.epoch);
	}
	return _versions;
}

/** Index of the newest sets.json revision committed at or before `epoch`. */
function versionIndexAt(epoch) {
	const v = versions();
	let lo = 0;
	for (let i = 0; i < v.length; i++) if (v[i].epoch <= epoch) lo = i;
	return lo;
}

/** Select the sets.json revision used for subsequent Battle construction. */
function useSetsVersion(i) {
	const v = versions();
	currentSets = v[Math.max(0, Math.min(v.length - 1, i))].data;
}
const useSetsAt = epoch => useSetsVersion(versionIndexAt(epoch));

/** Player options and start options, for callers that only need metadata. */
function parseInputlog(text) {
	const players = {};
	let start = null;
	for (const line of String(text).split('\n')) {
		if (!line.startsWith('>')) continue;
		const sp = line.indexOf(' ');
		const type = line.slice(1, sp < 0 ? undefined : sp);
		const data = sp < 0 ? '' : line.slice(sp + 1);
		if (type === 'start') start = JSON.parse(data);
		else if (type === 'player') {
			const k = data.indexOf(' ');
			players[data.slice(0, k)] = JSON.parse(data.slice(k + 1));
		}
	}
	return { start, players };
}

/**
 * Replay an inputlog end to end, walking the lines IN ORDER.
 *
 * Order matters for more than tidiness: control commands (`>forcelose` on a forfeit,
 * `>forcewin`, `>forcetie`, `>tiebreak`, `>reseed`) are interleaved with choices, and
 * forfeits are common. Collecting choices up front and replaying only those silently
 * loses the ending.
 *
 * `onDecision(side, request, choice, battle)` fires before each recorded choice is
 * applied, which is how training data is extracted without materialising anything.
 */
function replayInputlog(text, { onDecision = null, anonymousNames = false } = {}) {
	const SIDES = new Set(['p1', 'p2', 'p3', 'p4']);
	let battle = null;
	for (const line of String(text).split('\n')) {
		if (!line.startsWith('>')) continue;
		const sp = line.indexOf(' ');
		const type = line.slice(1, sp < 0 ? undefined : sp);
		const data = sp < 0 ? '' : line.slice(sp + 1);

		if (SIDES.has(type)) {
			const choice = data.startsWith('/choose ') ? data.slice(8) : data;
			if (onDecision) {
				const req = battle.sides[Number(type[1]) - 1].activeRequest;
				if (req && !req.wait) onDecision(type, req, choice, battle);
			}
			battle.choose(type, choice);
			continue;
		}
		switch (type) {
		case 'version': case 'version-origin': case 'chat': break;
		case 'start':
			battle = new Battle({ ...JSON.parse(data), strictChoices: true });
			break;
		case 'player': {
			const k = data.indexOf(' ');
			const sid = data.slice(0, k);
			const opts = JSON.parse(data.slice(k + 1));
			battle.setPlayer(sid, anonymousNames
				? { name: sid, seed: opts.seed, team: opts.team } : opts);
			break;
		}
		case 'forcelose': battle.lose(data); break;
		case 'forcewin': battle.forceWin(data); break;
		case 'forcetie': battle.forceWin(null); break;
		case 'tiebreak': battle.tiebreak(); break;
		case 'reseed': battle.resetRNG(data); break;
		default: throw new Error(`unsupported inputlog line: ${line}`);
		}
	}
	if (!battle) return null;
	return { battle, truncated: !battle.ended };
}

/** +1 if p1 won, -1 if p2 won, 0 for a tie. Must be read before battle.destroy(). */
function resultOf(battle) {
	if (!battle.ended) return null;
	const w = battle.winner;
	return !w ? 0 : (w === battle.sides[0].name ? 1 : -1);
}

/** Open the replay database read-only, tolerating a concurrent writer. */
function openDb(DatabaseSync) {
	for (let attempt = 0; ; attempt++) {
		try {
			const db = new DatabaseSync(DB_PATH, { readOnly: true });
			db.exec('PRAGMA busy_timeout = 15000');
			return db;
		} catch (e) {
			if (attempt >= 11) throw e;
			const wait = 250 * (attempt + 1);
			process.stderr.write(`replay: db busy (${e.errstr || e.message}), retry in ${wait}ms\n`);
			Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, wait);
		}
	}
}

const decodeBody = z => JSON.parse(zlib.inflateSync(z).toString());

module.exports = {
	PS, REPO, VDIR, DB_PATH, ERA_START,
	versions, versionIndexAt, useSetsVersion, useSetsAt,
	parseInputlog, replayInputlog, resultOf, openDb, decodeBody,
	extractChannelMessages,
};
