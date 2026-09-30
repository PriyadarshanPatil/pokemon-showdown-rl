'use strict';
/**
 * Observation encoder for the live ladder.
 *
 * The agent consumes only one player's protocol channel plus the |request| JSON, and the
 * real server emits exactly that stream - so this reuses tracker.js and encode.js
 * VERBATIM. Porting them to Python would create a second implementation that the
 * leakage audit and golden test do not cover, and it would drift.
 *
 * Framing is newline-delimited JSON, deliberately NOT the binary envelope the training
 * worker uses: a ladder battle produces roughly one call every few seconds, so the bytes
 * are irrelevant and being able to `cat` the pipe while debugging is worth far more.
 *
 * stdin  - one command per line: {op:'init'|'lines'|'encode'|'drop', room, ...}
 * stdout - one response per line: {ok:true, ...} or {ok:false, error}
 */
const readline = require('readline');
const { Tracker } = require('./tracker');
const { encode, N_IDS, N_SCALARS } = require('./encode');
const { legalMask, toChoice, N_ACTIONS } = require('./actions');

/** room id -> Tracker. One battle per room; dropped when the battle ends. */
const rooms = new Map();

function handle(cmd) {
	switch (cmd.op) {
	case 'init': {
		// Perspective matters: the bot may be assigned p1 OR p2, and every opponent
		// feature is relative to it.
		if (cmd.perspective !== 'p1' && cmd.perspective !== 'p2') {
			throw new Error(`perspective must be p1 or p2, got ${cmd.perspective}`);
		}
		rooms.set(cmd.room, new Tracker(cmd.perspective));
		return { room: cmd.room, perspective: cmd.perspective };
	}
	case 'lines': {
		const tr = rooms.get(cmd.room);
		if (!tr) throw new Error(`unknown room ${cmd.room}`);
		tr.feed(cmd.lines || []);
		return { room: cmd.room, turn: tr.turn };
	}
	case 'encode': {
		const tr = rooms.get(cmd.room);
		if (!tr) throw new Error(`unknown room ${cmd.room}`);
		const req = cmd.request;
		const obs = encode(req, tr);
		return {
			room: cmd.room,
			ids: Array.from(obs.ids),
			scalars: Array.from(obs.scalars),
			mask: legalMask(req).map(Number),
			turn: tr.turn,
		};
	}
	case 'drop':
		rooms.delete(cmd.room);
		return { room: cmd.room, remaining: rooms.size };
	case 'actions':
		// The Python side needs action index -> choice string. Serving it from here
		// rather than re-deriving it in Python means the two cannot drift.
		return { choices: Array.from({ length: N_ACTIONS }, (_, i) => toChoice(i)) };
	case 'ping':
		return { n_ids: N_IDS, n_scalars: N_SCALARS, n_actions: N_ACTIONS, rooms: rooms.size };
	default:
		throw new Error(`unknown op ${cmd.op}`);
	}
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', line => {
	if (!line.trim()) return;
	let res;
	try {
		res = { ok: true, ...handle(JSON.parse(line)) };
	} catch (e) {
		res = { ok: false, error: String(e.message || e) };
	}
	process.stdout.write(JSON.stringify(res) + '\n');
});
rl.on('close', () => process.exit(0));
process.stderr.write(`encode_service ready: ids=${N_IDS} scalars=${N_SCALARS} actions=${N_ACTIONS}\n`);
