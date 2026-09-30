'use strict';
/**
 * Export ladder-shaped fixtures from stored replays.
 *
 * For each of the bot's decision points this emits exactly what a live server would have
 * sent it - the protocol lines on its own channel since the previous decision, plus the
 * |request| payload - together with the observation the LOCAL, already-audited encode
 * path produces. tests/test_ladder_equivalence.py then drives encode_service with the
 * same inputs and requires the observations to be byte-identical.
 *
 * That makes "the ladder bot runs the same agent we measured" a checkable claim rather
 * than an assumption, and it needs no network.
 */
const fs = require('fs');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const R = require('../node/src/replay');
const { Tracker } = require('../node/src/tracker');
const { encode } = require('../node/src/encode');
const { legalMask } = require('../node/src/actions');

const N = parseInt(process.argv[2] || '200');
const OUT = process.argv[3] || path.join(R.REPO, 'data', 'ladder_fixture.jsonl');

const db = R.openDb(DatabaseSync);
const rows = db.prepare(
	`SELECT b.id id, b.z z, r.uploadtime t FROM bodies b JOIN replays r ON r.id=b.id
	 WHERE r.has_inputlog=1 AND r.uploadtime >= ? ORDER BY r.id LIMIT ?`).all(R.ERA_START, N);

const out = fs.createWriteStream(OUT);
let battles = 0, decisions = 0;

for (const row of rows) {
	const j = R.decodeBody(row.z);
	if (!j.inputlog) continue;
	R.useSetsAt(row.t);

	// Emit fixtures for BOTH perspectives: the bot may be assigned p1 or p2 on ladder.
	for (const me of ['p1', 'p2']) {
		const tr = new Tracker(me);
		const chan = me === 'p1' ? 1 : 2;
		const room = `${j.id}-${me}`;
		const steps = [];
		let battle = null, cursor = 0, pending = [];
		const drain = () => {
			const slice = battle.log.slice(cursor); cursor = battle.log.length;
			if (!slice.length) return;
			const lines = R.extractChannelMessages(slice.join('\n'), [chan])[chan];
			pending.push(...lines);
			tr.feed(lines);
		};
		let ok = true;
		try {
			const res = R.replayInputlog(j.inputlog, {
				anonymousNames: true,
				onDecision: (sid, req, choice, b) => {
					battle = b; drain();
					if (sid !== me) return;
					const obs = encode(req, tr);
					steps.push({
						lines: pending.slice(),
						request: req,
						ids: Array.from(obs.ids),
						scalars: Array.from(obs.scalars),
						mask: legalMask(req).map(Number),
					});
					pending = [];
				},
			});
			if (!res || res.truncated) ok = false;
			if (res && res.battle) res.battle.destroy();
		} catch { ok = false; }
		if (!ok || !steps.length) continue;
		out.write(JSON.stringify({ room, perspective: me, steps }) + '\n');
		decisions += steps.length;
	}
	battles++;
	if (battles % 50 === 0) console.log(`  ...${battles}/${rows.length}`);
}
out.end();
console.log(`wrote ${OUT}: ${battles} battles, ${decisions.toLocaleString()} decision points`);
