'use strict';
/**
 * How often does each scripted policy predict the human's actual action?
 * This is the reference behaviour cloning must beat; without it a BC accuracy number
 * is uninterpretable.
 */
const { DatabaseSync } = require('node:sqlite');
const R = require('../node/src/replay');
const { Tracker } = require('../node/src/tracker');
const { legalMask, fromChoice, N_ACTIONS, TERA_OFFSET, SWITCH_OFFSET } = require('../node/src/actions');
const { policies } = require('../node/src/baselines');

const N = parseInt(process.argv[2] || '500');
const db = R.openDb(DatabaseSync);
const rows = db.prepare(
	`SELECT b.id id, b.z z, r.uploadtime t FROM bodies b JOIN replays r ON r.id=b.id
	 WHERE r.has_inputlog=1 AND r.uploadtime >= ? ORDER BY r.id LIMIT ?`).all(R.ERA_START, N);

const rng = () => 0.5;                       // deterministic tie-breaking
const agree = { random: 0, maxdamage: 0, heuristic: 0 };
const kind = { move: 0, tera: 0, switch: 0 };
const prior = new Array(N_ACTIONS).fill(0);
let total = 0, legalSum = 0, ok = 0;

for (const row of rows) {
	const j = R.decodeBody(row.z);
	if (!j.inputlog) continue;
	R.useSetsAt(row.t);
	const tr = { p1: new Tracker('p1'), p2: new Tracker('p2') };
	let cursor = 0, battle = null;
	const drain = () => {
		const slice = battle.log.slice(cursor); cursor = battle.log.length;
		if (!slice.length) return;
		const ch = R.extractChannelMessages(slice.join('\n'), [1, 2]);
		tr.p1.feed(ch[1]); tr.p2.feed(ch[2]);
	};
	try {
		const out = R.replayInputlog(j.inputlog, {
			onDecision: (sid, req, choice, b) => {
				battle = b; drain();
				const a = fromChoice(choice, req);
				const mask = legalMask(req);
				if (a < 0 || !mask[a]) return;
				total++; legalSum += mask.filter(Boolean).length; prior[a]++;
				kind[a < TERA_OFFSET ? 'move' : a < SWITCH_OFFSET ? 'tera' : 'switch']++;
				for (const name of Object.keys(agree)) {
					if (policies[name](req, tr[sid], mask, rng) === a) agree[name]++;
				}
			},
		});
		if (out && !out.truncated) ok++;
		if (out && out.battle) out.battle.destroy();
	} catch { /* drop this replay */ }
}

console.log(`${ok}/${rows.length} replays walked | ${total.toLocaleString()} human decisions`);
console.log(`mean legal actions: ${(legalSum / total).toFixed(2)}  =>  uniform-random accuracy ~ ${(100 / (legalSum / total)).toFixed(1)}%\n`);
console.log('human action mix:', Object.entries(kind).map(([k, v]) => `${k} ${(v / total * 100).toFixed(1)}%`).join('  '));
console.log(`most common single action: ${(Math.max(...prior) / total * 100).toFixed(1)}%\n`);
console.log('AGREEMENT WITH HUMAN ACTIONS');
for (const [k, v] of Object.entries(agree)) console.log(`  ${k.padEnd(12)} ${(v / total * 100).toFixed(1)}%`);
