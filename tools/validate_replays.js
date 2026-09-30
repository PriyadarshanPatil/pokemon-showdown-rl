'use strict';
/**
 * Validate replay reconstruction against the stored protocol log.
 *
 * Each replay is replayed with the era-correct sets.json. A reconstruction counts as
 * EXACT when the normalised logs match, and SEMANTIC when only cross-version protocol
 * cosmetics differ (bracket annotations, init ordering) - those are not evidence that
 * the reconstruction is wrong, so both count as usable.
 */
const fs = require('fs');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const R = require('../node/src/replay');
const { normalize, skeleton, redact } = require('../node/src/battle_log');
const { userHash } = require('../node/src/seeds');

// Defaults to the reconstructable era. Ordering by uploadtime across the whole table
// would otherwise start with pre-2025-07 replays, which cannot be reproduced at all
// (they need an era-matched teams.ts, not just era-matched sets.json) and report 0%.
const LIMIT = parseInt(process.argv[2] || '1000000');
const ALL = process.argv.includes('--all');
// Ordering by uploadtime alone samples one narrow window, and "exact" is sensitive to
// which deployed simulator versions the sample spans. --spread thins across the era.
const SPREAD = process.argv.includes('--spread');
const db = R.openDb(DatabaseSync);
const rows = db.prepare(
	`SELECT b.id id, b.z z, r.uploadtime t FROM bodies b JOIN replays r ON r.id=b.id
	 WHERE r.uploadtime >= ?${SPREAD ? ' AND (r.uploadtime % 997) < 2' : ''}
	 ORDER BY r.uploadtime LIMIT ?`).all(ALL ? 0 : R.ERA_START, LIMIT);
console.log(`validating ${rows.length} replays (${ALL ? 'all eras' : 'reconstructable era only'})`);

const offsets = {}, byMonth = {};
const diverges = [];
let match = 0, semantic = 0, diverge = 0, error = 0, n = 0;

for (const row of rows) {
	const j = R.decodeBody(row.z);
	if (!j.inputlog) continue;
	n++;
	const names = (j.players || []).slice().sort((a, b) => b.length - a.length);
	const inputlog = redact(j.inputlog, names, userHash);
	const want = normalize(redact(j.log, names, userHash).split('\n'), false);
	const base = R.versionIndexAt(row.t);

	// Deployment lags the commit, so try the dated revision then its neighbours.
	const rank = { MATCH: 0, SEMANTIC: 1, DIVERGE: 2 };
	let status = null, usedOff = null;
	for (const off of [0, -1, -2, 1, -3]) {
		const vi = base + off;
		if (vi < 0 || vi >= R.versions().length) continue;
		R.useSetsVersion(vi);
		let got = null, verdict;
		let out = null;
		try {
			out = R.replayInputlog(inputlog);
			got = normalize(out.battle.log, true);
			verdict = got === want ? 'MATCH' : (skeleton(got) === skeleton(want) ? 'SEMANTIC' : 'DIVERGE');
		} catch (e) {
			verdict = 'ERROR:' + String(e.message || '').slice(0, 70);
		} finally { if (out && out.battle) try { out.battle.destroy(); } catch {} }
		if (verdict === 'MATCH') { status = 'MATCH'; usedOff = off; break; }
		const cur = status === null ? 99 : (rank[status] ?? 3);
		if ((rank[verdict] ?? 3) < cur) { status = verdict; usedOff = off; }
	}

	const mk = new Date(row.t * 1000).toISOString().slice(0, 7);
	byMonth[mk] = byMonth[mk] || { m: 0, s: 0, d: 0, e: 0 };
	if (status === 'MATCH') { match++; byMonth[mk].m++; offsets[usedOff] = (offsets[usedOff] || 0) + 1; }
	else if (status === 'SEMANTIC') { semantic++; byMonth[mk].s++; }
	else if (status === 'DIVERGE') { diverge++; byMonth[mk].d++; diverges.push({ id: j.id, t: row.t }); }
	else { error++; byMonth[mk].e++; }
	if (n % 200 === 0) console.log(`  ...${n}/${rows.length}`);
}

const usable = match + semantic;
console.log(`\nOVERALL: exact ${match}/${n} (${(match / n * 100).toFixed(1)}%) | +semantic ${semantic}` +
	` => USABLE ${usable}/${n} (${(usable / n * 100).toFixed(1)}%) | diverge ${diverge} | error ${error}`);
console.log('\nby month:');
for (const k of Object.keys(byMonth).sort()) {
	const b = byMonth[k], tot = b.m + b.s + b.d + b.e;
	console.log(`  ${k}  exact ${b.m}/${tot} (${(b.m / tot * 100).toFixed(0)}%)  usable ${((b.m + b.s) / tot * 100).toFixed(0)}%  s=${b.s} d=${b.d} e=${b.e}`);
}
console.log('\nwhich revision worked (0 = the one dated at/before the replay):');
for (const k of Object.keys(offsets).sort((a, b) => offsets[b] - offsets[a])) {
	console.log(`  offset ${k}: ${offsets[k]} (${(offsets[k] / match * 100).toFixed(1)}%)`);
}
fs.writeFileSync(path.join(R.REPO, 'data', 'diverges.json'), JSON.stringify(diverges, null, 1));
