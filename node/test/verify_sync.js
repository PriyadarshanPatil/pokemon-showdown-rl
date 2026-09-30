'use strict';
/**
 * Gate: the synchronous driver must produce the same battle as the shipped async
 * BattleStream path. Drive SyncBattle with a seeded policy, take its inputLog, replay
 * that through BattleStream, and compare normalised logs.
 */
const PS = require('../src/ps_dir');
const Sim = require(`${PS}/dist/sim`);
const { State } = require(`${PS}/dist/sim/state`);
const { playBattle, randomPick } = require('./harness');

const N = parseInt(process.argv[2] || '1000');
const FORMAT = process.argv[3] || 'gen9randombattle';
const norm = log => State.normalizeLog(log).join('\n');

async function viaStream(inputLog) {
	// keepAlive: writeEnd() calls battle.destroy(), which wipes the log we need.
	const stream = new Sim.BattleStream({ keepAlive: true });
	const done = (async () => { for await (const _ of stream) { /* drain */ } })();
	for (const line of inputLog) await stream.write(line);
	const log = stream.battle.log.slice();
	await stream.writeEnd().catch(() => {});
	await Promise.race([done, new Promise(r => setImmediate(r))]);
	return log;
}

(async () => {
	let ok = 0, bad = 0, err = 0;
	for (let ep = 0; ep < N; ep++) {
		let inputLog, syncLog;
		try {
			const r = playBattle({ format: FORMAT, tag: 'verify', ep, pick: randomPick, turnCap: 1e9 });
			inputLog = r.battle.battle.inputLog.slice();
			syncLog = norm(r.battle.battle.log);
			r.battle.destroy();
		} catch (e) { err++; console.log(`ep=${ep} SYNC ERROR ${e.message}`); continue; }

		const streamLog = norm(await viaStream(inputLog));
		if (streamLog === syncLog) ok++;
		else {
			bad++;
			if (bad <= 2) {
				const a = syncLog.split('\n'), b = streamLog.split('\n');
				let i = 0; while (i < Math.max(a.length, b.length) && a[i] === b[i]) i++;
				console.log(`ep=${ep} MISMATCH at line ${i}\n  sync  : ${a[i]}\n  stream: ${b[i]}`);
			}
		}
	}
	console.log(`\nsync vs BattleStream over ${N} ${FORMAT} battles: ${ok} identical, ${bad} mismatch, ${err} error`);
	process.exitCode = (bad || err) ? 1 : 0;
})();
