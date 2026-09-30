'use strict';
/**
 * Phase 0 steps 3-4: reproducibility probe + throughput benchmark.
 * Reference path only: BattleStream + getPlayerStreams + RandomPlayerAI.
 */
const crypto = require('crypto');
const { seedsFor } = require('../node/src/seeds');
const PS = require('../node/src/ps_dir');
const Sim = require(`${PS}/dist/sim`);
const { RandomPlayerAI } = require(`${PS}/dist/sim/tools/random-player-ai`);
const { State } = require(`${PS}/dist/sim/state`);

function hex(n) { return crypto.randomBytes(n).toString('hex'); }
const seeds = ep => seedsFor('psrl', ep);

async function runBattle(format, ep) {
	const s = seeds(ep);
	const stream = new Sim.BattleStream();
	const streams = Sim.getPlayerStreams(stream);
	const p1 = new RandomPlayerAI(streams.p1, { seed: `sodium,${hex(16)}` });
	const p2 = new RandomPlayerAI(streams.p2, { seed: `sodium,${hex(16)}` });
	void p1.start(); void p2.start();
	void streams.omniscient.write(
		`>start ${JSON.stringify({ formatid: format, seed: s.seed })}\n` +
		`>player p1 ${JSON.stringify({ name: 'A', seed: s.p1seed })}\n` +
		`>player p2 ${JSON.stringify({ name: 'B', seed: s.p2seed })}`
	);
	for await (const _ of streams.omniscient) { /* drain */ }
	const b = stream.battle;
	const decisions = b.inputLog.filter(l => /^>p[12] /.test(l)).length;
	const log = State.normalizeLog(b.log).join('\n');
	const out = { turns: b.turn, decisions, winner: b.winner, hash: crypto.createHash('sha256').update(log).digest('hex').slice(0, 16) };
	b.destroy();
	return out;
}

// Deterministic replay: same seeds AND scripted choices -> identical log.
async function runScripted(format, ep) {
	const s = seeds(ep);
	const battle = new Sim.Battle({
		formatid: format, seed: s.seed,
		p1: { name: 'A', seed: s.p1seed }, p2: { name: 'B', seed: s.p2seed },
	});
	let guard = 0;
	while (!battle.ended && guard++ < 2000) battle.makeChoices();  // autoChoose both sides
	const log = State.normalizeLog(battle.log).join('\n');
	const out = { turns: battle.turn, winner: battle.winner, hash: crypto.createHash('sha256').update(log).digest('hex').slice(0, 16) };
	battle.destroy();
	return out;
}

(async () => {
	const mode = process.argv[2] || 'all';
	const format = process.argv[3] || 'gen9randombattle';
	const N = parseInt(process.argv[4] || '200');

	if (mode === 'repro' || mode === 'all') {
		console.log(`Config.potd = ${JSON.stringify(global.Config && global.Config.potd)}`);
		if (global.Config && global.Config.potd) throw new Error('Config.potd is set — random teams would be biased');
		console.log('--- repro probe (scripted autoChoose, fully deterministic) ---');
		for (const ep of [1, 2, 3]) {
			const a = await runScripted(format, ep);
			const b = await runScripted(format, ep);
			console.log(`  ep=${ep} turns=${a.turns} winner=${a.winner || 'tie'} hashA=${a.hash} hashB=${b.hash} ${a.hash === b.hash ? 'MATCH' : '*** MISMATCH ***'}`);
			if (a.hash !== b.hash) process.exitCode = 1;
		}
	}

	if (mode === 'bench' || mode === 'all') {
		console.log(`--- throughput: ${N} battles of ${format} (single process) ---`);
		const t0 = process.hrtime.bigint();
		let turns = 0, decisions = 0;
		for (let i = 0; i < N; i++) {
			const r = await runBattle(format, 1000 + i);
			turns += r.turns; decisions += r.decisions;
		}
		const secs = Number(process.hrtime.bigint() - t0) / 1e9;
		console.log(`  ${N} battles in ${secs.toFixed(2)}s`);
		console.log(`  ${(N / secs).toFixed(1)} battles/sec | ${(decisions / secs).toFixed(0)} decisions/sec`);
		console.log(`  mean ${(turns / N).toFixed(1)} turns, ${(decisions / N).toFixed(1)} decisions/battle`);
		if (process.env.EMIT_JSON) console.log('JSON ' + JSON.stringify({ battles: N, secs, bps: N / secs, dps: decisions / secs }));
	}
})().catch(e => { console.error(e); process.exit(1); });
