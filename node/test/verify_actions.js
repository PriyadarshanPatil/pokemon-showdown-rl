'use strict';
/**
 * Gate: the legality mask must never propose an action the simulator rejects.
 *
 * Probing mutates shared choice state (Side#choose calls clearChoice, which recomputes
 * forced-switch accounting - during Revival Blessing switchFlag is already false, so
 * clearing makes a legal revival look illegal). So each probe runs against a FRESH
 * toJSON/fromJSON clone and the live battle is never touched. That also exercises state
 * serialization.
 *
 * `[Unavailable choice]` is not a failure: trapping is hidden by design until you try.
 * Being STRICTER than the simulator is safe and sometimes deliberate (we forbid
 * `terastallize` whenever the request does not advertise canTerastallize, even though
 * the simulator tolerates the suffix), so those are reported, not failed.
 */
const PS = require('../src/ps_dir');
const { Battle } = require(`${PS}/dist/sim`);
const { N_ACTIONS, legalMask, toChoice } = require('../src/actions');
const { playBattle, randomPick } = require('./harness');

const N = parseInt(process.argv[2] || '400');
const FORMAT = process.argv[3] || 'gen9randombattle';
const PROBE_RATE = parseFloat(process.argv[4] || '0.04');

let probes = 0, falseNeg = 0, stricter = 0, unavailable = 0, decisions = 0, clones = 0, cloneFail = 0;
const fnEx = {}, fpEx = {};

for (let ep = 0; ep < N; ep++) {
	const r = playBattle({
		format: FORMAT, tag: 'actions', ep, rngSeed: ep * 2246822519, pick: randomPick,
		onDecision: (i, ctx) => {
			if (ctx.rng() >= PROBE_RATE) return;
			decisions++;
			const snapshot = ctx.battle.battle.toJSON();
			for (let a = 0; a < N_ACTIONS; a++) {
				let clone = null;
				try { clone = Battle.fromJSON(snapshot); clones++; }
				catch { cloneFail++; continue; }
				const side = clone.sides[i];
				const mask = legalMask(side.activeRequest);
				probes++;
				let accepted = false, unavail = false, why = '';
				try {
					accepted = side.choose(toChoice(a)) !== false;
					if (!accepted) why = side.choice.error || '(none)';
				} catch (e) {
					const m = String(e.message);
					if (m.startsWith('[Unavailable choice]')) unavail = true;
					why = m; accepted = false;
				}
				if (mask[a] && !accepted && !unavail) {
					falseNeg++;
					const k = `a=${a} ${why.replace(/Your [^ ]+ /, 'Your <mon> ').slice(0, 80)}`;
					fnEx[k] = (fnEx[k] || 0) + 1;
				}
				if (!mask[a] && accepted) {
					stricter++;
					const ac = side.activeRequest && side.activeRequest.active && side.activeRequest.active[0];
					const k = `a=${a} canTera=${JSON.stringify(ac && ac.canTerastallize)} nMoves=${ac && ac.moves && ac.moves.length}`;
					fpEx[k] = (fpEx[k] || 0) + 1;
				}
				if (unavail) unavailable++;
				try { clone.destroy(); } catch {}
			}
		},
	});
	r.battle.destroy();
}

const top = (o, n) => Object.entries(o).sort((a, b) => b[1] - a[1]).slice(0, n);
console.log(`${FORMAT}: ${N} battles, ${clones} clones (${cloneFail} failures), ${decisions} probed decision points, ${probes} probes`);
console.log(`  UNSAFE false negatives (mask allowed, sim refused): ${falseNeg}`);
console.log(`  stricter than sim (safe): ${stricter}`);
console.log(`  [Unavailable choice] (hidden trapping, expected): ${unavailable}`);
if (falseNeg) { console.log('\n  FALSE NEGATIVE causes:'); top(fnEx, 6).forEach(([k, v]) => console.log(`    ${v}x  ${k}`)); }
if (stricter) { console.log('\n  stricter-than-sim cases:'); top(fpEx, 4).forEach(([k, v]) => console.log(`    ${v}x  ${k}`)); }
process.exitCode = (falseNeg || cloneFail) ? 1 : 0;
