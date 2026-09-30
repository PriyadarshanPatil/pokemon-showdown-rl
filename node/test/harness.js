'use strict';
/**
 * Shared battle-driving loop for tests and tools.
 *
 * The retry dance is subtle - trapping is hidden until you attempt a switch, and the
 * simulator answers with `[Unavailable choice]` plus an updated request - so it lives in
 * exactly one place rather than being re-derived in every test.
 */
const { SyncBattle } = require('../src/sync_battle');
const { Tracker } = require('../src/tracker');
const { seedsFor, mulberry32 } = require('../src/seeds');

/**
 * @param {object}   o
 * @param {function} o.pick        (side, ctx) -> action index, where ctx has
 *                                 {request, tracker, mask, masks, battle, rng}
 * @param {function} [o.onDecision](side, ctx) called before actions are applied
 * @param {function} [o.onStep]    (views) called after each barrier
 * @returns {{battle, result, turns}}  result is +1 p1, -1 p2, 0 tie
 */
function playBattle({ format = 'gen9randombattle', tag = 'harness', ep = 0, pick,
	onDecision = null, onStep = null, turnCap = 300, guard = 2000, seedOverride = null,
	rngSeed = null }) {
	const b = new SyncBattle({ formatid: format, ...(seedOverride || seedsFor(tag, ep)) });
	const trackers = [new Tracker('p1'), new Tracker('p2')];
	// callers pass their own rngSeed where an existing golden file depends on it
	const rng = mulberry32(((rngSeed === null ? ep * 2654435761 : rngSeed)) >>> 0);
	const v0 = b.drainViews();
	trackers[0].feed(v0.p1); trackers[1].feed(v0.p2);
	if (onStep) onStep(v0);

	let n = 0;
	try {
		while (!b.ended && n++ < guard) {
			const act = b.toAct();
			const masks = b.masks();
			const reqs = b.requests();
			const ctxFor = i => ({ request: reqs[i], tracker: trackers[i], mask: masks[i], masks, battle: b, rng });
			if (onDecision) for (let i = 0; i < 2; i++) if (act[i]) onDecision(i, ctxFor(i));
			const chosen = [0, 1].map(i => (act[i] ? pick(i, ctxFor(i)) : null));

			let r = b.step(chosen[0], chosen[1]);
			let retries = 0;
			while (r.retrySide >= 0 && retries++ < 20) {
				// Keep the other side's action: when the FIRST side is rejected the second
				// was never applied and must be re-supplied.
				const i = r.retrySide;
				chosen[i] = pick(i, { request: b.requests()[i], tracker: trackers[i],
					mask: r.masks[i], masks: r.masks, battle: b, rng });
				r = b.step(chosen[0], chosen[1]);
			}
			if (r.views) {
				trackers[0].feed(r.views.p1); trackers[1].feed(r.views.p2);
				if (onStep) onStep(r.views);
			}
			if (!b.ended && b.turn > turnCap) b.battle.tie();
		}
		return { battle: b, trackers, result: b.result, turns: b.turn };
	} finally { /* caller destroys after reading result */ }
}

/** Uniformly random legal action. */
const randomPick = (i, ctx) => {
	const legal = [];
	for (let k = 0; k < ctx.mask.length; k++) if (ctx.mask[k]) legal.push(k);
	if (!legal.length) throw new Error(`no legal action for side ${i}`);
	return legal[Math.floor(ctx.rng() * legal.length)];
};

module.exports = { playBattle, randomPick };
