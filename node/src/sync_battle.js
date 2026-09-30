'use strict';
/**
 * Synchronous battle driver.
 *
 * battle.choose() runs commitChoices()/turnLoop() inline, and getRequests() populates
 * side.activeRequest before sendUpdates() is ever called - so no async layer is needed.
 * Per-player views come from extractChannelMessages on the log slice.
 */
const PS = require('./ps_dir');
const { Battle } = require(`${PS}/dist/sim`);
const { extractChannelMessages } = require(`${PS}/dist/sim/battle`);
const { legalMask, toChoice } = require('./actions');

class SyncBattle {
	constructor({ formatid, seed, p1seed, p2seed, p1team, p2team, names }) {
		this.logCursor = 0;
		const p1 = { name: (names && names[0]) || 'p1' };
		const p2 = { name: (names && names[1]) || 'p2' };
		if (p1team) p1.team = p1team; else p1.seed = p1seed;
		if (p2team) p2.team = p2team; else p2.seed = p2seed;
		// strictChoices: an illegal choice throws instead of silently stalling, so a
		// masking bug fails loudly.
		this.battle = new Battle({ formatid, seed, strictChoices: true, p1, p2 });
	}

	get ended() { return this.battle.ended; }
	get turn() { return this.battle.turn; }
	/** 1 => p1 won, -1 => p2 won, 0 => tie. */
	get result() {
		if (!this.battle.ended) return null;
		const w = this.battle.winner;
		if (!w) return 0;
		return w === this.battle.sides[0].name ? 1 : -1;
	}

	requests() { return this.battle.sides.map(s => s.activeRequest); }
	masks() { return this.requests().map(legalMask); }
	/** Which sides must act this step. */
	toAct() { return this.requests().map(r => !!(r && !r.wait)); }

	/** Protocol lines emitted since the last call, per channel. */
	drainViews() {
		const slice = this.battle.log.slice(this.logCursor);
		this.logCursor = this.battle.log.length;
		if (!slice.length) return { p1: [], p2: [], omniscient: [] };
		const ch = extractChannelMessages(slice.join('\n'), [-1, 1, 2]);
		return { omniscient: ch[-1], p1: ch[1], p2: ch[2] };
	}

	/**
	 * Apply one action index per side (null where that side need not act).
	 *
	 * Trapping is deliberately hidden until you try to switch: the sim answers with
	 * `[Unavailable choice]` and an updated request (SIM-PROTOCOL.md). That is real
	 * information the agent only gets by attempting, so we surface it as a retry
	 * rather than pretending the mask knew. Already-accepted choices are preserved
	 * because Side#choose only clears its own side.
	 */
	step(a1, a2) {
		const act = this.toAct();
		const sides = ['p1', 'p2'];
		const acts = [a1, a2];
		for (let i = 0; i < 2; i++) {
			if (!act[i]) continue;
			// On a retry the other side's choice is already registered; don't re-ask it.
			if (this.battle.sides[i].isChoiceDone()) continue;
			if (acts[i] === null || acts[i] === undefined) throw new Error(`side ${sides[i]} must act`);
			try {
				this.battle.choose(sides[i], toChoice(acts[i]));
			} catch (e) {
				if (!String(e.message).startsWith('[Unavailable choice]')) {
					e.message += ` | side=${sides[i]} action=${acts[i]} choice="${toChoice(acts[i])}"` +
						` liveMask=${this.masks()[i].map(Number).join('')}`;
					throw e;
				}
				return { retrySide: i, masks: this.masks(), views: null };
			}
		}
		return { retrySide: -1, masks: null, views: this.drainViews() };
	}

	destroy() { try { this.battle.destroy(); } catch {} }
}

module.exports = { SyncBattle };
