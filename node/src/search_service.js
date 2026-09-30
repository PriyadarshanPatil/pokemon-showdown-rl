'use strict';
/**
 * Battle-owning service for Phase 6 gate 3.2 — the oracle look-ahead probe.
 *
 * encode_service.js deliberately owns no battle: on the ladder the server is the simulator. A
 * look-ahead needs the opposite, a sim it can fork, so this owns SyncBattles and exposes one
 * extra op — `expand` — that clones the live battle once per legal action, applies it, and
 * encodes each successor.
 *
 * **This is an ORACLE and is not deployable.** The clone carries the true hidden state and the
 * live PRNG, so stepping it returns exactly what WOULD happen for that action, damage roll and
 * all. A real agent cannot know any of that. The gate's question is whether even a searcher with
 * perfect one-step foresight beats greedy by >= 4 points; if it does not, no honest search will.
 *
 * The encoding stays honest regardless: successors are encoded through a Tracker fed only with
 * our own protocol channel, so the observation the value net scores is one a real agent could
 * have built. Only the choice of which futures to look at is privileged.
 *
 * Framing is newline-delimited JSON, as in encode_service.js.
 *   stdin  {op:'new'|'obs'|'expand'|'commit'|'drop'|'ping', game, ...}
 *   stdout {ok:true, ...} | {ok:false, error}
 */
const readline = require('readline');
const PS = require('./ps_dir');
const { State } = require(`${PS}/dist/sim/state.js`);
const { SyncBattle } = require('./sync_battle');
const { seedsFor, mulberry32 } = require('./seeds');
const { Tracker } = require('./tracker');
const { encode, N_IDS, N_SCALARS } = require('./encode');
const { legalMask, N_ACTIONS } = require('./actions');
const { policies } = require('./baselines');

/** game id -> {sb, trackers, rng, opponent, pendingFoe, retrying} */
const games = new Map();
const TURN_CAP = 300;   // as worker.js: truncate a stall to a tie rather than let it dominate

const legalOf = mask => { const o = []; for (let i = 0; i < mask.length; i++) if (mask[i]) o.push(i); return o; };

function feed(g, views) {
	if (!views) return;
	g.trackers[0].feed(views.p1);
	g.trackers[1].feed(views.p2);
}

/** Our side's observation at the current decision, or null when we are not to act. */
function obsOf(g) {
	const req = g.sb.requests()[0];
	if (!req || req.wait) return null;
	const o = encode(req, g.trackers[0]);
	return { ids: Array.from(o.ids), scalars: Array.from(o.scalars), mask: legalMask(req).map(Number) };
}

/** The scripted opponent's action for p2, or null when it need not act. */
function foeAction(g) {
	const req = g.sb.requests()[1];
	if (!req || req.wait) return null;
	return policies[g.opponent](req, g.trackers[1], legalMask(req), g.rng);
}

function handle(cmd) {
	switch (cmd.op) {
	case 'new': {
		const sb = new SyncBattle({ formatid: 'gen9randombattle', ...seedsFor(cmd.tag || 'g32', cmd.ep | 0) });
		const g = {
			sb,
			trackers: [new Tracker('p1'), new Tracker('p2')],
			rng: mulberry32(((cmd.ep | 0) * 2654435761) >>> 0),
			opponent: cmd.opponent || 'heuristic',
		};
		if (!policies[g.opponent]) throw new Error(`unknown opponent ${g.opponent}`);
		games.set(cmd.game, g);
		feed(g, sb.drainViews());        // only here: no step has run yet
		return { game: cmd.game, turn: sb.turn, ended: sb.ended, obs: obsOf(g) };
	}
	case 'obs': {
		const g = games.get(cmd.game);
		if (!g) throw new Error(`unknown game ${cmd.game}`);
		return { game: cmd.game, turn: g.sb.turn, ended: g.sb.ended, result: g.sb.result, obs: obsOf(g) };
	}
	case 'expand': {
		// One clone per legal action of ours, each stepped against the opponent's actual reply.
		const g = games.get(cmd.game);
		if (!g) throw new Error(`unknown game ${cmd.game}`);
		const req = g.sb.requests()[0];
		if (!req || req.wait) throw new Error('expand called when we are not to act');
		const mine = legalOf(legalMask(req));
		const a2 = foeAction(g);
		const out = [];
		for (const a of mine) {
			const copy = State.deserializeBattle(State.serializeBattle(g.sb.battle));
			const sim = Object.create(SyncBattle.prototype);
			sim.battle = copy;
			sim.logCursor = copy.log.length;
			// step() RETURNS the drained views; draining separately would see nothing. It also
			// signals retrySide when a choice turns out to be unavailable (trapping), which is
			// real information the mask cannot carry.
			let ok = true, views = null, b2 = a2;
			for (let tries = 0; tries < 5; tries++) {
				let r;
				try { r = sim.step(a, b2); } catch { ok = false; break; }
				if (r.retrySide < 0) { views = r.views; break; }
				if (r.retrySide === 1) {
					// deterministic re-pick, so expand never advances the game's own rng
					const legal2 = legalOf(r.masks[1]);
					if (!legal2.length) { ok = false; break; }
					b2 = legal2[0];
					continue;
				}
				ok = false; break;            // our action is genuinely unavailable
			}
			// The successor is scored through a tracker fed only with OUR channel, carrying our
			// real knowledge forward, so the observation stays one a real agent could build.
			const tr = cloneTracker(g.trackers[0]);
			if (views) tr.feed(views.p1);
			// Run the clone on to OUR next decision. Without this, a successor where the foe has
			// to replace a KO'd Pokemon has no request to encode and would be scored as a draw -
			// penalising exactly the moves that got the KO.
			const localRng = mulberry32((a * 2654435761) >>> 0);
			for (let hops = 0; hops < 8 && ok && !sim.ended; hops++) {
				const q = sim.requests()[0];
				if (q && !q.wait) break;
				const q2 = sim.requests()[1];
				if (!q2 || q2.wait) break;             // neither side to act: nothing to advance
				const m2 = legalMask(q2);
				let c2 = policies[g.opponent](q2, g.trackers[1], m2, localRng);
				let r3;
				try { r3 = sim.step(null, c2); } catch { ok = false; break; }
				if (r3.retrySide >= 0) {
					const legal2 = legalOf(r3.masks[1]);
					if (!legal2.length) { ok = false; break; }
					try { r3 = sim.step(null, legal2[0]); } catch { ok = false; break; }
					if (r3.retrySide >= 0) { ok = false; break; }
				}
				if (r3.views) tr.feed(r3.views.p1);
			}
			const r2 = sim.requests()[0];
			const rec = { action: a, ok, ended: sim.ended, result: sim.result };
			if (!sim.ended && r2 && !r2.wait) {
				const o = encode(r2, tr);
				rec.ids = Array.from(o.ids);
				rec.scalars = Array.from(o.scalars);
				rec.mask = legalMask(r2).map(Number);
			}
			out.push(rec);
		}
		return { game: cmd.game, foeAction: a2, successors: out };
	}
	case 'commit': {
		const g = games.get(cmd.game);
		if (!g) throw new Error(`unknown game ${cmd.game}`);
		const a1 = cmd.action === undefined ? null : cmd.action;
		// On a retry our opponent's choice is already registered, so do not re-draw it - that
		// would advance the rng and desync from the successors expand just scored.
		let a2 = g.retrying ? g.pendingFoe
			: (cmd.foeAction === undefined ? foeAction(g) : cmd.foeAction);
		g.retrying = false;
		for (let tries = 0; tries < 20; tries++) {
			const r = g.sb.step(a1, a2);
			if (r.retrySide < 0) { feed(g, r.views); break; }
			if (r.retrySide === 1) {
				a2 = policies[g.opponent](g.sb.requests()[1], g.trackers[1], r.masks[1], g.rng);
				continue;
			}
			// Our own choice was unavailable. Hand the live mask back and let the caller re-pick.
			g.retrying = true; g.pendingFoe = a2;
			return { game: cmd.game, retry: true, mask: r.masks[0].map(Number),
				turn: g.sb.turn, ended: g.sb.ended };
		}
		if (!g.sb.ended && g.sb.turn > TURN_CAP) g.sb.battle.tie();
		g.pendingFoe = null;
		return { game: cmd.game, turn: g.sb.turn, ended: g.sb.ended, result: g.sb.result, obs: obsOf(g) };
	}
	case 'drop': {
		const g = games.get(cmd.game);
		if (g) g.sb.battle.destroy();
		games.delete(cmd.game);
		return { game: cmd.game, remaining: games.size };
	}
	case 'ping':
		return { n_ids: N_IDS, n_scalars: N_SCALARS, n_actions: N_ACTIONS, games: games.size };
	default:
		throw new Error(`unknown op ${cmd.op}`);
	}
}

/** A Tracker carrying the same history, so a successor is encoded from our real knowledge. */
function cloneTracker(tr) {
	const c = new Tracker(tr.me || 'p1');
	// Tracker state is plain data; a structured clone is both correct and cheaper than replaying
	// the protocol, which we no longer hold.
	for (const k of Object.keys(tr)) {
		const v = tr[k];
		c[k] = (v && typeof v === 'object') ? structuredClone(v) : v;
	}
	return c;
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', line => {
	if (!line.trim()) return;
	let res;
	try {
		res = { ok: true, ...handle(JSON.parse(line)) };
	} catch (e) {
		res = { ok: false, error: String((e && e.stack) || e) };
	}
	process.stdout.write(JSON.stringify(res) + '\n');
});
rl.on('close', () => process.exit(0));
process.stderr.write(`search_service ready: ids=${N_IDS} scalars=${N_SCALARS} actions=${N_ACTIONS}\n`);
