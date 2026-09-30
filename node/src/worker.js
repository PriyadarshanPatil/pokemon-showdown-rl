'use strict';
/**
 * Batched simulator worker (Phase 1 step 8).
 *
 * Holds B self-play battles and steps them in lockstep at the decision barrier.
 * Talks length-prefixed binary frames over stdin/stdout so the hot loop never touches
 * JSON or protocol text. Slot layout is [battle0 p1, battle0 p2, battle1 p1, ...].
 *
 * stdout is the data channel ONLY. Diagnostics go to stderr.
 */
const { seedsFor } = require('./seeds');
const { SyncBattle } = require('./sync_battle');
const { Tracker } = require('./tracker');
const { encode, N_IDS, N_SCALARS } = require('./encode');
const { N_ACTIONS } = require('./actions');
const { policies } = require('./baselines');

const argv = Object.fromEntries(process.argv.slice(2).map(a => {
	const [k, v] = a.replace(/^--/, '').split('='); return [k, v === undefined ? true : v];
}));
const B = parseInt(argv.batch || '32');
const FORMAT = argv.format || 'gen9randombattle';
const BASE_SEED = String(argv.seed || 'psrl');
const TURN_CAP = parseInt(argv.turnCap || '300');
const N_SLOTS = B * 2;
// When set, side p2 is driven by a scripted bot inside the worker, so evaluation of a
// Python policy against a baseline costs no extra IPC. p2 slots then never request an
// action (needsAction stays false) and Python only supplies p1.
const OPPONENT = argv.opponent && argv.opponent !== 'self' ? String(argv.opponent) : null;
// Ablation: per-step HP-differential shaping, off by default. Reward shaping is the
// classic way to get results in this domain that look better than they are, so it is
// opt-in and reported as its own arm rather than folded into the baseline.
const SHAPING = parseFloat(argv.shaping || '0');
if (OPPONENT && !policies[OPPONENT]) {
	process.stderr.write(`unknown opponent '${OPPONENT}'\n`); process.exit(1);
}
let botSeed = 12345;
const botRng = () => { botSeed ^= botSeed << 13; botSeed ^= botSeed >>> 17; botSeed ^= botSeed << 5; return ((botSeed >>> 0) % 1e6) / 1e6; };

class Slot {
	constructor(idx) { this.idx = idx; this.episode = idx; this.reset(); }
	reset() {
		this.b = new SyncBattle({ formatid: FORMAT, ...seedsFor(BASE_SEED, this.episode) });
		this.tr = [new Tracker('p1'), new Tracker('p2')];
		const v = this.b.drainViews();
		this.tr[0].feed(v.p1); this.tr[1].feed(v.p2);
		this.pending = [null, null];      // actions accumulated for this decision point
		this.retrySide = -1;
		this.justEnded = false;   // ended on the step just taken
		this.reported = false;    // terminal frame has been sent to Python
		this.result = 0;
		this.shaped = [0, 0];     // per-step shaping reward, consumed by buildFrame
		this.hpFrac = this.teamHp();
	}
	/** Summed HP fraction for each side, from the omniscient battle state. */
	teamHp() {
		return this.b.battle.sides.map(side =>
			side.pokemon.reduce((a, p) => a + (p.hp > 0 ? p.hp / Math.max(1, p.maxhp) : 0), 0) / 6);
	}

	needsAction() {
		if (this.b.ended) return [false, false];
		const act = this.retrySide >= 0
			? [this.retrySide === 0, this.retrySide === 1]
			: this.b.toAct();
		if (OPPONENT) act[1] = false;   // p2 is scripted in-worker
		return act;
	}
	/** Scripted action for p2, chosen from the live request. */
	botAction() {
		const req = this.b.requests()[1];
		if (!req || req.wait) return null;
		const mask = this.b.masks()[1];
		if (!mask.some(Boolean)) return null;
		return policies[OPPONENT](req, this.tr[1], mask, botRng);
	}
	masks() { return this.b.masks(); }
	obs(i) { return encode(this.b.requests()[i], this.tr[i]); }
	/** Apply the collected actions; returns true if the battle advanced. */
	apply(a0, a1) {
		if (this.b.ended) return false;
		// Guard: never hand the simulator an action the LIVE mask forbids. This fires
		// rarely (it did not reproduce in a 250-iteration run) but killing a multi-hour
		// training run over it is worse than substituting a legal action and counting it.
		// The count is reported on exit so the problem stays visible rather than silent.
		const live = this.b.masks();
		const fixed = [a0, a1];
		for (const i of [0, 1]) {
			const a = fixed[i];
			if (a === null || a === undefined) continue;
			if (live[i][a]) continue;
			globalThis.__illegal = (globalThis.__illegal || 0) + 1;
			if (globalThis.__illegal <= 3) {
				const framed = this.framedMask && this.framedMask[i];
				process.stderr.write(
					`ILLEGAL ACTION #${globalThis.__illegal} side=p${i + 1} action=${a}\n` +
					`  liveMask   = ${live[i].map(Number).join('')}\n` +
					`  framedMask = ${framed ? framed.map(Number).join('') : '(none)'}\n` +
					`  stale=${framed ? (framed.join() !== live[i].join()) : 'unknown'} ` +
					`retrySide=${this.retrySide} reqState=${this.b.battle.requestState}\n`);
			}
			const legal = live[i].indexOf(true);
			if (legal < 0) throw new Error(`side p${i + 1} has no legal action at all`);
			fixed[i] = legal;
		}
		a0 = fixed[0]; a1 = fixed[1];
		if (this.retrySide >= 0) {
			this.pending[this.retrySide] = this.retrySide === 0 ? a0 : a1;
		} else {
			this.pending = [a0, a1];
		}
		const r = this.b.step(this.pending[0], this.pending[1]);
		if (r.retrySide >= 0) {
			this.retrySide = r.retrySide;
			// a scripted p2 can re-pick immediately; only p1 needs another round trip
			if (OPPONENT && r.retrySide === 1) {
				this.pending[1] = this.botAction();
				return this.apply(this.pending[0], this.pending[1]);
			}
			return false;
		}
		this.retrySide = -1;
		if (r.views) { this.tr[0].feed(r.views.p1); this.tr[1].feed(r.views.p2); }
		if (!this.b.ended && this.b.turn > TURN_CAP) {
			// truncate: treat as a draw so long stalls do not dominate the batch
			this.b.battle.tie();
		}
		if (SHAPING) {
			const now = this.teamHp();
			// damage dealt minus damage taken, from each side's own perspective
			const d0 = (now[0] - this.hpFrac[0]) - (now[1] - this.hpFrac[1]);
			this.shaped = [SHAPING * d0, -SHAPING * d0];
			this.hpFrac = now;
		}
		if (this.b.ended) { this.justEnded = true; this.result = this.b.result; }
		return true;
	}
	recycle() {
		this.b.destroy();
		this.episode += B * 1000;   // keep episode streams disjoint across slots
		this.reset();               // clears justEnded and reported
	}
}

const slots = Array.from({ length: B }, (_, i) => new Slot(i));

// ---- framing ----------------------------------------------------------------
const MAGIC = 0x50535231; // "PSR1"
function writeFrame(buf) {
	const len = Buffer.allocUnsafe(4);
	len.writeUInt32LE(buf.length, 0);
	process.stdout.write(len);
	process.stdout.write(buf);
}

const OBS_BYTES = N_IDS * 4 + N_SCALARS * 4;
const PER_SLOT = OBS_BYTES + N_ACTIONS + 1 + 4 + 1;   // + mask + needsAction + reward + done

function buildFrame() {
	const head = Buffer.allocUnsafe(24);
	head.writeUInt32LE(MAGIC, 0);
	head.writeUInt32LE(N_SLOTS, 4);
	head.writeUInt32LE(N_IDS, 8);
	head.writeUInt32LE(N_SCALARS, 12);
	head.writeUInt32LE(N_ACTIONS, 16);
	head.writeUInt32LE(0, 20);
	const body = Buffer.allocUnsafe(N_SLOTS * PER_SLOT);
	let o = 0;
	for (const s of slots) {
		const need = s.needsAction();
		const masks = s.b.ended ? [null, null] : s.masks();
		// remember exactly what we told Python, so a mismatch at apply time is provable
		s.framedMask = masks.map(m => (m ? m.slice() : null));
		for (let i = 0; i < 2; i++) {
			if (need[i]) {
				const ob = s.obs(i);
				Buffer.from(ob.ids.buffer, ob.ids.byteOffset, N_IDS * 4).copy(body, o);
				o += N_IDS * 4;
				Buffer.from(ob.scalars.buffer, ob.scalars.byteOffset, N_SCALARS * 4).copy(body, o);
				o += N_SCALARS * 4;
				const m = masks[i];
				for (let k = 0; k < N_ACTIONS; k++) body[o + k] = m && m[k] ? 1 : 0;
				o += N_ACTIONS;
			} else {
				body.fill(0, o, o + OBS_BYTES + N_ACTIONS);
				o += OBS_BYTES + N_ACTIONS;
			}
			body[o++] = need[i] ? 1 : 0;
			// terminal reward only, +1/-1/0 from that side's perspective
			const rew = (s.justEnded ? (i === 0 ? s.result : -s.result) : 0) + (SHAPING ? s.shaped[i] : 0);
			body.writeFloatLE(rew, o); o += 4;
			body[o++] = s.justEnded ? 1 : 0;
		}
	}
	return Buffer.concat([head, body]);
}

function stepAll(actions) {
	for (const s of slots) {
		// Recycle only AFTER the terminal frame has been observed by Python, so the
		// final transition is never dropped. Keyed on `reported`, not `justEnded`,
		// because justEnded is cleared as soon as that frame is written.
		if (s.b.ended && s.reported) { s.recycle(); continue; }
		const need = s.needsAction();
		const rawAct = s.b.toAct();
		// In self-play a slot must still advance when only p2 needs to act (e.g. p1 is
		// waiting through a forced switch). Skipping those stalls the battle forever.
		const p2Scripted = !!(OPPONENT && rawAct[1]);
		if (!need[0] && !need[1] && !p2Scripted) continue;
		const a0 = need[0] ? actions[s.idx * 2] : null;
		const a1 = OPPONENT ? (rawAct[1] ? s.botAction() : null)
			: (need[1] ? actions[s.idx * 2 + 1] : null);
		globalThis.__lastCtx = JSON.stringify({
			slot: s.idx, a0, a1, need, rawAct, retrySide: s.retrySide, pending: s.pending,
			reqState: s.b.battle.requestState,
			masks: s.b.masks().map(m => m.map(Number).join('')),
			p1: s.b.requests()[0] && s.b.requests()[0].side.pokemon.map((p, k) => `${k + 1}:${p.ident.split(': ')[1]}|${p.condition}|act=${!!p.active}`),
			p2: s.b.requests()[1] && s.b.requests()[1].side.pokemon.map((p, k) => `${k + 1}:${p.ident.split(': ')[1]}|${p.condition}|act=${!!p.active}`),
			fs1: s.b.requests()[0] && s.b.requests()[0].forceSwitch,
			fs2: s.b.requests()[1] && s.b.requests()[1].forceSwitch,
		});
		s.apply(a0, a1);
	}
	// any slot that ended this step keeps justEnded until the NEXT call, so Python
	// observes the terminal transition before the slot is recycled
}

// ---- stdin loop -------------------------------------------------------------
let inbuf = Buffer.alloc(0);
process.stdin.on('data', chunk => {
	inbuf = Buffer.concat([inbuf, chunk]);
	for (;;) {
		if (inbuf.length < 4) return;
		const len = inbuf.readUInt32LE(0);
		if (inbuf.length < 4 + len) return;
		const payload = inbuf.subarray(4, 4 + len);
		inbuf = inbuf.subarray(4 + len);
		if (len === 0) { writeFrame(buildFrame()); continue; }        // RESET / observe
		const actions = new Int32Array(payload.buffer, payload.byteOffset, len / 4);
		try {
			stepAll(actions);
		} catch (e) {
			process.stderr.write(`worker fatal: ${e.stack}\n`);
			if (globalThis.__lastCtx) process.stderr.write(`context: ${globalThis.__lastCtx}\n`);
			process.exit(1);
		}
		writeFrame(buildFrame());
		for (const s of slots) {
			if (s.justEnded) { s.justEnded = false; s.reported = true; }
			s.shaped = [0, 0];   // shaping reward is consumed exactly once
		}
	}
});
process.on('exit', () => {
	if (globalThis.__illegal) {
		process.stderr.write(`worker: substituted ${globalThis.__illegal} illegal client actions\n`);
	}
});
process.stdin.on('end', () => process.exit(0));
process.stderr.write(`worker ready: opponent=${OPPONENT || 'self'} shaping=${SHAPING} batch=${B} slots=${N_SLOTS} ids=${N_IDS} scalars=${N_SCALARS} actions=${N_ACTIONS} perSlot=${PER_SLOT}B\n`);
