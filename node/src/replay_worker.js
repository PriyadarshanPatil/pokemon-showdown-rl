'use strict';
/**
 * Behaviour-cloning batches streamed from human replays.
 *
 * Materialising observations would cost ~1.7 KB x 71M decisions ~= 120 GB, so nothing is
 * stored: each replay is re-simulated from its inputlog with the era-correct sets.json
 * and encoded on the fly through the SAME encoder the RL env uses. Three consequences:
 *   - zero disk, and the data can never drift from the current encoder
 *   - the emitted dataset contains no usernames at all, only numeric features
 *   - actions are ground truth from the inputlog, never inferred from the log
 *
 * Sequences, not loose samples: training with T=1 would leave the GRU untrained and the
 * weights would not transfer to sequential use in RL.
 */
const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const R = require('./replay');
const { Tracker } = require('./tracker');
const { encode, N_IDS, N_SCALARS } = require('./encode');
const { legalMask, fromChoice, N_ACTIONS } = require('./actions');

const argv = Object.fromEntries(process.argv.slice(2).map(a => {
	const [k, v] = a.replace(/^--/, '').split('='); return [k, v === undefined ? true : v];
}));
const BATCH = parseInt(argv.batch || '16');        // sequences per frame
const SEQLEN = parseInt(argv.seqlen || '64');      // steps per sequence
const SPLIT = argv.split || 'train';         // train | val, hashed on replay id
const MIN_TIME = parseInt(argv.minTime || String(R.ERA_START));
const SHARD = parseInt(argv.shard || '0');
const NSHARDS = parseInt(argv.nshards || '1');
// Imitating an undifferentiated corpus caps the policy at the corpus's own strength: the rated
// replays average 1481 and the modal bucket is 1000-1100. 0 keeps every replay, NULL ratings
// included, which is the historical behaviour and must stay byte-compatible.
const MIN_RATING = parseInt(argv.minRating || '0');
// Training wraps around the pool forever. A cache build must stop after exactly one sweep, or
// some replays land in the cache twice and others not at all.
const ONE_PASS = argv.onePass === '1' || argv.onePass === true;

const db = R.openDb(DatabaseSync);
// `r.rating >= ?` would drop the ~31% of rows with a NULL rating, so the clause is added only
// when a filter is actually asked for.
const stmt = db.prepare(
	`SELECT b.id id, b.z z, r.uploadtime t, r.rating rating
	 FROM bodies b JOIN replays r ON r.id = b.id
	 WHERE r.has_inputlog = 1 AND r.uploadtime >= ?
	 ${MIN_RATING ? 'AND r.rating >= ' + MIN_RATING : ''}
	 ORDER BY r.id LIMIT ? OFFSET ?`);

// deterministic 90/10 split by replay id, so a replay never spans both sides
const hashId = id => {
	let h = 2166136261;
	for (let i = 0; i < id.length; i++) { h ^= id.charCodeAt(i); h = Math.imul(h, 16777619); }
	return h >>> 0;
};
const isVal = id => (hashId(id) % 10) === 0;
// Shards must be disjoint or parallel loaders would feed the same replays repeatedly.
const inShard = id => NSHARDS === 1 || (Math.floor(hashId(id) / 10) % NSHARDS) === SHARD;

/** Walk one replay, yielding fixed-length sequences of decision points. */
function* samplesFrom(row) {
	const j = R.decodeBody(row.z);
	if (!j.inputlog) return;
	R.useSetsAt(row.t);

	const tr = { p1: new Tracker('p1'), p2: new Tracker('p2') };
	const pending = [];
	let battle = null, cursor = 0;
	const drain = () => {
		const slice = battle.log.slice(cursor); cursor = battle.log.length;
		if (!slice.length) return;
		const ch = R.extractChannelMessages(slice.join('\n'), [1, 2]);
		tr.p1.feed(ch[1]); tr.p2.feed(ch[2]);
	};

	let out = null, res = null;
	try {
		out = R.replayInputlog(j.inputlog, {
			anonymousNames: true,
			onDecision: (sid, req, choice, b) => {
				battle = b; drain();
				const a = fromChoice(choice, req);
				if (a < 0) return;
				const mask = legalMask(req);
				if (!mask[a]) return;
				const obs = encode(req, tr[sid]);
				pending.push({ ids: obs.ids, scalars: obs.scalars, mask, action: a, side: sid });
			},
		});
		if (out && out.battle && !out.truncated) res = R.resultOf(out.battle);
	} catch {
		return;   // divergence: drop rather than emit misaligned samples
	} finally { if (out && out.battle) try { out.battle.destroy(); } catch {} }

	if (res === null) return;
	for (const sid of ['p1', 'p2']) {
		const steps = pending.filter(x => x.side === sid);
		const outcome = sid === 'p1' ? res : -res;
		for (let i = 0; i < steps.length; i += SEQLEN) {
			const chunk = steps.slice(i, i + SEQLEN);
			// discount the terminal outcome over the whole episode, not the chunk
			const returns = chunk.map((_, k) => outcome * Math.pow(0.997, steps.length - (i + k) - 1));
			yield { steps: chunk, returns, outcome };
		}
	}
}

// ---- framing (same envelope as worker.js) ----
const MAGIC = 0x50535233; // "PSR3" - sequence frames
const PER_STEP = N_IDS * 4 + N_SCALARS * 4 + N_ACTIONS + 1 + 4 + 1;   // + action + return + valid
function writeFrame(buf) {
	const len = Buffer.allocUnsafe(4); len.writeUInt32LE(buf.length, 0);
	process.stdout.write(len); process.stdout.write(buf);
}
function frameOf(seqs) {
	const head = Buffer.allocUnsafe(28);
	head.writeUInt32LE(MAGIC, 0); head.writeUInt32LE(seqs.length, 4);
	head.writeUInt32LE(SEQLEN, 8); head.writeUInt32LE(N_IDS, 12);
	head.writeUInt32LE(N_SCALARS, 16); head.writeUInt32LE(N_ACTIONS, 20);
	head.writeUInt32LE(0, 24);
	const body = Buffer.alloc(seqs.length * SEQLEN * PER_STEP);   // zero-filled = invalid padding
	let o = 0;
	for (const q of seqs) {
		for (let t = 0; t < SEQLEN; t++) {
			const st = q.steps[t];
			if (st) {
				Buffer.from(st.ids.buffer, st.ids.byteOffset, N_IDS * 4).copy(body, o);
				Buffer.from(st.scalars.buffer, st.scalars.byteOffset, N_SCALARS * 4).copy(body, o + N_IDS * 4);
				const mo = o + N_IDS * 4 + N_SCALARS * 4;
				for (let k = 0; k < N_ACTIONS; k++) body[mo + k] = st.mask[k] ? 1 : 0;
				body[mo + N_ACTIONS] = st.action;
				body.writeFloatLE(q.returns[t], mo + N_ACTIONS + 1);
				body[mo + N_ACTIONS + 5] = 1;   // valid
			}
			o += PER_STEP;
		}
	}
	return Buffer.concat([head, body]);
}

let offset = 0, buf = [], replaysUsed = 0, replaysSeen = 0;
function fillBatch() {
	while (buf.length < BATCH) {
		const rows = stmt.all(MIN_TIME, 64, offset);
		if (!rows.length) {
			// The pool is exhausted. Anything still buffered is a real tail of up to BATCH-1
			// sequences; returning false here would drop it silently, once per shard.
			if (ONE_PASS) return buf.length > 0;
			offset = 0; if (!replaysUsed) return false; continue;
		}
		offset += rows.length;
		for (const r of rows) {
			if (isVal(r.id) !== (SPLIT === 'val')) continue;
			if (!inShard(r.id)) continue;
			replaysSeen++;
			let n = 0;
			for (const s of samplesFrom(r)) { buf.push(s); n++; }
			if (n) replaysUsed++;
		}
	}
	return true;
}

let inbuf = Buffer.alloc(0);
process.stdin.on('data', chunk => {
	inbuf = Buffer.concat([inbuf, chunk]);
	while (inbuf.length >= 4) {
		const len = inbuf.readUInt32LE(0);
		if (inbuf.length < 4 + len) return;
		inbuf = inbuf.subarray(4 + len);
		if (!fillBatch()) { writeFrame(frameOf([])); continue; }
		writeFrame(frameOf(buf.splice(0, BATCH)));
	}
});
process.stdin.on('end', () => process.exit(0));
process.on('exit', () => process.stderr.write(`replay_worker: ${replaysUsed}/${replaysSeen} replays yielded samples\n`));
process.stderr.write(`replay_worker ready: split=${SPLIT} minRating=${MIN_RATING} onePass=${ONE_PASS} shard=${SHARD}/${NSHARDS} batch=${BATCH} seqlen=${SEQLEN} perStep=${PER_STEP}B\n`);
