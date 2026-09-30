'use strict';
/**
 * Battle-state tracker built ONLY from one player's protocol channel.
 *
 * This is the boundary that keeps the actor honest: it never reads the Battle object,
 * so anything it knows, a real player on the ladder would also know. Own-side HP arrives
 * exact, opponent HP as a percentage (HP Percentage Mod), and opponent species/moves/items
 * only as they are revealed.
 */
const EMPTY_BOOSTS = () => ({ atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0 });
const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');

function newMon(name) {
	return {
		name, species: '', level: 100, gender: '',
		hp: 1, maxhp: 1, hpKnownExact: false,
		status: '', fainted: false,
		boosts: EMPTY_BOOSTS(), volatiles: {},
		item: '', itemKnown: false, ability: '', abilityKnown: false,
		moves: [],                       // revealed move ids, in order of first use
		teraType: '', terastallized: '',
		singleMove: [],                  // volatiles that end at this Pokemon's next move attempt
	};
}

function newSide(id) {
	return { id, mons: new Map(), activeName: null, conditions: {}, teamSize: 6, faintedCount: 0 };
}

/** One side's turn: the move its active used, HP fraction lost/healed, protect, switch. */
function newTurn() {
	return { mon: '', move: '', streak: 0, dmg: 0, heal: 0, protect: 0, switched: 0 };
}

class Tracker {
	/** @param {'p1'|'p2'} perspective */
	constructor(perspective = 'p1') {
		this.me = perspective;
		this.foe = perspective === 'p1' ? 'p2' : 'p1';
		this.sides = { p1: newSide('p1'), p2: newSide('p2') };
		this.weather = ''; this.weatherTurns = 0;
		this.terrain = ''; this.terrainTurns = 0;
		this.pseudo = {};
		this.turn = 0;
		// `cur` accumulates the turn in progress and `|turn|` moves it to `last`, so a decision
		// (a mid-turn forced switch included) sees the last completed turn.
		this.cur = { p1: newTurn(), p2: newTurn() };
		this.last = { p1: newTurn(), p2: newTurn() };
		this.matchupTurns = 0;                  // completed turns since either side switched
		this.singleTurn = [];                   // [mon, volatile] pairs that end with the turn
	}

	side(id) { return this.sides[id]; }

	/** `p1a: Great Tusk` -> {side, name} */
	_parseIdent(ident) {
		const m = /^(p[1-4])[a-c]?: (.*)$/.exec(ident || '');
		if (!m) return null;
		return { side: m[1], name: m[2] };
	}

	_mon(ident) {
		const p = this._parseIdent(ident);
		if (!p) return null;
		const s = this.sides[p.side];
		if (!s.mons.has(p.name)) s.mons.set(p.name, newMon(p.name));
		return s.mons.get(p.name);
	}

	_setCondition(mon, cond) {
		if (!mon || !cond) return;
		if (cond.endsWith(' fnt') || cond === '0 fnt') { mon.hp = 0; mon.fainted = true; return; }
		const m = /^(\d+)\/(\d+)/.exec(cond);
		if (m) {
			mon.hp = parseInt(m[1]); mon.maxhp = parseInt(m[2]);
			// maxhp 100 means the server is showing us percentages (opponent side)
			mon.hpKnownExact = mon.maxhp !== 100;
		}
		const st = /\b(brn|par|slp|frz|psn|tox)\b/.exec(cond);
		mon.status = st ? st[1] : '';
	}

	_switchIn(ident, details, cond) {
		const p = this._parseIdent(ident);
		if (!p) return;
		const s = this.sides[p.side];
		const prev = s.activeName && s.mons.get(s.activeName);
		if (prev) { prev.boosts = EMPTY_BOOSTS(); prev.volatiles = {}; }
		const mon = this._mon(ident);
		const parts = String(details || '').split(', ');
        mon.species = toID(parts[0]);
		for (const x of parts.slice(1)) {
			if (/^L\d+$/.test(x)) mon.level = parseInt(x.slice(1));
			else if (x === 'M' || x === 'F') mon.gender = x;
		}
		mon.boosts = EMPTY_BOOSTS(); mon.volatiles = {};
		this._setCondition(mon, cond);
		mon.fainted = false;
		s.activeName = mon.name;
	}

	_endSingleMove(mon) {
		for (const v of mon.singleMove) delete mon.volatiles[v];
		mon.singleMove = [];
	}

	feed(lines) { for (const l of lines) this.feedLine(l); }

	feedLine(line) {
		if (!line || line[0] !== '|') return;
		const p = line.split('|');
		const cmd = p[1];
		switch (cmd) {
		case 'turn':
			this.turn = parseInt(p[2]) || this.turn;
			this.last = this.cur;
			this.cur = { p1: newTurn(), p2: newTurn() };
			this.matchupTurns = this.last.p1.switched || this.last.p2.switched ? 0 : this.matchupTurns + 1;
			for (const [mon, v] of this.singleTurn) delete mon.volatiles[v];
			this.singleTurn = [];
			break;
		case 'teamsize': { const s = this.sides[p[2]]; if (s) s.teamSize = parseInt(p[3]) || 6; break; }
		case 'switch': case 'drag': case 'replace': {
			this._switchIn(p[2], p[3], p[4]);
			const q = this._parseIdent(p[2]);
			if (q && cmd !== 'replace') this.cur[q.side].switched = 1;   // replace = Illusion ending
			break;
		}
		case 'detailschange': case '-formechange': {
			const mon = this._mon(p[2]);
			if (mon) mon.species = toID(String(p[3] || '').split(', ')[0]);
			break;
		}
		case 'faint': { const mon = this._mon(p[2]); if (mon) { mon.fainted = true; mon.hp = 0; const q = this._parseIdent(p[2]); if (q) this.sides[q.side].faintedCount++; } break; }
		case '-damage': case '-heal': case '-sethp': {
			const mon = this._mon(p[2]), q = this._parseIdent(p[2]);
			const before = mon ? mon.hp / mon.maxhp : 0;
			this._setCondition(mon, p[3]);
			if (mon && q) {
				const d = mon.hp / mon.maxhp - before;
				if (d < 0) this.cur[q.side].dmg -= d; else this.cur[q.side].heal += d;
			}
			break;
		}
		case '-status': { const mon = this._mon(p[2]); if (mon) mon.status = p[3]; break; }
		case '-curestatus': { const mon = this._mon(p[2]); if (mon) mon.status = ''; break; }
		case '-cureteam': { const q = this._parseIdent(p[2]); if (q) for (const m of this.sides[q.side].mons.values()) m.status = ''; break; }
		case '-boost': case '-unboost': {
			const mon = this._mon(p[2]); if (!mon) break;
			const sign = cmd === '-boost' ? 1 : -1;
			const st = p[3];
			if (st in mon.boosts) mon.boosts[st] = Math.max(-6, Math.min(6, mon.boosts[st] + sign * (parseInt(p[4]) || 0)));
			break;
		}
		case '-setboost': { const mon = this._mon(p[2]); if (mon && p[3] in mon.boosts) mon.boosts[p[3]] = parseInt(p[4]) || 0; break; }
		case '-clearboost': case '-clearnegativeboost': {
			const mon = this._mon(p[2]); if (!mon) break;
			for (const k of Object.keys(mon.boosts)) {
				if (cmd === '-clearboost' || mon.boosts[k] < 0) mon.boosts[k] = 0;
			}
			break;
		}
		case '-clearallboost':
			for (const s of Object.values(this.sides)) for (const m of s.mons.values()) m.boosts = EMPTY_BOOSTS();
			break;
		case '-swapboost': break;   // rare; ignored
		case 'move': {
			const mon = this._mon(p[2]); if (!mon) break;
			this._endSingleMove(mon);
			const id = toID(p[3]);
			if (id && id !== 'struggle' && !mon.moves.includes(id)) mon.moves.push(id);
			// The turn record wants the move chosen, not one it called (Sleep Talk, Dancer),
			// but a locked-in move continuing (Outrage) is still that side's move.
			const from = p.slice(4).find(x => x.startsWith('[from]'));
			const q = this._parseIdent(p[2]);
			if (q && id && (!from || from === '[from]lockedmove')) {
				const c = this.cur[q.side], l = this.last[q.side];
				c.mon = mon.name; c.move = id;
				c.streak = l.move === id && l.mon === mon.name ? l.streak + 1 : 1;
			}
			break;
		}
		case '-start': {
			const mon = this._mon(p[2]); if (!mon) break;
			mon.volatiles[toID(String(p[3]).replace(/^move: /, ''))] = 1;
			break;
		}
		case '-end': {
			const mon = this._mon(p[2]); if (!mon) break;
			delete mon.volatiles[toID(String(p[3]).replace(/^move: /, ''))];
			break;
		}
		case '-item': { const mon = this._mon(p[2]); if (mon) { mon.item = toID(p[3]); mon.itemKnown = true; } break; }
		case '-enditem': { const mon = this._mon(p[2]); if (mon) { mon.item = ''; mon.itemKnown = true; } break; }
		case '-ability': { const mon = this._mon(p[2]); if (mon) { mon.ability = toID(p[3]); mon.abilityKnown = true; } break; }
		case '-terastallize': { const mon = this._mon(p[2]); if (mon) { mon.terastallized = toID(p[3]); mon.teraType = toID(p[3]); } break; }
		case '-weather': {
			const w = toID(p[2]);
			if (w === 'none') { this.weather = ''; this.weatherTurns = 0; }
			else if (line.includes('[upkeep]')) this.weatherTurns++;
			else { this.weather = w; this.weatherTurns = 0; }
			break;
		}
		case '-fieldstart': {
			const id = toID(String(p[2]).replace(/^move: /, ''));
			if (id.endsWith('terrain')) { this.terrain = id; this.terrainTurns = 0; } else this.pseudo[id] = 1;
			break;
		}
		case '-fieldend': {
			const id = toID(String(p[2]).replace(/^move: /, ''));
			if (id === this.terrain) { this.terrain = ''; this.terrainTurns = 0; } else delete this.pseudo[id];
			break;
		}
		case '-sidestart': {
			const q = /^(p[1-4])/.exec(p[2]); if (!q) break;
			const id = toID(String(p[3]).replace(/^move: /, ''));
			const c = this.sides[q[1]].conditions;
			c[id] = (c[id] || 0) + 1;   // spikes/toxic spikes stack
			break;
		}
		case '-sideend': {
			const q = /^(p[1-4])/.exec(p[2]); if (!q) break;
			delete this.sides[q[1]].conditions[toID(String(p[3]).replace(/^move: /, ''))];
			break;
		}
		case '-singleturn': {   // Protect, Endure, Roost...: over when the turn ends
			const mon = this._mon(p[2]), q = this._parseIdent(p[2]);
			const id = toID(String(p[3]).replace(/^move: /, ''));
			if (q && id === 'protect') this.cur[q.side].protect = 1;
			if (mon && id) { mon.volatiles[id] = 1; this.singleTurn.push([mon, id]); }
			break;
		}
		case '-singlemove': {   // Destiny Bond, Glaive Rush: until this Pokemon's next move attempt
			const mon = this._mon(p[2]), id = toID(String(p[3]).replace(/^move: /, ''));
			if (mon && id) { mon.volatiles[id] = 1; mon.singleMove.push(id); }
			break;
		}
		case '-mustrecharge': { const mon = this._mon(p[2]); if (mon) mon.volatiles.mustrecharge = 1; break; }
		case 'cant': {
			const mon = this._mon(p[2]); if (!mon) break;
			this._endSingleMove(mon);
			if (p[3] === 'recharge') delete mon.volatiles.mustrecharge;
			break;
		}
		case 'upkeep': if (this.terrain) this.terrainTurns++; break;
		default: break;
		}
	}

	activeMon(sideId) {
		const s = this.sides[sideId];
		return s.activeName ? s.mons.get(s.activeName) : null;
	}
	/** Opponent Pokemon revealed so far, in order of first appearance. */
	revealed(sideId) { return Array.from(this.sides[sideId].mons.values()); }
}

module.exports = { Tracker, EMPTY_BOOSTS, toID };
