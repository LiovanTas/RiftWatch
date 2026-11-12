"""Swing: how much each player moved their team's chance of winning, in percentage points.

The win-probability network (model.py) gives blue's chance of winning at any moment. Every
change in that chance over a game is handed to the players responsible:

  * fights -- a kill moves the chance at once (one fewer enemy alive, the bounty). The killing
    team's gain goes half to the killer and half split among assisters (all of it to the
    killer when nobody assisted); the player who died carries the other team's loss.
  * objectives -- dragons, Baron, Elder, heralds, voidgrubs, Atakhan: half the taking team's
    gain is split among the players who took part and half across the team; the other team's
    loss is shared by all five.
  * structures -- towers and inhibitors, the same way.
  * economy -- whatever moved between minutes that no event explains (farming, plates,
    trading, map pressure) goes to the team it favoured, half evenly and half by each player's
    gold and XP earned that minute; the other team shares the loss evenly. The last step, from
    the final minute to the result, is shared evenly too.

Every game starts at 50% (the network sees an even state and answers 50), so a winning
team's Swing adds up to +50 and a losing team's to -50; a player's Swing is their part of it.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg

from riftwatch.winprob.state import KEPT_EVENTS, Replay, compact, team_of

VERSION = "sw3"
PARTS = ("fights", "objectives", "structures", "economy")
SCORED = ("CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL")
MONSTER = {"DRAGON": "Dragon", "ELDER_DRAGON": "Elder Dragon", "BARON_NASHOR": "Baron",
           "RIFTHERALD": "Rift Herald", "HORDE": "a voidgrub", "ATAKHAN": "Atakhan"}
DRAGON = {"FIRE_DRAGON": "Infernal", "WATER_DRAGON": "Ocean", "EARTH_DRAGON": "Mountain",
          "AIR_DRAGON": "Cloud", "HEXTECH_DRAGON": "Hextech", "CHEMTECH_DRAGON": "Chemtech"}
LANE = {"TOP_LANE": "top", "MID_LANE": "mid", "BOT_LANE": "bot"}
PLAYS = 3                       # biggest plays and costliest moments kept per player
# Share of an objective's or structure's gain that goes to the whole team rather than only the
# players who took part (the jungler takes nearly every objective; the team made it possible),
# and share of the economy gain split evenly rather than by gold and XP earned (supports earn
# little but still move the game). Both chosen so no role is favoured by the rules themselves.
TEAM_SHARE = 0.5
EVEN_ECONOMY = 0.5


def _clock(ms: float) -> str:
    s = int(ms // 1000)
    return f"{s // 60}:{s % 60:02d}"


def _team(side: int) -> list[int]:
    return list(range(1, 6)) if side == 100 else list(range(6, 11))


def game_swing(model, frames: list[dict], events: list[dict],
               names: dict[int, str] | None = None) -> dict[str, Any] | None:
    """Swing for all ten players of one game, from its per-minute participant frames and
    events (``compact`` of a timeline). ``names`` maps participant id to champion name."""
    replay = Replay(frames, events)
    if len(frames) < 2 or replay.winner not in (100, 200):
        return None
    names = names or {}
    end = next((e.get("timestamp", 0) for e in replay.events if e.get("type") == "GAME_END"), replay.end_ms)
    last = len(frames) - 1
    times = [min(m * 60_000, end) for m in range(len(frames))]

    # Every state the model is asked about, then one batch prediction: each minute, and just
    # before and after each kill, objective and structure.
    pairs = [replay.pair(t) for t in times]
    scored = []
    for i, e in enumerate(replay.events):
        if e.get("type") not in SCORED:
            continue
        t = e.get("timestamp", 0)
        bonus = {}
        if e["type"] == "CHAMPION_KILL":
            bonus = {300 - team_of(e.get("victimId", 1)):
                     float(e.get("bounty", 300)) + float(e.get("shutdownBounty", 0))}
        scored.append((i, len(pairs)))
        pairs.append(replay.pair(t, i))
        pairs.append(replay.pair(t, i + 1, bonus))
    p = [float(x) for x in model.blue([a for a, _ in pairs], [b for _, b in pairs])]
    curve = p[: len(frames)]

    credit = {pid: dict.fromkeys(PARTS, 0.0) for pid in range(1, 11)}
    plays: dict[int, list[tuple[float, int]]] = {pid: [] for pid in range(1, 11)}

    def give(pids, amount: float, part: str, moment: int | None = None) -> None:
        pids = [x for x in pids if 1 <= x <= 10]
        for x in pids:
            credit[x][part] += amount / len(pids)
            if moment is not None:
                plays[x].append((amount / len(pids), moment))

    moments, explained = [], [0.0] * len(frames)
    for i, before in scored:
        e = replay.events[i]
        d = p[before + 1] - p[before]                      # change in blue's chance
        t = e.get("timestamp", 0)
        explained[min(int(t // 60_000), last - 1)] += d
        n = len(moments)
        if e["type"] == "CHAMPION_KILL":
            victim, killer = e.get("victimId", 0), e.get("killerId", 0)
            side, part = 300 - team_of(victim), "fights"
            gain = d if side == 100 else -d
            helpers = [a for a in (e.get("assistingParticipantIds") or []) if team_of(a) == side]
            if 1 <= killer <= 10 and team_of(killer) == side:
                give([killer], gain * (0.5 if helpers else 1.0), part, n)
                if helpers:
                    give(helpers, gain * 0.5, part, n)
                got = [killer, *helpers]
                text = f"{names.get(killer, 'A champion')} killed {names.get(victim, 'a champion')}"
            else:                                          # executed by a tower or minions
                give(helpers or _team(side), gain, part, n)
                got = helpers
                text = f"{names.get(victim, 'A champion')} was executed"
            give([victim], -gain, part, n)
            lost, by = [victim], killer if 1 <= killer <= 10 else None
        else:
            if e["type"] == "ELITE_MONSTER_KILL":
                side, part = e.get("killerTeamId") or 100, "objectives"
                name = MONSTER.get(e.get("monsterType"), "an objective")
                if e.get("monsterType") == "DRAGON" and e.get("monsterSubType") in DRAGON:
                    name = f"{DRAGON[e['monsterSubType']]} Dragon"
                text = f"{'Blue' if side == 100 else 'Red'} took {name}"
            else:
                side, part = 300 - (e.get("teamId") or 100), "structures"
                what = "inhibitor" if e.get("buildingType") == "INHIBITOR_BUILDING" else "tower"
                lane = LANE.get(e.get("laneType"))
                text = f"{'Blue' if side == 100 else 'Red'} destroyed the {lane + ' ' if lane else ''}{what}"
            gain = d if side == 100 else -d
            got = [x for x in [e.get("killerId", 0), *(e.get("assistingParticipantIds") or [])]
                   if 1 <= x <= 10 and team_of(x) == side]
            if got:
                give(got, gain * (1 - TEAM_SHARE), part, n)
                give(_team(side), gain * TEAM_SHARE, part)     # not anyone's own play
            else:
                give(_team(side), gain, part, n)
            give(_team(300 - side), -gain, part, n)
            lost, by = [], None
        moments.append({"t": t, "clock": _clock(t), "text": text, "part": part, "team": side,
                        "swing": round(100 * abs(d), 1), "players": sorted(set(got)), "lost": lost, "by": by,
                        "before": round(100 * p[before], 1), "after": round(100 * p[before + 1], 1)})

    # Economy: what moved between minutes that the events don't explain.
    for m in range(last):
        rest = (curve[m + 1] - curve[m]) - explained[m]
        for side, amount in ((100, rest), (200, -rest)):
            earned = {}
            for pid in _team(side):
                a, b = frames[m].get(str(pid)) or {}, frames[m + 1].get(str(pid)) or {}
                earned[pid] = max(0.0, b.get("totalGold", 0) - a.get("totalGold", 0) + b.get("xp", 0) - a.get("xp", 0))
            total = sum(earned.values())
            if amount > 0 and total > 0:
                for pid, v in earned.items():
                    credit[pid]["economy"] += amount * (EVEN_ECONOMY / 5 + (1 - EVEN_ECONOMY) * v / total)
            else:
                give(_team(side), amount, "economy")
    finish = (1.0 if replay.winner == 100 else 0.0) - curve[-1]
    give(_team(100), finish, "economy")
    give(_team(200), -finish, "economy")

    players = {}                                           # keyed by str(pid), as JSON stores it
    for pid, c in credit.items():
        mine = sorted(plays[pid], key=lambda x: -abs(x[0]))
        players[str(pid)] = {**{k: round(100 * v, 1) for k, v in c.items()},
                        "total": round(100 * sum(c.values()), 1),
                        "best": [{"moment": m, "swing": round(100 * a, 1)}
                                 for a, m in sorted(plays[pid], key=lambda x: -x[0])[:PLAYS] if a > 0.0005],
                        "worst": [{"moment": m, "swing": round(100 * a, 1)}
                                  for a, m in sorted(plays[pid], key=lambda x: x[0])[:PLAYS] if a < -0.0005],
                        "events": len(mine)}
    # The chart: each minute, and the jump at each event.
    series = [(t, 100 * v) for t, v in zip(times, curve, strict=True)]
    for i, before in scored:
        t = replay.events[i].get("timestamp", 0)
        series += [(t, 100 * p[before]), (t + 1, 100 * p[before + 1])]
    series.sort()
    return {"version": VERSION, "winner": replay.winner, "end_s": round(end / 1000),
            "curve": [round(100 * x, 1) for x in curve],
            "series": [[round(t / 1000, 1), round(v, 1)] for t, v in series],
            "players": players, "moments": moments}


def top_moments(data: dict, n: int = 8) -> list[dict]:
    """The events that moved the chance most, biggest first."""
    return sorted(data.get("moments", []), key=lambda m: -m["swing"])[:n]


# -- stored per game -----------------------------------------------------------------------

def version(model) -> str:
    """This calculation and the model it ran on: stored Swing from an older model is redone."""
    return f"{VERSION}-{getattr(model, 'stamp', '0')}"


def stored(conn: psycopg.Connection, match_ids: list[str], model) -> dict[str, dict]:
    rows = conn.execute("SELECT match_id, data FROM match_swing WHERE match_id = ANY(%s) AND version = %s",
                        (list(match_ids), version(model))).fetchall()
    return {m: d for m, d in rows}


def save(conn: psycopg.Connection, match_id: str, data: dict, model) -> None:
    """Store a game's Swing, and each player's on its own row (player_swing)."""
    ver = version(model)
    with conn.transaction():
        conn.execute("""INSERT INTO match_swing (match_id, version, data) VALUES (%s, %s, %s)
                        ON CONFLICT (match_id) DO UPDATE SET version = EXCLUDED.version, data = EXCLUDED.data,
                                                            created_at = now()""",
                     (match_id, ver, json.dumps(data)))
        conn.execute("DELETE FROM player_swing WHERE match_id = %s", (match_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO player_swing (match_id, participant_id, version, total, fights, objectives,
                                             structures, economy) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [(match_id, int(pid), ver, v["total"], *(v[k] for k in PARTS)) for pid, v in data["players"].items()])


def champion_swing(conn: psycopg.Connection, model, champion_id: int, position: str | None,
                   min_games: int = 15) -> dict | None:
    """A champion's average Swing (and in wins and in losses) over stored games."""
    if model is None:
        return None
    role_sql, params = ("AND b.position = %s", [position]) if position else ("", [])
    row = conn.execute(f"""
        SELECT count(*), avg(s.total), avg(s.total) FILTER (WHERE b.win), avg(s.total) FILTER (WHERE NOT b.win)
          FROM participant_builds b JOIN player_swing s USING (match_id, participant_id)
         WHERE s.version = %s AND b.champion_id = %s AND NOT b.remake {role_sql}""",
                       (version(model), champion_id, *params)).fetchone()
    if not row or row[0] < min_games:
        return None
    return {"games": row[0], "avg": round(row[1], 1), "wins": round(row[2] or 0, 1),
            "losses": round(row[3] or 0, 1)}


def player_best(conn: psycopg.Connection, model, puuid: str) -> tuple[str, float] | None:
    """The player's stored game with the most Swing: (match id, Swing)."""
    if model is None:
        return None
    return conn.execute("""
        SELECT b.match_id, s.total
          FROM participant_builds b JOIN player_swing s USING (match_id, participant_id)
         WHERE b.puuid = %s AND s.version = %s AND NOT b.remake AND b.duration_s >= 900
         ORDER BY s.total DESC LIMIT 1""", (puuid, version(model))).fetchone()


def _names(conn, match_ids: list[str], champion_name) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for match_id, pid, champ in conn.execute(
            "SELECT match_id, participant_id, champion_id FROM participant_builds WHERE match_id = ANY(%s)",
            (list(match_ids),)):
        out.setdefault(match_id, {})[pid] = champion_name(champ) if champion_name else str(champ)
    return out


COMPACT_SQL = """
    SELECT match_id,
           jsonb_path_query_array(raw, '$.info.frames[*].participantFrames'),
           jsonb_path_query_array(raw, '$.info.frames[*].events[*] ? (@.type == "CHAMPION_KILL"
               || @.type == "ELITE_MONSTER_KILL" || @.type == "BUILDING_KILL"
               || @.type == "DRAGON_SOUL_GIVEN" || @.type == "GAME_END")')
      FROM match_timelines WHERE match_id = ANY(%s)"""


def for_matches(conn: psycopg.Connection, model, match_ids: list[str], champion_name=None,
                compute_missing: bool = True) -> dict[str, dict]:
    """Swing for each game: from the table, computing (and storing) any game with a timeline
    that isn't there yet. Only the parts of each timeline Swing reads are fetched."""
    if model is None or not match_ids:
        return {}
    found = stored(conn, match_ids, model)
    missing = [m for m in match_ids if m not in found]
    if missing and compute_missing:
        names = _names(conn, missing, champion_name)
        for match_id, frames, events in conn.execute(COMPACT_SQL, (missing,)).fetchall():
            try:
                data = game_swing(model, frames, events, names.get(match_id))
            except Exception:                              # a malformed timeline shouldn't break a page
                data = None
            if data is not None:
                save(conn, match_id, data, model)
                found[match_id] = data
    return found


def for_timeline(conn: psycopg.Connection, model, match_id: str, timeline: dict | None,
                 names: dict[int, str]) -> dict | None:
    """Swing for one game whose full timeline is already loaded (the match page)."""
    if model is None or not timeline:
        return None
    found = stored(conn, [match_id], model).get(match_id)
    if found is None:
        frames, events = compact(timeline)
        try:
            found = game_swing(model, frames, [e for e in events if e.get("type") in KEPT_EVENTS], names)
        except Exception:
            found = None
        if found is not None:
            save(conn, match_id, found, model)
    return found
