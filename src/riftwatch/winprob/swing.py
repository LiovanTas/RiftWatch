"""Swing: how much each player moved their team's chance of winning, in percentage points.

The win-probability network (model.py) gives blue's chance of winning at any moment. Every
change in that chance over a game is handed to the players responsible:

  * draft -- with a network that knows the champions, the game doesn't start at 50%: the
    difference is the teams' drafts, shared by each team.
  * kills -- a kill moves the chance at once (one fewer enemy alive, the bounty). Half the
    killing team's gain goes to the killer and half to the assisters (all of it to the killer
    when nobody helped).
  * deaths -- the player who died carries the other team's gain, and half of every objective
    or structure their team loses while they're dead (the other half is shared by the team).
  * objectives -- dragons, Baron, Elder, heralds, voidgrubs, Atakhan: half the gain to the
    players who took part, half to the whole team; the other team shares the loss.
  * structures -- towers and inhibitors, the same way.
  * items -- with a network that reads items, finishing an item moves the chance (more armor
    against a team doing physical damage counts for more); the buyer gets it, and the other
    team's loss counts as falling behind in economy.
  * economy -- whatever moved between minutes that no event explains (farming, plates,
    trading, map pressure) goes to the team it favoured, half evenly and half by each
    player's gold and XP earned that minute; the other team shares the loss evenly. The last
    step, from the final minute to the result, is shared evenly too.

So a winning team's Swing adds up to +50 and a losing team's to -50.

Each moment also says *why* it moved the chance that much: the jump is split exactly among
what changed (a player down, the bounty gold, the kill score, the dragon, an item's armor...)
by Shapley values over those changes (each change's average effect over every order they
could be added in), and a death records who did the damage and what kind, where it happened,
the death timer, the gold the player was holding, and what the enemy took while they were
dead. For each finished item the network is also asked about the champion's other common
items at that same moment.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from math import factorial
from typing import Any

import psycopg

from riftwatch.winprob.state import ITEM_FLAG_NAMES, KEPT_EVENTS, Replay, compact, respawn_seconds, team_of

VERSION = "sw5"
PARTS = ("draft", "kills", "deaths", "objectives", "structures", "items", "economy")
PART_NAMES = {"draft": "Draft", "kills": "Kills", "deaths": "Deaths", "objectives": "Objectives",
              "structures": "Structures", "items": "Items", "economy": "Economy"}
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
DEAD_SHARE = 0.5                # share of a loss while players are dead that the dead carry
ALTERNATIVES = 4                # other items the network is asked about for each finished item
# Which change each input belongs to, for the "why" of a moment.
GROUP = {"alive_own": "alive", "alive_enemy": "alive", "gold_diff": "gold", "kills_own": "kills",
         "kills_enemy": "kills", "dragons_own": "dragons", "dragons_enemy": "dragons", "soul_own": "soul",
         "soul_enemy": "soul", "elder_own": "elder", "elder_enemy": "elder", "baron_own": "baron",
         "baron_enemy": "baron", "barons_own": "baron", "barons_enemy": "baron", "heralds_own": "herald",
         "heralds_enemy": "herald", "grubs_own": "grubs", "grubs_enemy": "grubs", "atakhan_own": "atakhan",
         "atakhan_enemy": "atakhan", "towers_own": "towers", "towers_enemy": "towers",
         "inhibs_down_own": "inhibitors", "inhibs_down_enemy": "inhibitors"}
GROUP_NAMES = {"alive": "a player down", "gold": "bounty gold", "kills": "the kill score", "dragons": "the dragon",
               "soul": "dragon soul", "elder": "the Elder buff", "baron": "the Baron buff", "herald": "the herald",
               "grubs": "the voidgrubs", "atakhan": "Atakhan", "towers": "the tower", "inhibitors": "the open inhibitor"}
ZONE_NAMES = {"own_jungle": "their own jungle", "enemy_jungle": "the enemy jungle", "river": "the river",
              "top_lane": "top lane", "mid_lane": "mid lane", "bot_lane": "bot lane", "own_base": "their base",
              "enemy_base": "the enemy base"}


def _clock(ms: float) -> str:
    s = int(ms // 1000)
    return f"{s // 60}:{s % 60:02d}"


def _team(side: int) -> list[int]:
    return list(range(1, 6)) if side == 100 else list(range(6, 11))


def _group(key: str) -> str | None:
    if key in GROUP:
        return GROUP[key]
    for suffix in ("_own", "_enemy"):
        flag = key[: -len(suffix)] if key.endswith(suffix) else None
        if flag in ITEM_FLAG_NAMES:
            return f"item:{flag}"
    return None


def group_name(group: str) -> str:
    """Plain words for a change, for the why of a moment. An item's flags are measured against
    an average finished item (the item itself is "a finished item")."""
    if group.startswith("item:"):
        flag = group[5:]
        return "a finished item" if flag == "items" else f"{ITEM_FLAG_NAMES.get(flag, flag)} (vs an average item)"
    return GROUP_NAMES.get(group, group)


def _changed(before: tuple[dict, dict], after: tuple[dict, dict]) -> list[str]:
    groups = []
    for k, v in before[0].items():
        if k.startswith("_") or v == after[0].get(k):
            continue
        g = _group(k)
        if g and g not in groups:
            groups.append(g)
    return groups[:6]


def _mix(before: tuple[dict, dict], after: tuple[dict, dict], groups: set[str]) -> tuple[dict, dict]:
    """The before state with the inputs of ``groups`` taken from the after state."""
    out = []
    for b, a in zip(before, after, strict=True):
        s = dict(b)
        for k in b:
            if not k.startswith("_") and _group(k) in groups:
                s[k] = a[k]
        out.append(s)
    return out[0], out[1]


def _shapley(groups: list[str], value: dict[frozenset, float]) -> dict[str, float]:
    k = len(groups)
    out = {}
    for g in groups:
        rest = [x for x in groups if x != g]
        total = 0.0
        for r in range(len(rest) + 1):
            w = factorial(r) * factorial(k - r - 1) / factorial(k)
            for S in combinations(rest, r):
                s = frozenset(S)
                total += w * (value[s | {g}] - value[s])
        out[g] = total
    return out


@dataclass(frozen=True)
class Rules:
    """How each change in the win chance is shared out. ``kill_split``: "damage" shares a
    kill half by damage dealt to the victim (Riot's record of it) and half equally among the
    players credited (crowd control, shields and heals aren't in the damage record); "half"
    gives half to the killer and half to the assisters. ``economy_split``: "lane" shares an
    unexplained gain among the team's players by how far each out-earned their lane opponent
    that minute (gold and XP) and a loss by how far each fell behind theirs; "earned" shares
    gains by gold and XP earned and losses evenly. ``even_economy`` of either is shared evenly
    first."""
    team_share: float = TEAM_SHARE
    even_economy: float = EVEN_ECONOMY
    dead_share: float = DEAD_SHARE
    kill_split: str = "damage"
    economy_split: str = "lane"


RULES = Rules()
FIXED_RULES = Rules(kill_split="half", economy_split="earned")   # the fixed shares before sw5


def evaluate(model, frames: list[dict], events: list[dict], names: dict[int, str] | None = None,
             champions: dict[int, int] | None = None, purchases: dict[int, list[tuple[int, int]]] | None = None,
             positions: dict[int, str] | None = None) -> dict[str, Any] | None:
    """Everything about one game that needs the model: the win chance each minute and at
    every event, why each event moved it (Shapley), and the facts sharing needs (who took
    part, damage to each victim, who was dead, each player's gold and XP each minute and who
    they were laning against). Sharing it out (``share``) is cheap and separate, so other
    rules can be tried on stored games without the model."""
    catalog = model.catalog if getattr(model, "reads_items", False) else None
    replay = Replay(frames, events, champions, purchases, catalog)
    if len(frames) < 2 or replay.winner not in (100, 200):
        return None
    names = names or {}
    end = next((e.get("timestamp", 0) for e in replay.events if e.get("type") == "GAME_END"), replay.end_ms)
    last = len(frames) - 1
    times = [min(m * 60_000, end) for m in range(len(frames))]

    pairs: list[tuple[dict, dict]] = [replay.pair(t) for t in times]

    def ask(state: tuple[dict, dict]) -> int:
        pairs.append(state)
        return len(pairs) - 1

    def subsets_of(before, after, groups):
        return {frozenset(S): ask(_mix(before, after, set(S)))
                for r in range(len(groups) + 1) for S in combinations(groups, r)}

    scored = []   # (kind, t, payload, before slot, after slot, {subset: slot}, groups)
    for i, e in enumerate(replay.events):
        if e.get("type") not in SCORED:
            continue
        t = e.get("timestamp", 0)
        bonus = {}
        if e["type"] == "CHAMPION_KILL":
            bonus = {300 - team_of(e.get("victimId", 1)):
                     float(e.get("bounty", 300)) + float(e.get("shutdownBounty", 0))}
        upto = i + 1
        while (upto < len(replay.events) and replay.events[upto].get("type") == "DRAGON_SOUL_GIVEN"
               and abs(replay.events[upto].get("timestamp", 0) - t) <= 2000):
            upto += 1                                       # the soul comes with the dragon
        before, after = replay.pair(t, i), replay.pair(t, upto, bonus)
        groups = _changed(before, after)
        subsets = subsets_of(before, after, groups)
        scored.append(("event", t, i, subsets[frozenset()], subsets[frozenset(groups)], subsets, groups))
    owned: dict[int, set[int]] = {}
    nth: Counter = Counter()
    for ms, pid, item in replay.items:                   # finished items
        before, after = replay.pair(ms, item_time=ms - 1), replay.pair(ms, item_time=ms)
        groups = _changed(before, after)
        subsets = subsets_of(before, after, groups)
        nth[pid] += 1
        alts = []
        for alt in catalog.popular.get((champions or {}).get(pid, 0), []):
            if alt != item and alt not in owned.get(pid, set()) and len(alts) < ALTERNATIVES:
                extra = {team_of(pid): Counter(catalog.items[alt])}
                alts.append((alt, ask(replay.pair(ms, item_time=ms - 1, item_extra=extra))))
        owned.setdefault(pid, set()).add(item)
        scored.append(("item", ms, (pid, item, nth[pid], alts), subsets[frozenset()],
                       subsets[frozenset(groups)], subsets, groups))
    p = [float(x) for x in model.blue([a for a, _ in pairs], [b for _, b in pairs])]
    curve = p[: len(frames)]

    moments, facts = [], []
    scored.sort(key=lambda s: s[1])
    for kind, t, payload, b, a, subsets, groups in scored:
        d = p[a] - p[b]                                       # change in blue's chance
        values = {s: p[slot] for s, slot in subsets.items()}
        why = {g: round(100 * v, 2) for g, v in _shapley(groups, values).items()} if groups else {}
        m: dict[str, Any] = {"t": t, "clock": _clock(t), "swing": round(100 * abs(d), 1),
                             "before": round(100 * p[b], 1), "after": round(100 * p[a], 1),
                             "why": why, "by": None, "lost": [], "dead": []}
        f: dict[str, Any] = {"t": t, "d": round(d, 6)}
        if kind == "item":
            pid, item, slot, alts = payload
            side = team_of(pid)
            gain = d if side == 100 else -d
            alt_rows = [{"item": alt, "name": catalog.item_names.get(alt, str(alt)),
                         "swing": round(100 * ((p[s_] - p[b]) if side == 100 else (p[b] - p[s_])), 2)}
                        for alt, s_ in alts]
            m.update(part="items", team=side, players=[pid], by=pid, item=item, slot=slot,
                     text=f"{names.get(pid, 'A champion')} finished {catalog.item_names.get(item, 'an item')}",
                     gain=round(100 * gain, 2), alternatives=sorted(alt_rows, key=lambda r: -r["swing"]))
            f.update(kind="items", side=side, by=pid)
        else:
            e = replay.events[payload]
            if e["type"] == "CHAMPION_KILL":
                victim, killer = e.get("victimId", 0), e.get("killerId", 0)
                side = 300 - team_of(victim)
                helpers = [x for x in (e.get("assistingParticipantIds") or []) if team_of(x) == side]
                credited = 1 <= killer <= 10 and team_of(killer) == side
                got = [killer, *helpers] if credited else helpers
                text = (f"{names.get(killer, 'A champion')} killed {names.get(victim, 'a champion')}" if credited
                        else f"{names.get(victim, 'A champion')} was executed")
                damage: Counter = Counter()
                for dmg in e.get("victimDamageReceived") or []:
                    source = dmg.get("participantId") or 0
                    if source in got:
                        damage[source] += ((dmg.get("physicalDamage") or 0) + (dmg.get("magicDamage") or 0)
                                           + (dmg.get("trueDamage") or 0))
                m.update(part="kills", team=side, players=sorted(set(got)), lost=[victim],
                         by=killer if 1 <= killer <= 10 else None, text=text, death=_death(replay, e, payload))
                f.update(kind="kills", side=side, killer=killer if credited else 0, helpers=helpers,
                         victim=victim, damage={str(k): v for k, v in damage.items()})
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
                got = [x for x in [e.get("killerId", 0), *(e.get("assistingParticipantIds") or [])]
                       if 1 <= x <= 10 and team_of(x) == side]
                dead = sorted(v for v in replay.dead_at(t, payload) if team_of(v) == 300 - side)
                m.update(part=part, team=side, players=sorted(set(got)), dead=dead, text=text,
                         by=e.get("killerId") if 1 <= (e.get("killerId") or 0) <= 10 else None)
                f.update(kind=part, side=side, got=sorted(set(got)), dead=dead)
        moments.append(m)
        facts.append(f)

    # What the enemy took while each dead player was down.
    for m in moments:
        death = m.get("death")
        if not death:
            continue
        victim = m["lost"][0]
        back = m["t"] + 1000 * death["timer"]
        follow = [j for j, x in enumerate(moments) if m["t"] < x["t"] <= back
                  and x["part"] in ("objectives", "structures") and x["team"] != team_of(victim)]
        death["while_dead"] = follow
        death["while_dead_cost"] = round(sum(moments[j]["swing"] for j in follow), 1)

    # Gold and XP each player earned each minute, and their lane opponent.
    earned = {}
    for pid in range(1, 11):
        row = []
        for m in range(last):
            a, b = frames[m].get(str(pid)) or {}, frames[m + 1].get(str(pid)) or {}
            row.append(round(max(0.0, b.get("totalGold", 0) - a.get("totalGold", 0) + b.get("xp", 0) - a.get("xp", 0))))
        earned[str(pid)] = row
    opponent = {}
    for pid, pos in (positions or {}).items():
        opp = next((q for q, qpos in (positions or {}).items() if qpos == pos and team_of(q) != team_of(pid)), None)
        if opp:
            opponent[str(pid)] = opp
    series = [(t, 100 * v) for t, v in zip(times, curve, strict=True)]
    for _, t, _, b, a, _, _ in scored:
        series += [(t, 100 * p[b]), (t + 1, 100 * p[a])]
    series.sort()
    return {"version": VERSION, "winner": replay.winner, "end_s": round(end / 1000),
            "start": round(100 * curve[0], 1), "reads_items": catalog is not None,
            "curve": [round(100 * x, 1) for x in curve],
            "series": [[round(t / 1000, 1), round(v, 1)] for t, v in series],
            "moments": moments,
            "facts": {"winner": replay.winner, "times": times, "curve": [round(x, 6) for x in curve],
                      "events": facts, "earned": earned, "opponent": opponent}}


def _pts(p: float) -> float:
    """A probability as percentage points to one decimal. Adding 0.0 turns -0.0 into 0.0:
    Postgres stores both as 0, so a fresh result must match a stored one exactly (coaching
    is cached by its evidence text, where "-0.0" and "+0.0" differ)."""
    return round(100 * p, 1) + 0.0


def share(facts: dict[str, Any], rules: Rules = RULES) -> dict[str, dict[str, Any]]:
    """Share every change in the win chance among the ten players (see the module docstring):
    per player, each part, the total, the biggest plays and costliest moments (by index into
    the game's moments) and the running total each minute, in percentage points."""
    credit, plays, gains = credit_of(facts, rules)
    players = {}                                           # keyed by str(pid), as JSON stores it
    for pid, c in credit.items():
        running, line = 0.0, []
        for g in gains[pid]:
            running += g
            line.append(_pts(running))
        players[str(pid)] = {**{k: _pts(v) for k, v in c.items()},
                             "total": _pts(sum(c.values())),
                             "best": [{"moment": i, "swing": _pts(a)}
                                      for a, i in sorted(plays[pid], key=lambda x: -x[0])[:PLAYS] if a > 0.0005],
                             "worst": [{"moment": i, "swing": _pts(a)}
                                       for a, i in sorted(plays[pid], key=lambda x: x[0])[:PLAYS] if a < -0.0005],
                             "line": line}
    return players


def credit_of(facts: dict[str, Any], rules: Rules = RULES):
    """The exact shares, as probabilities (not rounded): per player each part, their plays
    (share, moment index) and what they gained each minute."""
    curve, times, events = facts["curve"], facts["times"], facts["events"]
    last = len(curve) - 1
    credit = {pid: dict.fromkeys(PARTS, 0.0) for pid in range(1, 11)}
    plays: dict[int, list[tuple[float, int]]] = {pid: [] for pid in range(1, 11)}
    gains = [[0.0] * (last + 1) for _ in range(11)]

    def give(pids, amount: float, part: str, t: float, moment: int | None = None, weights=None) -> None:
        pids = [x for x in pids if 1 <= x <= 10]
        if not pids:
            return
        if weights is None:
            weights = {x: 1 / len(pids) for x in pids}
        m = min(int(t // 60_000), last)
        for x in pids:
            share_ = amount * weights[x]
            credit[x][part] += share_
            gains[x][m] += share_
            if moment is not None:
                plays[x].append((share_, moment))

    draft = curve[0] - 0.5
    give(_team(100), draft, "draft", 0)
    give(_team(200), -draft, "draft", 0)
    explained = [0.0] * len(curve)
    for n, f in enumerate(events):
        d, t, side = f["d"], f["t"], f["side"]
        explained[min(int(t // 60_000), max(0, last - 1))] += d
        gain = d if side == 100 else -d
        kind = f["kind"]
        if kind == "items":
            give([f["by"]], gain, "items", t, n)
            give(_team(300 - side), -gain, "economy", t)
        elif kind == "kills":
            killer, helpers = f["killer"], f["helpers"]
            if killer:
                got = [killer, *helpers]
                if rules.kill_split == "half" or len(got) == 1:
                    weights = {killer: 0.5 if helpers else 1.0, **{h: 0.5 / len(helpers) for h in helpers}}
                else:
                    damage = {x: float(f["damage"].get(str(x), 0)) for x in got}
                    total = sum(damage.values())
                    weights = {x: 0.5 * (damage[x] / total if total else 1 / len(got)) + 0.5 / len(got) for x in got}
                give(got, gain, "kills", t, n, weights)
            else:                                          # executed by a tower or minions
                give(helpers or _team(side), gain, "kills", t, n)
            give([f["victim"]], -gain, "deaths", t, n)
        else:
            got, dead = f["got"], f["dead"]
            if got:
                give(got, gain * (1 - rules.team_share), kind, t, n)
                give(_team(side), gain * rules.team_share, kind, t)       # not anyone's own play
            else:
                give(_team(side), gain, kind, t)
            if dead:                                       # lost while players were dead
                give(dead, -gain * rules.dead_share, "deaths", t, n)
                give(_team(300 - side), -gain * (1 - rules.dead_share), kind, t)
            else:
                give(_team(300 - side), -gain, kind, t)

    # Economy: what moved between minutes that the events don't explain.
    earned, opponent = facts["earned"], facts.get("opponent") or {}
    for m in range(last):
        rest = (curve[m + 1] - curve[m]) - explained[m]
        for side, amount in ((100, rest), (200, -rest)):
            team = _team(side)
            got = {pid: float(earned[str(pid)][m]) for pid in team}
            if rules.economy_split == "lane":
                gap = {pid: got[pid] - float(earned[str(opponent[str(pid)])][m]) if str(pid) in opponent else 0.0
                       for pid in team}
                # A gain goes to whoever out-earned their opponent, a loss to whoever fell behind.
                stake = {pid: max(0.0, gap[pid] if amount > 0 else -gap[pid]) for pid in team}
            else:
                stake = got if amount > 0 else dict.fromkeys(team, 0.0)
            total = sum(stake.values())
            weights = {pid: rules.even_economy / 5 + (1 - rules.even_economy) * stake[pid] / total if total > 0
                       else 1 / 5 for pid in team}
            give(team, amount, "economy", (m + 1) * 60_000, weights=weights)
    finish = (1.0 if facts["winner"] == 100 else 0.0) - curve[-1]
    give(_team(100), finish, "economy", times[-1])
    give(_team(200), -finish, "economy", times[-1])
    return credit, plays, gains


def game_swing(model, frames: list[dict], events: list[dict], names: dict[int, str] | None = None,
               champions: dict[int, int] | None = None,
               purchases: dict[int, list[tuple[int, int]]] | None = None,
               positions: dict[int, str] | None = None, rules: Rules = RULES) -> dict[str, Any] | None:
    """Swing for all ten players of one game, from its per-minute participant frames and
    events (``compact`` of a timeline), each player's champion, purchases and position."""
    data = evaluate(model, frames, events, names, champions, purchases, positions)
    if data is None:
        return None
    data["players"] = share(data["facts"], rules)
    data["rules"] = asdict(rules)
    return data


def _death(replay: Replay, e: dict, index: int) -> dict:
    """Who did the damage and what kind, where, the death timer and the gold held."""
    from riftwatch.ml.situations import own_zone

    victim, t = e.get("victimId", 0), e.get("timestamp", 0)
    by_source: Counter = Counter()
    kinds: Counter = Counter()
    for d in e.get("victimDamageReceived") or []:
        amount = (d.get("physicalDamage") or 0) + (d.get("magicDamage") or 0) + (d.get("trueDamage") or 0)
        by_source[d.get("participantId") or 0] += amount
        kinds["physical"] += d.get("physicalDamage") or 0
        kinds["magic"] += d.get("magicDamage") or 0
        kinds["true"] += d.get("trueDamage") or 0
    total = sum(by_source.values())
    enemies = {x for x in [e.get("killerId", 0), *(e.get("assistingParticipantIds") or []), *by_source]
               if 1 <= (x or 0) <= 10 and team_of(x) != team_of(victim)}
    pos = e.get("position") or {}
    zone = own_zone(pos.get("x", 7400), pos.get("y", 7400), team_of(victim)) if pos else ""
    level = int(replay.frame_value(victim, "level", t)) or 1
    allies_dead = len([v for v in replay.dead_at(t, index) if team_of(v) == team_of(victim)])
    return {"damage": round(total), "sources": [{"pid": pid, "share": round(100 * v / total)}
                                                 for pid, v in by_source.most_common(4) if pid and total],
            "kinds": {k: round(100 * v / total) for k, v in kinds.items()} if total else {},
            "enemies": len(enemies), "zone": ZONE_NAMES.get(zone, zone.replace("_", " ")),
            "timer": round(respawn_seconds(level, t / 60_000)), "level": level,
            "held": round(replay.frame_value(victim, "currentGold", t)), "allies_dead": allies_dead}


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
                                             structures, economy, draft, kills, deaths, items)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(match_id, int(pid), ver, v["total"], v["kills"] + v["deaths"], v["objectives"], v["structures"],
                  v["economy"], v["draft"], v["kills"], v["deaths"], v["items"])
                 for pid, v in data["players"].items()])


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


def for_matches(conn: psycopg.Connection, model, match_ids: list[str], champion_name=None,
                compute_missing: bool = True) -> dict[str, dict]:
    """Swing for each game: from the table, computing (and storing) any game with a timeline
    that isn't there yet. Only the parts of each timeline Swing reads are fetched."""
    from riftwatch.winprob.model import GAME_SQL, players, positions

    if model is None or not match_ids:
        return {}
    found = stored(conn, match_ids, model)
    missing = [m for m in match_ids if m not in found]
    if missing and compute_missing:
        names = _names(conn, missing, champion_name)
        for match_id, frames, events, rows in conn.execute(GAME_SQL + " WHERE t.match_id = ANY(%s)",
                                                           (missing,)).fetchall():
            champions, purchases = players(rows)
            try:
                data = game_swing(model, frames, events, names.get(match_id), champions, purchases, positions(rows))
            except Exception:                              # a malformed timeline shouldn't break a page
                data = None
            if data is not None:
                save(conn, match_id, data, model)
                found[match_id] = data
    return found


def for_timeline(conn: psycopg.Connection, model, match_id: str, timeline: dict | None,
                 rows: list[dict], champion_name) -> dict | None:
    """Swing for one game whose full timeline is already loaded (the match page); ``rows`` are
    its participant_builds rows."""
    if model is None or not timeline:
        return None
    found = stored(conn, [match_id], model).get(match_id)
    if found is None:
        frames, events = compact(timeline)
        champions = {r["participant_id"]: r["champion_id"] for r in rows}
        purchases = {r["participant_id"]: list(zip(r.get("purchase_s") or [], r.get("purchases") or [], strict=False))
                     for r in rows}
        names = {pid: champion_name(c) for pid, c in champions.items()}
        places = {r["participant_id"]: r["position"] for r in rows if r.get("position")}
        try:
            found = game_swing(model, frames, [e for e in events if e.get("type") in KEPT_EVENTS], names,
                               champions, purchases, places)
        except Exception:
            found = None
        if found is not None:
            save(conn, match_id, found, model)
    return found


# -- what's typical, for a moment's leverage ------------------------------------------------

NORMS_FILE = "norms.json"
_norms: dict[str, Any] = {}


def build_norms(conn: psycopg.Connection, model, models_dir) -> dict:
    """The typical swing of each kind of moment in each ten minutes of a game, over every
    stored game: a moment's leverage is its swing over that."""
    from pathlib import Path

    rows = conn.execute("""
        SELECT m->>'part', least((m->>'t')::bigint / 600000, 3), avg((m->>'swing')::float), count(*)
          FROM match_swing s, jsonb_array_elements(s.data->'moments') m
         WHERE s.version = %s GROUP BY 1, 2""", (version(model),)).fetchall()
    out: dict[str, dict[str, float]] = {}
    for part, bucket, avg, n in rows:
        if n >= 50:
            out.setdefault(part, {})[str(bucket)] = round(avg, 2)
    # Costly habits in deaths, per role, from a sample of stored games.
    from riftwatch.winprob import habits

    sample = conn.execute("""
        SELECT b.position, s.data, b.participant_id
          FROM (SELECT match_id, data FROM match_swing WHERE version = %s ORDER BY md5(match_id) LIMIT 3000) s
          JOIN participant_builds b USING (match_id)
         WHERE NOT b.remake AND b.queue_id IN (420, 440)""", (version(model),)).fetchall()
    data = {"version": version(model), "norms": out, "habits": habits.role_norms(sample)}
    path = Path(models_dir) / "winprob" / NORMS_FILE
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _norms.clear()
    return data


def norms(models_dir) -> dict[str, Any]:
    """part -> ten-minute bucket ("0".."3") -> typical swing, and "habits" -> role -> costly
    habits per game (cached per process)."""
    from pathlib import Path

    path = Path(models_dir) / "winprob" / NORMS_FILE
    key = str(path)
    if key not in _norms:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        _norms[key] = {**data.get("norms", {}), "habits": data.get("habits", {})}
    return _norms[key]


def leverage(moment: dict, table: dict[str, dict[str, float]]) -> float | None:
    """How many times a typical moment of its kind, at that stage of the game, this was."""
    typical = (table.get(moment.get("part", "")) or {}).get(str(min(int(moment["t"] // 600_000), 3)))
    return round(moment["swing"] / typical, 1) if typical else None
