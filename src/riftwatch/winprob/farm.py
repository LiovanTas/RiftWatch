"""Farm given up for plays: the gold and XP a laner or jungler would have earned by staying on
their farm, against what they took while they were away making a play (a roam, a gank, a
fight, an objective, a tower) or dead.

Everything is measured, from the timeline's minute frames (CS, gold, XP and position):

  * a **farming minute** is one where the player was alive throughout, in their own area at
    both ends of it and took no part in a kill, death, objective or structure. Until 14:00
    (when turret plates fall) the area is their lane, or for a jungler the jungle and river;
    after it, anywhere but a base, as side waves and camps are wherever they are. Minutes in
    their own base (a back) are neither farming nor away, and the first minute is skipped
    (minions haven't arrived);
  * their **pace** is the CS they took over their farming minutes in this game against what's
    typical for their role at those minutes of a game (``farm.json``, from stored games),
    pulled towards typical as if typical were three more farming minutes: a player farming
    at 90% of typical is expected to keep farming at about 90%;
  * the **window** of a play is the run of minutes the player was away around it (at most two
    minutes either side of the plays in that run), plus up to two farming minutes after, so
    farm they caught up on (camps that waited, a wave that bounced back) counts in their
    favour;
  * the **CS given up** is pace x typical minus what they took over the window, never below
    zero, turned into gold and XP with what a CS is worth for that role at that stage of the
    game in stored farming minutes (gold: the slope of gold on CS, since passive gold keeps
    coming while you're away; XP: XP per CS, since lane XP stops when you leave);
  * the **typical** CS given up by a window like it (same role and stage of the game, with or
    without the player dying in it), from the same stored games.

Swing then asks the win-probability model what that much gold, XP and CS would have been
worth to the player's team at the end of the window (``swing.evaluate``), for this window and
for a typical one. The model already expects the typical cost (a kill or a death moves the
chance by what such plays usually lead to), so only the difference is charged to the play.
Plays in the same window share it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from riftwatch.features.map import zone

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM")
AREA = {"TOP": frozenset({"top_lane"}), "MIDDLE": frozenset({"mid_lane"}), "BOTTOM": frozenset({"bot_lane"}),
        "JUNGLE": frozenset({"blue_jungle", "red_jungle", "river"})}
ROLE_NAMES = {"TOP": "top laner", "JUNGLE": "jungler", "MIDDLE": "mid laner", "BOTTOM": "bot laner"}
LANING_ENDS = 14        # turret plates fall at 14:00: after it nobody's tied to a lane
SIDE = 2                # minutes away counted on each side of the plays in a run
AFTER = 2               # farming minutes after a window that can make up for it
PRIOR = 3.0             # typical pace counts as this many farming minutes
LAST_MINUTE = 40        # minutes from here on share one typical rate
STAGES = (("early", 0, 14), ("mid", 14, 25), ("late", 25, 10_000))
FILE = "farm.json"
# Used until farm.json is built from stored games (rounded from a build of it).
DEFAULT_RATE = {"TOP": 9.0, "JUNGLE": 9.0, "MIDDLE": 9.5, "BOTTOM": 9.5}
DEFAULT_GOLD = {"TOP": 22.0, "JUNGLE": 25.0, "MIDDLE": 22.0, "BOTTOM": 23.0}
DEFAULT_XP = {"TOP": 66.0, "JUNGLE": 60.0, "MIDDLE": 57.0, "BOTTOM": 47.0}


def stage_of(minute: int) -> str:
    return next(name for name, lo, hi in STAGES if lo <= minute < hi)


def _p(frame: dict, pid: int) -> dict:
    return frame.get(str(pid)) or {}


def cs(frame: dict, pid: int) -> float:
    p = _p(frame, pid)
    return float((p.get("minionsKilled") or 0) + (p.get("jungleMinionsKilled") or 0))


def where(frame: dict, pid: int) -> str | None:
    pos = _p(frame, pid).get("position")
    return zone(float(pos.get("x", 0)), float(pos.get("y", 0))) if pos else None


def in_area(frame: dict, pid: int, role: str, minute: int = 0) -> bool:
    """In the player's own area: their lane (a jungler's jungle or river) during laning; after
    it, anywhere but a base (side waves and camps are wherever they are)."""
    z = where(frame, pid)
    if z is None:                                     # no position: don't call it away
        return True
    if minute >= LANING_ENDS:
        return z not in ("blue_base", "red_base")
    return z in AREA[role]


def involvement(events: list[dict]) -> dict[int, list[float]]:
    """When each player took part in a kill (either side of it), objective or structure."""
    out: dict[int, list[float]] = defaultdict(list)
    for e in events:
        kind, t = e.get("type"), e.get("timestamp", 0)
        if kind == "CHAMPION_KILL":
            pids = [e.get("killerId"), e.get("victimId"), *(e.get("assistingParticipantIds") or [])]
        elif kind == "ELITE_MONSTER_KILL" and e.get("killerTeamId") in (100, 200):
            pids = [e.get("killerId"), *(e.get("assistingParticipantIds") or [])]
        elif kind == "BUILDING_KILL" and e.get("teamId") in (100, 200):
            pids = [e.get("killerId"), *(e.get("assistingParticipantIds") or [])]
        else:
            continue
        for p in pids:
            if 1 <= (p or 0) <= 10:
                out[int(p)].append(t)
    return out


def death_spans(replay) -> dict[int, list[tuple[float, float]]]:
    """Each player's deaths as (died ms, back ms)."""
    out: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for i, victim, back in replay.deaths:
        out[victim].append((replay.times[i], back))
    return out


def minutes(frames: list[dict], pid: int, role: str, deaths: list[tuple[float, float]],
            plays: list[float]) -> list[str]:
    """What each minute m (frame m to frame m + 1) was for the player: "farm", "away" (dead,
    in a play, or out of their area) or "base" (a back: neither farm nor a play). The first
    minute is "base": everyone walks out and minions haven't arrived."""
    n = len(frames) - 1
    home = "blue_base" if pid <= 5 else "red_base"
    out = []
    for m in range(n):
        if m == 0 or home in (where(frames[m], pid), where(frames[m + 1], pid)):
            out.append("base")
        elif in_area(frames[m], pid, role, m) and in_area(frames[m + 1], pid, role, m):
            out.append("farm")
        else:
            out.append("away")
    for died, back in deaths:                         # dead, then the walk back from base
        for m in range(int(died // 60_000), min(n, int(back // 60_000) + 1)):
            out[m] = "away"
    for t in plays:
        m = int(t // 60_000)
        if 0 <= m < n:
            out[m] = "away"
    return out


def farming(frames: list[dict], pid: int, role: str, deaths: list[tuple[float, float]],
            plays: list[float]) -> list[bool]:
    """For each minute: whether it was a farming minute."""
    return [kind == "farm" for kind in minutes(frames, pid, role, deaths, plays)]


def windows(kinds: list[str], plays: set[int]) -> list[tuple[int, int, list[int], list[int]]]:
    """Each run of minutes away that holds a play: (first, last minute charged, the farming
    minutes after it that count, the minutes of plays inside)."""
    n, out, m = len(kinds), [], 0
    while m < n:
        if kinds[m] != "away":
            m += 1
            continue
        a = m
        while m + 1 < n and kinds[m + 1] == "away":
            m += 1
        b = m
        m += 1
        inside = [mm for mm in range(a, b + 1) if mm in plays]
        if not inside:
            continue                                         # a walk with no play in it
        lo, hi = max(a, min(inside) - SIDE), min(b, max(inside) + SIDE)
        after = []
        for mm in range(hi + 1, min(n, hi + 1 + AFTER)):
            if kinds[mm] != "farm":
                break
            after.append(mm)
        out.append((lo, hi, after, inside))
    return out


def _player_minutes(frames: list[dict], pid: int, role: str, deaths, plays) -> tuple[list[str], list[float]]:
    kinds = minutes(frames, pid, role, deaths, plays)
    took = [cs(frames[m + 1], pid) - cs(frames[m], pid) for m in range(len(frames) - 1)]
    return kinds, took


def _given(kinds: list[str], took: list[float], line: list[float], plays: set[int],
           died: set[int]) -> tuple[float, list[tuple[int, int, list[int], list[int], float, bool]]]:
    """The player's pace and each window: (first, last, after, play minutes, CS given up, died)."""
    n = len(kinds)

    def typical(m: int) -> float:
        return 0.0 if m == 0 else float(line[min(m, len(line) - 1)])

    own = [m for m in range(n) if kinds[m] == "farm"]
    prior = PRIOR * fmean(typical(m) for m in range(n)) if n else 0.0
    pace = (sum(took[m] for m in own) + prior) / (sum(typical(m) for m in own) + prior) if prior > 0 else 1.0
    out = []
    for lo, hi, after, inside in windows(kinds, plays):
        lost = max(0.0, sum(pace * typical(mm) - took[mm] for mm in [*range(lo, hi + 1), *after]))
        out.append((lo, hi, after, inside, lost, any(lo <= d <= hi for d in died)))
    return pace, out


def _sensitivity(model, replay, n: int, roles: dict[int, str]) -> dict[int, list[float]]:
    """For each laner and jungler: what one more CS (and its gold and XP) in their lane would
    be worth to their team at each of the ``n`` minute frames, by the model: the weight a
    window ending there gets. One model call for the game."""
    from riftwatch.winprob.state import team_of

    base = [replay.pair(m * 60_000) for m in range(n)]
    pids = [p for p, r in sorted(roles.items()) if r in ROLES]
    states = list(base)
    for pid in pids:
        states += [shifted(b, team_of(pid), DEFAULT_GOLD[roles[pid]], DEFAULT_XP[roles[pid]], 1.0, roles[pid])
                   for b in base]
    p = [float(x) for x in model.blue([a for a, _ in states], [b for _, b in states])]
    out = {}
    for k, pid in enumerate(pids):
        sign = 1.0 if team_of(pid) == 100 else -1.0
        out[pid] = [max(0.0, sign * (p[n * (k + 1) + m] - p[m])) for m in range(n)]
    return out


def build_table(conn, sample: int = 1500, log=print, model=None) -> dict[str, Any]:
    """From a sample of stored games: the typical CS per farming minute for each role and
    minute of a game; what a CS is worth in gold and XP for each role and stage; and the
    typical CS a window of plays gives up, by role, stage and whether the player died in it.

    With ``model``, the typical is weighted by what a CS was worth at the end of each window
    (sum of weight x CS over sum of weights): the charge, worth(CS) - worth(typical), is about
    worth-per-CS x (CS - typical), so this weighting makes it average zero over stored games,
    as an expected cost should. Unweighted, windows late in decided games (long death timers,
    little a CS can change) pull the typical up and credit the average death."""
    from riftwatch.winprob.model import GAME_SQL, players, positions
    from riftwatch.winprob.state import Replay

    rates: dict[str, dict[int, list[float]]] = {r: defaultdict(list) for r in ROLES}
    pairs: dict[str, dict[str, list[tuple[float, float, float]]]] = {r: defaultdict(list) for r in ROLES}
    kept = []                                    # (role, kinds, took, play minutes, death minutes)
    games = 0
    # Pick the sample first: sorting GAME_SQL itself would read every timeline.
    sql = GAME_SQL + """ WHERE t.match_id IN (
                             SELECT tt.match_id FROM match_timelines tt JOIN matches m USING (match_id)
                              WHERE m.queue_id IN (420, 440) ORDER BY md5(tt.match_id) LIMIT %s)"""
    with conn.transaction(), conn.cursor(name="farm_table") as cur:
        cur.itersize = 100
        cur.execute(sql, (sample,))
        for _, frames, events, rows in cur:
            if not frames or len(frames) < 6:
                continue
            roles = positions(rows)
            if model is not None:
                champions, purchases = players(rows)
                catalog = model.catalog if getattr(model, "reads_items", False) else None
                replay = Replay(frames, events, champions, purchases, catalog, roles)
                if replay.winner not in (100, 200):
                    continue
            else:
                replay = Replay(frames, events)
            plays, deaths = involvement(replay.events), death_spans(replay)
            weights = _sensitivity(model, replay, len(frames), roles) if model is not None else {}
            for pid, role in roles.items():
                if role not in ROLES:
                    continue
                kinds, took = _player_minutes(frames, pid, role, deaths.get(pid, []), plays.get(pid, []))
                for m, kind in enumerate(kinds):
                    if kind != "farm":
                        continue
                    a, b = _p(frames[m], pid), _p(frames[m + 1], pid)
                    d_gold = float(b.get("totalGold", 0)) - float(a.get("totalGold", 0))
                    d_xp = float(b.get("xp", 0)) - float(a.get("xp", 0))
                    rates[role][min(m, LAST_MINUTE)].append(took[m])
                    pairs[role][stage_of(m)].append((took[m], d_gold, d_xp))
                kept.append((role, kinds, took, {int(t // 60_000) for t in plays.get(pid, [])},
                             {int(d // 60_000) for d, _ in deaths.get(pid, [])}, weights.get(pid)))
            games += 1
            if games % 500 == 0:
                log(f"  {games:,} games read for farm norms")
    table: dict[str, Any] = {"games": games, "cs_rate": {}, "gold_per_cs": {}, "xp_per_cs": {}, "minutes": {},
                             "typical_cs": {}}
    for role in ROLES:
        line, last = [0.0], DEFAULT_RATE[role]                   # nobody farms the first minute
        for m in range(1, LAST_MINUTE + 1):
            got = rates[role].get(m) or []
            last = fmean(got) if len(got) >= 30 else last
            line.append(round(last, 3))
        table["cs_rate"][role] = line
        table["minutes"][role] = sum(len(v) for v in rates[role].values())
        table["gold_per_cs"][role], table["xp_per_cs"][role] = {}, {}
        for stage, _, _ in STAGES:
            rows_ = pairs[role].get(stage) or []
            if len(rows_) < 100:
                table["gold_per_cs"][role][stage] = DEFAULT_GOLD[role]
                table["xp_per_cs"][role][stage] = DEFAULT_XP[role]
                continue
            c = [r[0] for r in rows_]
            g = [r[1] for r in rows_]
            mc, mg = fmean(c), fmean(g)
            var = sum((x - mc) ** 2 for x in c)
            slope = sum((x - mc) * (y - mg) for x, y in zip(c, g, strict=True)) / var if var else DEFAULT_GOLD[role]
            table["gold_per_cs"][role][stage] = round(slope, 2)
            table["xp_per_cs"][role][stage] = round(sum(r[2] for r in rows_) / max(1.0, sum(c)), 2)
    lost: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)    # (CS, weight)
    for role, kinds, took, plays, died, weight in kept:
        _, found = _given(kinds, took, table["cs_rate"][role], plays, died)
        for lo, hi, after, _, given, dead in found:
            end = min(len(kinds), (after[-1] if after else hi) + 1)
            w = weight[end] if weight is not None else 1.0
            lost[(role, stage_of(lo), "died" if dead else "alive")].append((given, w))
    table["weighted"] = model is not None
    table["plain_cs"] = {}
    for role in ROLES:
        table["typical_cs"][role], table["plain_cs"][role] = {}, {}
        for stage, _, _ in STAGES:
            for k in ("alive", "died"):
                rows_ = lost[(role, stage, k)]
                if len(rows_) < 30:
                    continue
                total_w = sum(w for _, w in rows_)
                plain = fmean(c for c, _ in rows_)
                typical = sum(c * w for c, w in rows_) / total_w if total_w > 0 else plain
                table["typical_cs"][role].setdefault(stage, {})[k] = round(typical, 2)
                table["plain_cs"][role].setdefault(stage, {})[k] = round(plain, 2)
    log(f"  farm norms from {games:,} games: " + ", ".join(
        f"{r.lower()} {table['minutes'][r]:,} farming minutes" for r in ROLES))
    return table


def save_table(table: dict, models_dir) -> Path:
    path = Path(models_dir) / "winprob" / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    return path


def load_table(folder) -> dict | None:
    """farm.json from the model's folder, or None."""
    path = Path(folder) / FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def default_table() -> dict[str, Any]:
    return {"cs_rate": {r: [DEFAULT_RATE[r] * min(1.0, m / 3) for m in range(LAST_MINUTE + 1)] for r in ROLES},
            "gold_per_cs": {r: {s: DEFAULT_GOLD[r] for s, _, _ in STAGES} for r in ROLES},
            "xp_per_cs": {r: {s: DEFAULT_XP[r] for s, _, _ in STAGES} for r in ROLES},
            "typical_cs": {}}


def given_up(replay, frames: list[dict], moments: list[dict], roles: dict[int, str],
             table: dict | None = None) -> list[dict[str, Any]]:
    """Each laner's and jungler's farm given up around their plays: one row per player per run
    of minutes away that holds at least one of their plays, with what a typical run like it
    gives up. ``moments`` are Swing's moments (a player's plays are the non-item moments
    they're in, as a taker or the one who died)."""
    table = table or default_table()
    n = len(frames) - 1
    if n < 2:
        return []
    plays, deaths = involvement(replay.events), death_spans(replay)
    out = []
    for pid, role in sorted(roles.items()):
        if role not in ROLES or role not in table.get("cs_rate", {}):
            continue
        kinds, took = _player_minutes(frames, pid, role, deaths.get(pid, []), plays.get(pid, []))
        mine: dict[int, list[int]] = defaultdict(list)          # minute -> this player's moments
        for j, mo in enumerate(moments):
            if mo.get("part") == "items" or (pid not in (mo.get("players") or []) and pid not in (mo.get("lost") or [])):
                continue
            mine[min(int(mo["t"] // 60_000), n - 1)].append(j)
        died = {int(d // 60_000) for d, _ in deaths.get(pid, [])}
        pace, found = _given(kinds, took, table["cs_rate"][role], set(mine), died)
        for lo, hi, after, inside, lost, dead in found:
            js = [j for mm in inside for j in mine.get(mm, [])]
            if not js:
                continue
            stage = stage_of(lo)
            gold, xp = float(table["gold_per_cs"][role][stage]), float(table["xp_per_cs"][role][stage])
            typical = float(((table.get("typical_cs") or {}).get(role, {}).get(stage) or {})
                            .get("died" if dead else "alive", 0.0))
            out.append({"pid": pid, "role": role, "moments": js, "from": lo, "to": hi, "away": hi - lo + 1,
                        "cs": round(lost, 1), "gold": round(lost * gold), "xp": round(lost * xp),
                        "typical_cs": round(typical, 1), "typical_gold": round(typical * gold),
                        "typical_xp": round(typical * xp), "died": dead,
                        "pace": round(pace, 2), "frame": min(n, (after[-1] if after else hi) + 1)})
    return out


def shifted(pair: tuple[dict, dict], team: int, gold: float, xp: float, cs_: float,
            role: str | None = None) -> tuple[dict, dict]:
    """The state with ``team`` holding that much more gold, XP and CS, all of it in the
    ``role`` player's lane."""
    from riftwatch.winprob.state import LANES

    lane = dict(LANES).get(role or "")
    blue, red = dict(pair[0]), dict(pair[1])
    for state, sign in ((blue, 1 if team == 100 else -1), (red, 1 if team == 200 else -1)):
        state["gold_diff"] = state.get("gold_diff", 0.0) + sign * gold / 1000
        state["xp_diff"] = state.get("xp_diff", 0.0) + sign * xp / 1000
        state["cs_diff"] = state.get("cs_diff", 0.0) + sign * cs_ / 10
        if lane and f"lane_gold_{lane}" in state:
            state[f"lane_gold_{lane}"] += sign * gold / 1000
            state[f"lane_xp_{lane}"] += sign * xp / 1000
    return blue, red
