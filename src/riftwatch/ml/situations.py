"""Turn a game into (situation, decision, outcome) examples for each role.

A *situation* is what the player could plausibly know at a minute mark: the clock, their
own state, their lane matchup and every lane's state on the scoreboard (kills, levels, CS),
their teammates' positions, where the enemy jungler was last *seen*, and objective history.
Hidden enemy positions are deliberately left out -- the timeline knows them, but the player
usually didn't, and a model trained on them would learn to see through fog of war.

A *decision* is what the player did over the next minute, read from where they were one
frame later and what happened in between. Each role has its own set (a jungler ganks or
invades; a mid laner roams top or bot; a support stays with the ADC or goes warding). Labels
may use enemy positions -- they describe what happened -- features may not.

An *outcome* is what followed in the few minutes *after* the decision minute: objectives,
kills, deaths, gold.

Every game is viewed from the player's own side: red-side coordinates are reflected across
the river, so "own jungle" and "top lane" mean the same thing for both teams. (Flipping
both axes instead would swap top and bot lane.)

Timelines have one frame per minute and no health or ability data, so this sees roams,
recalls, rotations and the *results* of laning (CS, levels, kills), not individual trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from riftwatch.features.extract import REAL_WARDS
from riftwatch.features.map import MAP_X, MAP_Y, zone

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
DECISIONS: dict[str, tuple[str, ...]] = {
    "JUNGLE": ("gank_top", "gank_mid", "gank_bot", "objective", "invade", "farm", "rotate",
               "base", "other"),
    "TOP": ("lane", "roam", "push", "fight", "objective", "base", "other"),
    "MIDDLE": ("lane", "roam_top", "roam_bot", "push", "fight", "objective", "base", "other"),
    "BOTTOM": ("lane", "push", "fight", "roam", "objective", "base", "other"),
    "UTILITY": ("with_adc", "roam_mid", "roam_top", "ward", "fight", "objective", "base", "other"),
}
JUNGLE_DECISIONS = DECISIONS["JUNGLE"]
LANES = ("TOP", "MIDDLE", "BOTTOM")
HOME_LANE = {"TOP": "top", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "bot"}
OPPONENT_ROLE = {"TOP": "TOP", "MIDDLE": "MIDDLE", "BOTTOM": "BOTTOM", "UTILITY": "UTILITY",
                 "JUNGLE": "JUNGLE"}
FIRST_MINUTE = 2
LAST_MINUTE = 15        # last minute whose *decision* is labelled (needs frame m+1)
OUTCOME_MINUTES = 3     # how far ahead outcomes look

_EPIC = {"DRAGON", "ELDER_DRAGON", "HORDE", "RIFTHERALD", "BARON_NASHOR", "ATAKHAN"}

# Lane centerlines in game units. "In lane" means within LANE_WIDTH of one: the coarse map
# zones used elsewhere put blue's gromp inside "top lane", which made clearing a camp look
# like a gank. Mirror-symmetric: each side lane sits EDGE units in from the map's edges.
_EDGE = 1100
LANE_PATHS = {
    "top": ((_EDGE, 3500), (_EDGE, MAP_Y - _EDGE), (MAP_X - 3500, MAP_Y - _EDGE)),
    "mid": ((3000, 3000), (MAP_X - 3000, MAP_Y - 3000)),
    "bot": ((3500, _EDGE), (MAP_X - _EDGE, _EDGE), (MAP_X - _EDGE, MAP_Y - 3500)),
}
LANE_WIDTH = 1000
# A gank needs a target: in lane *and* this close to the enemy laner (or part of a kill).
GANK_RANGE = 2000
DUO_RANGE = 2000        # support counts as "with the ADC" within this distance
_LANE_ROLE = {"top": "TOP", "mid": "MIDDLE", "bot": "BOTTOM"}
BASE_RADIUS = 0.26      # as a fraction of the map, matching features.map


def _segment_distance(px, py, a, b) -> float:
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5


def lane_at(x: float, y: float) -> str | None:
    """"top" / "mid" / "bot" if (x, y) is on that lane outside both bases, else None."""
    u, v = x / MAP_X, y / MAP_Y
    if (u < BASE_RADIUS and v < BASE_RADIUS) or (u > 1 - BASE_RADIUS and v > 1 - BASE_RADIUS):
        return None
    best, best_d = None, LANE_WIDTH
    for lane, path in LANE_PATHS.items():
        d = min(_segment_distance(x, y, a, b) for a, b in zip(path, path[1:], strict=False))
        if d <= best_d:
            best, best_d = lane, d
    return best


def own_side(x: float, y: float, team_id: int) -> tuple[float, float]:
    """Coordinates as seen from ``team_id``'s side, as fractions of the map. Blue is
    unchanged; red is reflected across the river line (u + v = 1)."""
    u, v = x / MAP_X, y / MAP_Y
    if team_id == 200:
        u, v = 1 - v, 1 - u
    return u, v


def own_zone(x: float, y: float, team_id: int) -> str:
    """Map zone from the player's perspective: own_/enemy_ jungle and base, lanes, river."""
    u, v = own_side(x, y, team_id)
    z = zone(u * MAP_X, v * MAP_Y)
    return {"blue_base": "own_base", "red_base": "enemy_base",
            "blue_jungle": "own_jungle", "red_jungle": "enemy_jungle"}.get(z, z)


@dataclass
class Example:
    match_id: str
    participant_id: int
    champion_id: int
    team_id: int
    role: str
    minute: int
    features: dict[str, float]
    decision: str
    outcome: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {"match_id": self.match_id, "participant_id": self.participant_id,
                "champion_id": self.champion_id, "team_id": self.team_id, "role": self.role,
                "minute": self.minute, "decision": self.decision,
                **{f"f_{k}": v for k, v in self.features.items()},
                **{f"o_{k}": v for k, v in self.outcome.items()}}


def _minute_frames(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    # Same rule as feature extraction: frames sit on or just after each minute mark.
    from riftwatch.features.extract import _minute_frames as frames

    return frames(timeline["info"]["frames"])


def _events(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    out = [e for f in timeline["info"]["frames"] for e in f.get("events", [])]
    out.sort(key=lambda e: e["timestamp"])
    return out


def _took_part(pid: int, e: dict[str, Any]) -> bool:
    return e.get("killerId") == pid or pid in (e.get("assistingParticipantIds") or [])


def _pos(frame: dict[str, Any], pid: int | None) -> dict[str, float] | None:
    if pid is None:
        return None
    return frame["participantFrames"][str(pid)].get("position")


def _dist(a: dict[str, float] | None, b: dict[str, float] | None) -> float:
    if not a or not b:
        return 1e9
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5


class _Game:
    """Indexes shared by every player's examples in one game."""

    def __init__(self, match: dict[str, Any], timeline: dict[str, Any]) -> None:
        info = match["info"]
        self.match_id = match["metadata"]["matchId"]
        self.parts = {p["participantId"]: p for p in info["participants"]}
        self.team_of = {pid: p["teamId"] for pid, p in self.parts.items()}
        self.role_of = {pid: p.get("teamPosition") or "" for pid, p in self.parts.items()}
        self.by_team_role = {(self.team_of[pid], self.role_of[pid]): pid for pid in self.parts}
        self.frames = _minute_frames(timeline)
        self.events = _events(timeline)

    def who(self, team: int, role: str) -> int | None:
        return self.by_team_role.get((team, role))


def examples(match: dict[str, Any], timeline: dict[str, Any], role: str) -> list[Example]:
    """One example per minute for each player of ``role`` (both teams)."""
    if role not in DECISIONS:
        raise ValueError(f"unknown role {role!r}")
    g = _Game(match, timeline)
    out: list[Example] = []
    for pid in (p for p in g.parts if g.role_of[p] == role):
        for m in range(FIRST_MINUTE, min(LAST_MINUTE, len(g.frames) - 2) + 1):
            out.append(_example(g, pid, role, m))
    _add_previous_decisions(out, role)
    return out


def jungle_examples(match: dict[str, Any], timeline: dict[str, Any]) -> list[Example]:
    return examples(match, timeline, "JUNGLE")


def _example(g: _Game, pid: int, role: str, m: int) -> Example:
    team = g.team_of[pid]
    enemy = 300 - team
    t0, t1 = g.frames[m]["timestamp"], g.frames[m + 1]["timestamp"]
    pf = g.frames[m]["participantFrames"]
    pos = pf[str(pid)].get("position") or {"x": 0, "y": 0}
    u, v = own_side(pos["x"], pos["y"], team)
    opponent = g.who(enemy, OPPONENT_ROLE[role])
    enemy_jg = g.who(enemy, "JUNGLE")
    ally_jg = g.who(team, "JUNGLE")

    def stat(who: int | None, key: str, frame: dict[str, Any] | None = None) -> float:
        if who is None:
            return 0.0
        return float((frame or g.frames[m])["participantFrames"][str(who)].get(key, 0))

    past = [e for e in g.events if e["timestamp"] <= t0]
    kills = [e for e in past if e["type"] == "CHAMPION_KILL"]
    epics = [e for e in past if e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") in _EPIC]
    team_of = g.team_of

    def team_count(evts, side, mtype=None) -> int:
        return sum(team_of.get(e.get("killerId")) == side
                   and (mtype is None or e.get("monsterType") == mtype) for e in evts)

    plates = [e for e in past if e["type"] == "TURRET_PLATE_DESTROYED"]
    towers = [e for e in past if e["type"] == "BUILDING_KILL"]
    main_cs = "jungleMinionsKilled" if role == "JUNGLE" else "minionsKilled"
    f: dict[str, float] = {
        "minute": m,
        "u": round(u, 4), "v": round(v, 4),
        "level": stat(pid, "level"),
        "current_gold": stat(pid, "currentGold"),
        "total_gold": stat(pid, "totalGold"),
        "cs": stat(pid, main_cs),
        # Scoreboard-visible differences to the lane opponent.
        "level_diff_opp": stat(pid, "level") - stat(opponent, "level"),
        "cs_diff_opp": stat(pid, main_cs) - stat(opponent, main_cs),
        "kill_diff_opp": (sum(e.get("killerId") == pid for e in kills)
                          - sum(e.get("killerId") == opponent for e in kills)),
        "deaths": float(sum(e.get("victimId") == pid for e in kills)),
        "team_kill_diff": float(team_count(kills, team) - team_count(kills, enemy)),
        "own_dragons": float(team_count(epics, team, "DRAGON")),
        "enemy_dragons": float(team_count(epics, enemy, "DRAGON")),
        "own_grubs": float(team_count(epics, team, "HORDE")),
        "minutes_since_dragon": (t0 - max([e["timestamp"] for e in epics
                                           if e.get("monsterType") == "DRAGON"], default=0)) / 60_000,
        # turret plates / towers taken *by* each team (teamId names the team that lost it)
        "own_plates": float(sum(e.get("teamId") == enemy for e in plates)),
        "enemy_plates": float(sum(e.get("teamId") == team for e in plates)),
        "own_towers": float(sum(e.get("teamId") == enemy for e in towers)),
        "enemy_towers": float(sum(e.get("teamId") == team for e in towers)),
        "died_recently": float(any(e.get("victimId") == pid and t0 - e["timestamp"] < 60_000
                                   for e in kills)),
    }
    cs_1 = stat(pid, main_cs, g.frames[m - 1])
    cs_2 = stat(pid, main_cs, g.frames[max(m - 2, 0)])
    f["cs_gain_1m"] = f["cs"] - cs_1
    f["cs_gain_2m"] = f["cs"] - cs_2

    zone_now = own_zone(pos["x"], pos["y"], team)
    for z in ("own_jungle", "enemy_jungle", "river", "top_lane", "mid_lane", "bot_lane", "own_base"):
        f[f"in_{z}"] = float(zone_now == z)
    if role != "JUNGLE":
        f["in_home_lane"] = float(lane_at(pos["x"], pos["y"]) == HOME_LANE[role])

    # Every lane from the player's side: who is winning it (scoreboard) and where the ally
    # laner is (minimap).
    for lane in LANES:
        ally, foe = g.who(team, lane), g.who(enemy, lane)
        key = lane.lower()
        f[f"{key}_level_diff"] = stat(ally, "level") - stat(foe, "level")
        f[f"{key}_cs_diff"] = stat(ally, "minionsKilled") - stat(foe, "minionsKilled")
        f[f"{key}_kill_diff"] = float(sum(e.get("killerId") == ally for e in kills)
                                      - sum(e.get("killerId") == foe for e in kills))
        f[f"{key}_ally_dead"] = float(any(e.get("victimId") == ally and t0 - e["timestamp"] < 30_000
                                          for e in kills))
        apos = _pos(g.frames[m], ally)
        if apos and ally != pid:
            au, av = own_side(apos["x"], apos["y"], team)
            f[f"{key}_ally_push"] = round(au + av, 3)     # 0 at own base corner, 2 at theirs
            f[f"{key}_ally_dist"] = round(((au - u) ** 2 + (av - v) ** 2) ** 0.5, 3)
        else:
            f[f"{key}_ally_push"] = round(u + v, 3) if ally == pid else 0.0
            f[f"{key}_ally_dist"] = 0.0

    # The ally jungler's position is on the minimap, so known; for non-junglers it's one of
    # the strongest signals for when to fight or roam.
    if role != "JUNGLE":
        jpos = _pos(g.frames[m], ally_jg)
        if jpos:
            ju, jv = own_side(jpos["x"], jpos["y"], team)
            f["ally_jg_u"], f["ally_jg_v"] = round(ju, 4), round(jv, 4)
            f["ally_jg_dist"] = round(((ju - u) ** 2 + (jv - v) ** 2) ** 0.5, 3)
        else:
            f["ally_jg_u"] = f["ally_jg_v"] = f["ally_jg_dist"] = 0.0

    # Bot lane plays as a duo.
    if role in ("BOTTOM", "UTILITY"):
        partner = g.who(team, "UTILITY" if role == "BOTTOM" else "BOTTOM")
        f["duo_dist"] = round(_dist(pos, _pos(g.frames[m], partner)) / 1000, 3)
        f["duo_level_diff"] = stat(partner, "level") - stat(g.who(enemy, g.role_of.get(partner, "")), "level")
    if role == "UTILITY":
        f["wards_placed"] = float(sum(e["type"] == "WARD_PLACED" and e.get("creatorId") == pid
                                      and e.get("wardType") in REAL_WARDS for e in past))

    # Where the enemy jungler was last *seen*: fights on the kill feed and minimap only.
    seen = [e for e in kills if enemy_jg is not None and (
        e.get("killerId") == enemy_jg or e.get("victimId") == enemy_jg
        or enemy_jg in (e.get("assistingParticipantIds") or []))]
    if seen:
        last = seen[-1]
        lp = last.get("position") or {"x": MAP_X / 2, "y": MAP_Y / 2}
        su, sv = own_side(lp["x"], lp["y"], team)
        f["enemy_jg_seen_minutes_ago"] = min((t0 - last["timestamp"]) / 60_000, 15.0)
        f["enemy_jg_seen_topside"] = float(sv > su)       # above the mid diagonal
        f["enemy_jg_seen_on_our_half"] = float(su + sv < 1)
    else:
        f["enemy_jg_seen_minutes_ago"] = 15.0
        f["enemy_jg_seen_topside"] = 0.5
        f["enemy_jg_seen_on_our_half"] = 0.5

    if role == "JUNGLE":
        last_gain = 0
        for k in range(m, 0, -1):
            if stat(pid, main_cs, g.frames[k]) > stat(pid, main_cs, g.frames[k - 1]):
                last_gain = k
                break
        f["minutes_since_farm"] = float(m - last_gain)

    npos = _pos(g.frames[m + 1], pid) or {"x": 0, "y": 0}
    window = [e for e in g.events if t0 < e["timestamp"] <= t1]
    cs_gain_next = stat(pid, main_cs, g.frames[m + 1]) - f["cs"]
    if role == "JUNGLE":
        decision = _jungle_decision(g, pid, npos, window, cs_gain_next, m)
    else:
        decision = _laner_decision(g, pid, role, npos, window, m)
    # Outcomes start *after* the decision minute: an objective taken as the decision must
    # not count as its own outcome (that made "objective -> objective" a certainty).
    ahead = [e for e in g.events if t1 < e["timestamp"] <= t1 + OUTCOME_MINUTES * 60_000]
    return Example(g.match_id, pid, g.parts[pid]["championId"], team, role, m, f, decision,
                   _outcome(g, pid, role, ahead, m))


# -- decisions ------------------------------------------------------------------------------------

def _team_objective(g: _Game, pid: int, window) -> bool:
    team = g.team_of[pid]
    return any(e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") in _EPIC
               and g.team_of.get(e.get("killerId")) == team and _took_part(pid, e) for e in window)


def _fight_lane(g: _Game, pid: int, window) -> tuple[bool, str | None]:
    """(took part in a champion kill this minute, the lane it was in or None)."""
    for e in window:
        if e["type"] == "CHAMPION_KILL" and (_took_part(pid, e) or e.get("victimId") == pid):
            p = e.get("position") or {}
            return True, lane_at(p.get("x", -1e9), p.get("y", -1e9))
    return False, None


def _jungle_decision(g: _Game, pid: int, npos, window, cs_gain_next: float, m: int) -> str:
    team = g.team_of[pid]
    if _team_objective(g, pid, window):
        return "objective"
    if own_zone(npos["x"], npos["y"], team) == "own_base":
        return "base"
    # A kill the jungler took part in, in a lane, marks a gank there even if they've left.
    for e in window:
        if e["type"] == "CHAMPION_KILL" and _took_part(pid, e):
            p = e.get("position") or {}
            lane = lane_at(p.get("x", -1e9), p.get("y", -1e9))
            if lane:
                return f"gank_{lane}"
    lane_next = lane_at(npos["x"], npos["y"])
    if lane_next:
        target = g.who(300 - team, _LANE_ROLE[lane_next])
        if _dist(npos, _pos(g.frames[m + 1], target)) <= GANK_RANGE:
            return f"gank_{lane_next}"
    if own_zone(npos["x"], npos["y"], team) == "enemy_jungle":
        return "invade"
    if cs_gain_next >= 3:
        return "farm"
    if lane_next:
        return "rotate"     # in a lane with no target near: crossing the map, shoving, scuttle
    return "other"


def _laner_decision(g: _Game, pid: int, role: str, npos, window, m: int) -> str:
    team = g.team_of[pid]
    home = HOME_LANE[role]
    if _team_objective(g, pid, window):
        return "objective"
    zone_next = own_zone(npos["x"], npos["y"], team)
    if zone_next == "own_base":
        return "base"
    lane_next = lane_at(npos["x"], npos["y"])
    fought, fight_lane = _fight_lane(g, pid, window)

    if role == "UTILITY":
        adc = g.who(team, "BOTTOM")
        if fought and fight_lane not in (None, "bot"):
            return f"roam_{fight_lane}" if fight_lane in ("mid", "top") else "fight"
        if lane_next == "mid":
            return "roam_mid"
        if lane_next == "top":
            return "roam_top"
        if lane_next == "bot" and _dist(npos, _pos(g.frames[m + 1], adc)) <= DUO_RANGE:
            return "with_adc"
        placed = sum(e["type"] == "WARD_PLACED" and e.get("creatorId") == pid
                     and e.get("wardType") in REAL_WARDS for e in window)
        if zone_next in ("river", "own_jungle", "enemy_jungle") and placed:
            return "ward"
        if fought:
            return "fight"
        return "with_adc" if lane_next == "bot" else "other"

    # Pressure on the home lane's turret: plates or the tower itself, with this player.
    if any(e["type"] in ("TURRET_PLATE_DESTROYED", "BUILDING_KILL")
           and _took_part(pid, e) for e in window):
        return "push"
    if fought and fight_lane != home:
        return _roam_label(role, fight_lane, npos, team) if fight_lane else "fight"
    if lane_next and lane_next != home:
        return _roam_label(role, lane_next, npos, team)
    if lane_next == home:
        return "lane"
    if zone_next in ("river", "own_jungle", "enemy_jungle") and _dist_to_lane(npos, home) > 2500:
        return _roam_label(role, None, npos, team)
    return "lane" if _dist_to_lane(npos, home) <= 2500 else "other"


def _dist_to_lane(p, lane: str) -> float:
    path = LANE_PATHS[lane]
    return min(_segment_distance(p["x"], p["y"], a, b) for a, b in zip(path, path[1:], strict=False))


def _roam_label(role: str, lane: str | None, npos, team: int) -> str:
    if role == "MIDDLE":
        if lane in ("top", "bot"):
            return f"roam_{lane}"
        u, v = own_side(npos["x"], npos["y"], team)
        return "roam_top" if v > u else "roam_bot"
    return "roam"


def _add_previous_decisions(examples: list[Example], role: str, lags: int = 2) -> None:
    """What the player did in each of the last ``lags`` minutes -- their own actions, so
    known to them. Play is sequential: a full clear is usually followed by a gank, a back
    by a return to lane."""
    by_player: dict[int, dict[int, str]] = {}
    for e in examples:
        by_player.setdefault(e.participant_id, {})[e.minute] = e.decision
    for e in examples:
        history = by_player[e.participant_id]
        for lag in range(1, lags + 1):
            prev = history.get(e.minute - lag, "none")
            for d in DECISIONS[role]:
                e.features[f"prev{lag}_{d}"] = float(prev == d)


# -- outcomes -------------------------------------------------------------------------------------

def _outcome(g: _Game, pid: int, role: str, ahead, m: int) -> dict[str, float]:
    team = g.team_of[pid]
    enemy = 300 - team
    start = m + 1                                   # end of the decision minute
    later = min(start + OUTCOME_MINUTES, len(g.frames) - 1)
    team_of = g.team_of

    def team_gold(frame, side) -> float:
        return sum(p.get("totalGold", 0) for k, p in frame["participantFrames"].items()
                   if team_of.get(int(k)) == side)

    swing_now = team_gold(g.frames[start], team) - team_gold(g.frames[start], enemy)
    swing_later = team_gold(g.frames[later], team) - team_gold(g.frames[later], enemy)
    kills = [e for e in ahead if e["type"] == "CHAMPION_KILL"]
    out = {
        "team_objective": float(any(e["type"] == "ELITE_MONSTER_KILL"
                                    and team_of.get(e.get("killerId")) == team for e in ahead)),
        "enemy_objective": float(any(e["type"] == "ELITE_MONSTER_KILL"
                                     and team_of.get(e.get("killerId")) == enemy for e in ahead)),
        "team_kills": float(sum(team_of.get(e.get("killerId")) == team for e in kills)),
        "team_deaths": float(sum(team_of.get(e.get("victimId")) == team for e in kills)),
        "player_died": float(any(e.get("victimId") == pid for e in kills)),
        "gold_swing": swing_later - swing_now,
    }
    opponent = g.who(enemy, OPPONENT_ROLE[role])
    if role != "JUNGLE" and opponent is not None:
        def cs_diff(frame) -> float:
            mine = frame["participantFrames"][str(pid)].get("minionsKilled", 0)
            theirs = frame["participantFrames"][str(opponent)].get("minionsKilled", 0)
            return float(mine - theirs)
        out["lane_cs_swing"] = cs_diff(g.frames[later]) - cs_diff(g.frames[start])
    return out
