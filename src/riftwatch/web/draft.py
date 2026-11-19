"""Champ-select draft assistant: which champion to pick in a role, given the bans, both teams'
picks so far, and (with a Riot ID) the player's own champion pool and playstyle.

Each candidate's expected win chance is built additively in log-odds from components, each
measured from stored ranked games and pulled towards "no effect" for small samples:

  1. strength -- the champion's win rate in this role at the player's rank (shrunk towards
     50%, harder than the tier board: the best of many noisy candidates is usually lucky)
  2. lane -- its record against the enemy laner, shrunk towards its own overall rate (so only
     the matchup's own effect counts, not the champion's strength twice)
  3. with your team / against their team -- its record alongside each ally and against each
     other enemy, each pair shrunk hard (pairs are small samples), the sum capped
  4. team comp -- what it does to the team's damage mix and frontline, from how teams with
     each mix have done in stored games (all-physical teams win 44%), scaled by how much of the
     comp is known yet
  5. comfort -- how much experience on a champion helps on average (stored games: 49.7% on a
     first game, 52.5% after ten), plus how this player does on it beside everyone else (their
     Swing against the champion's typical, shrunk by the Swing rating's spread; or their win
     rate when there's no Swing)
  6. playstyle -- how the player does on this class of champion against their own average

Riot's public API doesn't expose champ select, so picks are entered by hand; nothing reads the
game client.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import psycopg

from riftwatch.stats import champions as champ_stats
from riftwatch.web.gamedata import POSITIONS, GameData

QUEUES = (420, 440)
MIN_ROLE_GAMES = 30            # a champion needs this many games in the role to be suggested...
MIN_POOL_GAMES = 3             # ... unless the player has this many on it there
# Priors (games of "no effect" each record is pulled towards). Stronger than the tier board's
# 40: picking the best of ~90 noisy candidates rewards lucky small samples, so the draft
# shrinks harder to keep its win chances honest.
K_STRENGTH = 100
K_LANE = 80
K_PAIR = 150                   # an ally or other-enemy pair (noisier still)
K_POOL = 30                    # a player's own record on a champion
K_CLASS = 100                  # a player's record on a class
PAIR_CAP = 0.20                # most the ally or the enemy pairs can add, log-odds (~5 points)
PERSONAL_CAP = 0.50            # most comfort can add or take, log-odds (~12 points)
DEFAULT_RATING_K = 37.0        # Swing rating shrinkage (games) when validation isn't there
RANK_FOR_TIER = {"IRON": "gold_minus", "BRONZE": "gold_minus", "SILVER": "gold_minus", "GOLD": "gold_minus",
                 "PLATINUM": "platinum_plus", "EMERALD": "emerald_plus", "DIAMOND": "diamond_plus",
                 "MASTER_PLUS": "master_plus"}
COMFORT_BUCKETS = ((0, 0, "0"), (1, 2, "1-2"), (3, 9, "3-9"), (10, 29, "10-29"), (30, 10_000, "30+"))
PARTS = (("strength", "Strength"), ("lane", "Lane matchup"), ("allies", "With your team"),
         ("enemies", "Against their team"), ("comp", "Team comp"), ("comfort", "Your comfort"),
         ("style", "Your playstyle"))


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def pts(x: float) -> float:
    """A log-odds change as percentage points near an even game."""
    return round(100 * (sigmoid(x) - 0.5), 1)


def toward(wins: float, games: float, prior: float, k: float) -> float:
    """A win rate pulled towards ``prior`` as if ``k`` games at that rate were added."""
    return (wins + k * prior) / (games + k)


# -- the inputs and the evidence ------------------------------------------------------------

@dataclass
class Draft:
    role: str
    bans: list[int] = field(default_factory=list)
    allies: list[tuple[int, str | None]] = field(default_factory=list)
    enemies: list[tuple[int, str | None]] = field(default_factory=list)

    @property
    def taken(self) -> set[int]:
        return set(self.bans) | {c for c, _ in self.allies} | {c for c, _ in self.enemies}


@dataclass
class Evidence:
    """Everything the score reads, gathered from the database (``gather``) or made by hand in
    tests."""
    role_table: dict[int, tuple[int, int]]                  # champion -> (games, wins) in the role
    classes: dict[int, tuple[str, float]]                   # champion -> (class, magic damage share)
    comp: dict[str, Any]                                    # team-level tables (``comp_tables``)
    lane_opponent: int | None = None
    lane: dict[int, tuple[int, int]] = field(default_factory=dict)          # candidate -> vs the laner
    pairs: dict[tuple[int, int, bool], tuple[int, int]] = field(default_factory=dict)  # (cand, pick, ally)
    player: dict[str, Any] | None = None
    rank_label: str = "All ranks"


def _interp(x: float, points: list[tuple[float, float]]) -> float:
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return points[-1][1]


def _magic_rate(comp: dict, share: float) -> float:
    return _interp(share, [(c, r) for c, r, _ in comp["magic"]])


def _tank_rate(comp: dict, tanks: int) -> float:
    n, w = comp["tanks"].get(min(tanks, 2), (0, 0))
    return w / n if n else 0.5


def _comfort_rate(comp: dict, games: int) -> tuple[float, str]:
    for lo, hi, label in COMFORT_BUCKETS:
        if lo <= games <= hi:
            n, w = comp["comfort"].get(label, (0, 0))
            return (w / n if n else comp["overall"]), label
    return comp["overall"], "0"


# -- the score ------------------------------------------------------------------------------

def needs(draft: Draft, ev: Evidence) -> list[dict]:
    """What the player's team and the enemy team look like so far, in plain words."""
    out = []
    allies = [ev.classes.get(c) for c, _ in draft.allies if ev.classes.get(c)]
    if allies:
        share = sum(k[1] for k in allies) / len(allies)
        if share < 0.3:
            low = _magic_rate(ev.comp, 0.2)
            out.append({"kind": "need", "text": f"Your team so far is {100 - round(100 * share)}% physical damage. "
                        f"All-physical teams have won {100 * low:.0f}% of stored games: a magic-damage pick helps."})
        elif share > 0.62:
            out.append({"kind": "need", "text": f"Your team so far is {round(100 * share)}% magic damage: "
                        "a physical-damage pick balances it."})
        if not any(k[0] == "Tank" for k in allies) and len(allies) >= 2:
            out.append({"kind": "need", "text": "No tank yet: teams with a tank have won "
                        f"{100 * _tank_rate(ev.comp, 1):.1f}% against {100 * _tank_rate(ev.comp, 0):.1f}% without."})
    foes = [ev.classes.get(c) for c, _ in draft.enemies if ev.classes.get(c)]
    if foes:
        share = sum(k[1] for k in foes) / len(foes)
        if share > 0.55:
            out.append({"kind": "enemy", "text": f"The enemy is {round(100 * share)}% magic damage so far: magic resist will matter."})
        elif share < 0.3:
            out.append({"kind": "enemy", "text": f"The enemy is {100 - round(100 * share)}% physical damage so far: armor will matter."})
        tanks = sum(1 for k in foes if k[0] == "Tank")
        if tanks >= 2:
            out.append({"kind": "enemy", "text": f"The enemy has {tanks} tanks: damage that cuts through armor or health helps."})
    return out


def score(draft: Draft, ev: Evidence, name_of=lambda c: str(c), limit: int = 12) -> dict[str, Any]:
    """Every candidate's expected win chance and its parts, best first, plus picks to avoid."""
    taken = draft.taken
    player = ev.player or {}
    pool = player.get("champs", {})
    candidates = {c for c, (n, _) in ev.role_table.items() if n >= MIN_ROLE_GAMES}
    candidates |= {c for c, p in pool.items() if p.get("role_games", 0) >= MIN_POOL_GAMES}
    candidates -= taken
    allies_known = len(draft.allies)
    comp_weight = (allies_known + 1) / 5          # how much of our comp is known with this pick
    ally_classes = [ev.classes.get(c) for c, _ in draft.allies if ev.classes.get(c)]
    avg_share = ev.comp.get("avg_magic", 0.42)
    overall_rate = ev.comp.get("overall", 0.5)
    rows = []
    for c in candidates:
        games, wins = ev.role_table.get(c, (0, 0))
        base = toward(wins, games, 0.5, K_STRENGTH) if games else 0.5
        parts: dict[str, dict[str, Any]] = {}
        parts["strength"] = {"x": logit(base), "games": games,
                             "detail": f"{100 * base:.1f}% as {POSITIONS.get(draft.role, draft.role).lower()} "
                                       f"({ev.rank_label}, {games:,} games)" if games else "too few games in this role to say"}
        # lane
        if ev.lane_opponent:
            n, w = ev.lane.get(c, (0, 0))
            p = toward(w, n, base, K_LANE)
            parts["lane"] = {"x": logit(p) - logit(base), "games": n,
                             "detail": (f"{w}-{n - w} against {name_of(ev.lane_opponent)}" if n else
                                        f"no stored games against {name_of(ev.lane_opponent)}")}
        # pairs
        for key, members, ally in (("allies", [a for a, _ in draft.allies], True),
                                   ("enemies", [e for e, _ in draft.enemies if e != ev.lane_opponent], False)):
            if not members:
                continue
            total, notes, games_seen = 0.0, [], 0
            for m in members:
                n, w = ev.pairs.get((c, m, ally), (0, 0))
                games_seen += n
                if not n:
                    continue
                d = logit(toward(w, n, base, K_PAIR)) - logit(base)
                total += d
                notes.append((d, f"{'with' if ally else 'vs'} {name_of(m)} {w}-{n - w}"))
            total = max(-PAIR_CAP, min(PAIR_CAP, total))
            notes.sort(key=lambda t: -abs(t[0]))
            parts[key] = {"x": total, "games": games_seen,
                          "detail": ", ".join(t for _, t in notes[:2]) or "no stored games together"}
        # comp
        cls = ev.classes.get(c)
        if cls:
            known = [k[1] for k in ally_classes]
            filler = 4 - len(known)
            with_c = (sum(known) + cls[1] + filler * avg_share) / 5
            without = (sum(known) + (filler + 1) * avg_share) / 5
            dmg = logit(_magic_rate(ev.comp, with_c)) - logit(_magic_rate(ev.comp, without))
            tanks = sum(1 for k in ally_classes if k[0] == "Tank")
            front = (logit(_tank_rate(ev.comp, tanks + 1)) - logit(_tank_rate(ev.comp, tanks))) if cls[0] == "Tank" else 0.0
            parts["comp"] = {"x": comp_weight * (dmg + front), "games": None,
                             "detail": f"{cls[0].lower()}, {round(100 * cls[1])}% magic damage; your team "
                                       f"{round(100 * with_c)}% magic with it"}
        # comfort and playstyle
        if player:
            mine = pool.get(c, {})
            total_games = mine.get("games", 0)
            rate, bucket = _comfort_rate(ev.comp, total_games)
            x = logit(rate) - logit(overall_rate)
            detail = f"{total_games} games on it" if total_games else "never played it"
            swing = (player.get("swing") or {}).get(c)
            if swing and swing.get("typical") is not None and swing["games"] >= MIN_POOL_GAMES:
                k = player.get("rating_k", DEFAULT_RATING_K)
                gap = (swing["avg"] - swing["typical"]) * swing["games"] / (swing["games"] + k)
                x += logit(min(0.95, max(0.05, 0.5 + gap / 100)))
                detail += f"; Swing {swing['avg']:+.1f} a game vs {swing['typical']:+.1f} typical"
            elif mine.get("role_games", 0) >= MIN_POOL_GAMES:
                n, w = mine["role_games"], mine["role_wins"]
                x += logit(toward(w, n, base, K_POOL)) - logit(base)
                detail += f"; {w}-{n - w} in this role"
            parts["comfort"] = {"x": max(-PERSONAL_CAP, min(PERSONAL_CAP, x)), "games": total_games, "detail": detail}
            if cls and player.get("classes"):
                n, w = player["classes"].get(cls[0], (0, 0))
                pn, pw = player.get("overall", (0, 0))
                mean = pw / pn if pn else 0.5
                if n:
                    parts["style"] = {"x": logit(toward(w, n, mean, K_CLASS)) - logit(mean), "games": n,
                                      "detail": f"{w}-{n - w} on {cls[0].lower()}s ({100 * w / n:.0f}%, you average {100 * mean:.0f}%)"}
        total = sum(p["x"] for p in parts.values())
        comps = [{"key": key, "label": label, "pts": pts(parts[key]["x"]), "detail": parts[key]["detail"],
                  "games": parts[key]["games"]} for key, label in PARTS if key in parts]
        reasons = sorted((p for p in comps if p["key"] != "strength" and abs(p["pts"]) >= 0.3), key=lambda p: -abs(p["pts"]))[:3]
        rows.append({"champion_id": c, "win": round(100 * sigmoid(total), 1), "total": total, "parts": comps,
                     "reasons": reasons, "games": games, "pool_games": pool.get(c, {}).get("games", 0),
                     "lane_x": parts.get("lane", {}).get("x")})
    rows.sort(key=lambda r: -r["total"])
    avoid = []
    if ev.lane_opponent:
        bad = sorted((r for r in rows if r["lane_x"] is not None and r["lane_x"] < -0.08), key=lambda r: r["lane_x"])
        avoid = bad[:4]
    yours = [r for r in rows if r["pool_games"] >= MIN_POOL_GAMES][:5]
    return {"picks": rows[:limit], "avoid": avoid, "yours": yours, "needs": needs(draft, ev),
            "candidates": len(rows), "rank": ev.rank_label}


# -- gathering the evidence -----------------------------------------------------------------

def role_table(conn: psycopg.Connection, role: str, rank: str) -> dict[int, tuple[int, int]]:
    """Each champion's ranked solo games and wins in a role at a rank (cached ten minutes)."""
    def build():
        scope, params = champ_stats._scope(None, rank)
        return {c: (n, w) for c, n, w in conn.execute(
            f"""SELECT champion_id, count(*), sum(win::int) FROM participant_builds JOIN ({scope}) s USING (match_id)
                 WHERE position = %s AND NOT remake GROUP BY 1""", [*params, role])}
    return champ_stats.cached(("draft_role", role, rank), build)


def comp_tables(conn: psycopg.Connection, classes: dict[int, tuple[str, float]]) -> dict[str, Any]:
    """How teams have done by damage mix and by number of tanks, and players by how many
    games they'd already played on the champion (cached ten minutes)."""
    def build():
        teams: dict[tuple, list] = defaultdict(list)
        result: dict[tuple, bool] = {}
        for match_id, team, champ, win in conn.execute(
                """SELECT match_id, team_id, champion_id, win FROM participant_builds
                    WHERE queue_id = ANY(%s) AND NOT remake""", (list(QUEUES),)):
            teams[(match_id, team)].append(champ)
            result[(match_id, team)] = win
        magic: dict[int, list] = defaultdict(lambda: [0, 0, 0.0])
        tanks: dict[int, list] = defaultdict(lambda: [0, 0])
        shares = []
        for key, champs in teams.items():
            known = [classes[c] for c in champs if c in classes]
            if len(champs) != 5 or len(known) < 4:
                continue
            share = sum(k[1] for k in known) / len(known)
            shares.append(share)
            b = min(int(share * 10), 8)          # tenths of magic share
            magic[b][0] += 1
            magic[b][1] += result[key]
            magic[b][2] += share
            t = min(sum(1 for k in known if k[0] == "Tank"), 2)
            tanks[t][0] += 1
            tanks[t][1] += result[key]
        comfort = {}
        for label, n, w in conn.execute("""
                SELECT CASE WHEN prior = 0 THEN '0' WHEN prior < 3 THEN '1-2' WHEN prior < 10 THEN '3-9'
                            WHEN prior < 30 THEN '10-29' ELSE '30+' END, count(*), sum(win::int)
                  FROM (SELECT win, row_number() OVER (PARTITION BY puuid, champion_id ORDER BY game_start) - 1 AS prior
                          FROM participant_builds WHERE queue_id = ANY(%s) AND NOT remake) x GROUP BY 1""",
                                        (list(QUEUES),)):
            comfort[label] = (n, w)
        total_n = sum(n for n, _ in comfort.values())
        total_w = sum(w for _, w in comfort.values())
        # Damage-mix rates, merged so each point has enough teams, as (mean share, rate, teams).
        points, acc = [], [0, 0, 0.0]
        for b in sorted(magic):
            acc = [acc[0] + magic[b][0], acc[1] + magic[b][1], acc[2] + magic[b][2]]
            if acc[0] >= 400:
                points.append((acc[2] / acc[0], acc[1] / acc[0], acc[0]))
                acc = [0, 0, 0.0]
        if acc[0] and points:
            n0 = points[-1][2] + acc[0]
            points[-1] = ((points[-1][0] * points[-1][2] + acc[2]) / n0,
                          (points[-1][1] * points[-1][2] + acc[1]) / n0, n0)
        return {"magic": points or [(0.42, 0.5, 0)], "tanks": {k: tuple(v) for k, v in tanks.items()},
                "comfort": comfort, "overall": total_w / total_n if total_n else 0.5,
                "avg_magic": sum(shares) / len(shares) if shares else 0.42, "teams": len(shares)}
    return champ_stats.cached(("draft_comp",), build)


def lane_opponent(conn: psycopg.Connection, draft: Draft) -> int | None:
    """The enemy in the player's role: as entered, else the enemy pick most played there."""
    for champ, role in draft.enemies:
        if role == draft.role:
            return champ
    unroled = [c for c, r in draft.enemies if not r]
    best, best_share = None, 0.0
    for c in unroled:
        rows = dict(conn.execute("""SELECT position, count(*) FROM participant_builds
                                     WHERE champion_id = %s AND position <> '' AND queue_id = ANY(%s)
                                     GROUP BY 1""", (c, list(QUEUES))).fetchall())
        total = sum(rows.values())
        share = rows.get(draft.role, 0) / total if total else 0.0
        if share > max(best_share, 0.35):
            best, best_share = c, share
    return best


def gather(conn: psycopg.Connection, gd: GameData, draft: Draft, classes: dict[int, tuple[str, float]],
           puuid: str | None = None, rank: str | None = None, win_model=None, rating_k: float | None = None) -> Evidence:
    """The evidence for one draft: the role at the player's rank (or all ranks), lane and pair
    records for the candidates, team-level tables, and the player's pool and playstyle."""
    from riftwatch.web import views

    if rank is None:
        rank = RANK_FOR_TIER.get(views.player_tier(conn, puuid), "all") if puuid else "all"
    table = role_table(conn, draft.role, rank)
    if sum(n for n, _ in table.values()) < 2000 and rank != "all":
        rank, table = "all", role_table(conn, draft.role, "all")
    ev = Evidence(role_table=table, classes=classes, comp=comp_tables(conn, classes),
                  rank_label=champ_stats.RANKS.get(rank, champ_stats.RANKS["all"])[0])
    if puuid:
        ev.player = with_role(player_profile(conn, puuid, classes, win_model, rating_k), draft.role)
    candidates = {c for c, (n, _) in table.items() if n >= MIN_ROLE_GAMES}
    if ev.player:
        candidates |= {c for c, p in ev.player["champs"].items() if p["role_games"] >= MIN_POOL_GAMES}
    candidates -= draft.taken
    ev.lane_opponent = lane_opponent(conn, draft)
    if ev.lane_opponent and candidates:
        ev.lane = {c: (n, w) for c, n, w in conn.execute(
            """SELECT champion_id, count(*), sum(win::int) FROM participant_builds
                WHERE position = %s AND opponent_champion_id = %s AND queue_id = ANY(%s) AND NOT remake
                  AND champion_id = ANY(%s) GROUP BY 1""",
            (draft.role, ev.lane_opponent, list(QUEUES), list(candidates)))}
    picks = [c for c, _ in draft.allies] + [c for c, _ in draft.enemies if c != ev.lane_opponent]
    if picks and candidates:
        for c, other, same, n, w in conn.execute(
                """SELECT x.champion_id, y.champion_id, x.team_id = y.team_id, count(*), sum(x.win::int)
                     FROM participant_builds x
                     JOIN participant_builds y ON y.match_id = x.match_id AND y.participant_id <> x.participant_id
                    WHERE x.position = %s AND x.queue_id = ANY(%s) AND NOT x.remake
                      AND x.champion_id = ANY(%s) AND y.champion_id = ANY(%s)
                    GROUP BY 1, 2, 3""", (draft.role, list(QUEUES), list(candidates), picks)):
            ev.pairs[(c, other, bool(same))] = (n, w)
    return ev


def player_profile(conn: psycopg.Connection, puuid: str, classes: dict[int, tuple[str, float]],
                   win_model=None, rating_k: float | None = None) -> dict[str, Any] | None:
    """The player's champion pool (games and wins, in the role and overall), record by class,
    and Swing on each champion against its typical."""
    rows = conn.execute("""SELECT champion_id, position, count(*), sum(win::int) FROM participant_builds
                            WHERE puuid = %s AND queue_id = ANY(%s) AND NOT remake GROUP BY 1, 2""",
                        (puuid, list(QUEUES))).fetchall()
    if not rows:
        return None
    champs: dict[int, dict[str, int]] = defaultdict(lambda: {"games": 0, "wins": 0, "role_games": 0, "role_wins": 0})
    by_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = [0, 0]
    for champ, _, n, w in rows:
        champs[champ]["games"] += n
        champs[champ]["wins"] += w
        cls = classes.get(champ)
        if cls:
            by_class[cls[0]][0] += n
            by_class[cls[0]][1] += w
        total[0] += n
        total[1] += w
    return {"champs": dict(champs), "classes": {k: tuple(v) for k, v in by_class.items()}, "overall": tuple(total),
            "positions": rows, "swing": _swing(conn, win_model, puuid), "rating_k": rating_k or DEFAULT_RATING_K}


def with_role(profile: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    """The profile with each champion's games and wins in ``role`` filled in."""
    if not profile:
        return profile
    for champ, position, n, w in profile["positions"]:
        if position == role:
            profile["champs"][champ]["role_games"] += n
            profile["champs"][champ]["role_wins"] += w
    return profile


def _swing(conn, win_model, puuid: str) -> dict[int, dict]:
    if win_model is None:
        return {}
    try:
        from riftwatch.web import swingview

        return swingview.champion_table(conn, win_model, puuid)
    except Exception:                        # Swing is optional for the draft; never break it
        return {}


# -- the request ----------------------------------------------------------------------------

def parse_ids(text: str | None, limit: int) -> list[tuple[int, str | None]]:
    """"103:MIDDLE,64,..." -> [(103, 'MIDDLE'), (64, None)], at most ``limit``, no repeats."""
    out, seen = [], set()
    for part in (text or "").split(","):
        champ, _, role = part.strip().partition(":")
        if not champ.strip().isdigit():
            continue
        c = int(champ)
        if c in seen:
            continue
        seen.add(c)
        out.append((c, role.strip().upper() if role.strip().upper() in POSITIONS else None))
        if len(out) >= limit:
            break
    return out


def suggest(conn: psycopg.Connection, gd: GameData, draft: Draft, classes: dict[int, tuple[str, float]],
            puuid: str | None = None, rank: str | None = None, win_model=None, rating_k: float | None = None,
            limit: int = 12) -> dict[str, Any]:
    """The draft assistant's answer, ready for the page: picks with their parts, picks to avoid,
    the player's own best options and what the teams need."""
    ev = gather(conn, gd, draft, classes, puuid, rank, win_model, rating_k)

    def name_of(c: int) -> str:
        return gd.champion(c).name

    result = score(draft, ev, name_of, limit)

    def view(r: dict) -> dict:
        champ = gd.champion(r["champion_id"])
        return {"id": champ.id, "name": champ.name, "slug": champ.slug, "icon": gd.champion_icon(champ.id),
                "win": r["win"], "parts": r["parts"], "reasons": r["reasons"], "games": r["games"],
                "pool_games": r["pool_games"]}
    return {"role": draft.role, "role_name": POSITIONS.get(draft.role, draft.role), "rank": result["rank"],
            "picks": [view(r) for r in result["picks"]], "avoid": [view(r) for r in result["avoid"]],
            "yours": [view(r) for r in result["yours"]], "needs": result["needs"],
            "lane_opponent": name_of(ev.lane_opponent) if ev.lane_opponent else None,
            "personal": ev.player is not None, "candidates": result["candidates"],
            "teams": ev.comp.get("teams", 0)}


# -- the routes -----------------------------------------------------------------------------

def champion_classes(gd: GameData, win_model=None) -> dict[int, tuple[str, float]]:
    """Each champion's class and magic damage share: from the win model's catalog (measured
    from stored games) when there is one, else Data Dragon's class and a neutral 40%."""
    catalog = getattr(win_model, "catalog", None)
    if catalog is not None and catalog.champions:
        return dict(catalog.champions)
    return {cid: ((c.tags or ["Fighter"])[0], 0.4) for cid, c in gd.champions.items()}


def register(app, *, pool, game_data, render, win_model=None, models_dir=None) -> None:
    """Mount /draft and /api/draft/suggest on the site."""
    from fastapi import HTTPException, Query
    from fastapi.responses import HTMLResponse

    from riftwatch.web import recap

    def rating_k() -> float | None:
        try:
            from riftwatch.winprob import validate

            v = validate.load(models_dir) if models_dir else None
            return float(v["rating"]["k"]) if v and v.get("rating") else None
        except Exception:
            return None

    @app.get("/draft", response_class=HTMLResponse)
    def draft_page(role: str = "", me: str = Query("", max_length=40),
                   as_: str = Query("", alias="as", max_length=60)) -> HTMLResponse:
        """Champ select: enter bans and picks, get champions for your role and why."""
        gd = game_data()
        if not me and as_:                       # ?as=<region>/<Name-TAG>, as game pages use
            _, _, rid = as_.partition("/")
            name, sep, tag = rid.rpartition("-")
            me = f"{name}#{tag}" if sep else ""
        champs = sorted(({"id": c.id, "name": c.name, "slug": c.slug, "icon": gd.champion_icon(c.id)}
                         for c in gd.champions.values()), key=lambda c: c["name"])
        import json

        return render("draft.html", nav="draft", role=role.upper() if role.upper() in POSITIONS else "",
                      me=me, champs_json=json.dumps(champs), positions=list(POSITIONS.items()))

    @app.get("/api/draft/suggest")
    def draft_suggest(role: str = Query(..., max_length=10), bans: str = Query("", max_length=200),
                      allies: str = Query("", max_length=120), enemies: str = Query("", max_length=150),
                      me: str = Query("", max_length=40), rank: str = Query("", max_length=20)) -> dict[str, Any]:
        """Champions for ``role`` given the bans and picks (champion ids, picks as id:ROLE)."""
        role = role.upper()
        if role not in POSITIONS:
            raise HTTPException(400, "role must be TOP, JUNGLE, MIDDLE, BOTTOM or UTILITY")
        gd = game_data()
        draft = Draft(role=role, bans=[c for c, _ in parse_ids(bans, 10)],
                      allies=parse_ids(allies, 4), enemies=parse_ids(enemies, 5))
        with pool.connection() as conn:
            found = None
            name, sep, tag = me.strip().partition("#")
            if sep and name.strip() and tag.strip():
                found = recap.locate(conn, name.strip(), tag.strip())
            out = suggest(conn, gd, draft, champion_classes(gd, win_model), found[0] if found else None,
                          rank if rank in champ_stats.RANKS else None, win_model, rating_k())
        out["me"] = f"{found[2]}#{found[3]}" if found else None
        out["me_missing"] = bool(sep and not found)
        return out
