"""Champion statistics from the games RiftWatch has stored: a tier list and per-champion
builds, like the champion pages of op.gg and u.gg.

Ranked solo games only, remakes left out. Rates are over the games in scope (a patch, a rank
group): pick rate = games played / games in scope, ban rate = games banned / games in scope.
Win rates are shrunk towards 50% by PRIOR_GAMES (a champion 3-0 isn't a 100% champion), and a
champion's tier combines that win rate with how popular it is, ranked within its role.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, defaultdict
from statistics import median
from typing import Any

import psycopg
from psycopg.rows import dict_row

RANKED_SOLO = 420
PRIOR_GAMES = 40
MIN_GAMES = 20            # a champion needs this many games in a role to be listed
MIN_ROLE_SHARE = 0.08     # ... and this share of its own games
MIN_MATCHUP = 6           # matchups with fewer games aren't shown
CACHE_S = 600
RANKS = {
    "all": ("All ranks", None),
    "master_plus": ("Master+", ("MASTER_PLUS",)),
    "diamond_plus": ("Diamond+", ("DIAMOND", "MASTER_PLUS")),
    "emerald_plus": ("Emerald+", ("EMERALD", "DIAMOND", "MASTER_PLUS")),
    "platinum_plus": ("Platinum+", ("PLATINUM", "EMERALD", "DIAMOND", "MASTER_PLUS")),
    "gold_minus": ("Gold and below", ("IRON", "BRONZE", "SILVER", "GOLD")),
}
TIER_CUTS = ((0.05, "S+"), (0.17, "S"), (0.40, "A"), (0.70, "B"), (0.90, "C"), (1.01, "D"))

_cache: dict[tuple, tuple[float, Any]] = {}
_lock = threading.Lock()


def _cached(key: tuple, build):
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_S:
            return hit[1]
    value = build()
    with _lock:
        _cache[key] = (now, value)
    return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _scope(patch: str | None, rank: str) -> tuple[str, list]:
    """SQL selecting the match ids in scope, and its parameters."""
    where, params = ["queue_id = %s"], [RANKED_SOLO]
    if patch:
        where.append("patch = %s")
        params.append(patch)
    buckets = RANKS.get(rank, RANKS["all"])[1]
    if buckets:
        where.append("match_id IN (SELECT match_id FROM match_samples WHERE tier_bucket = ANY(%s))")
        params.append(list(buckets))
    return f"SELECT match_id FROM matches WHERE {' AND '.join(where)}", params


def patches(conn: psycopg.Connection) -> list[tuple[str, int]]:
    def build():
        rows = conn.execute("SELECT patch, count(*) FROM matches WHERE queue_id = %s "
                            "GROUP BY patch", (RANKED_SOLO,)).fetchall()
        return sorted(rows, key=lambda r: [int(x) for x in r[0].split(".") if x.isdigit()],
                      reverse=True)
    return _cached(("patches",), build)


def shrunk(wins: float, games: float) -> float:
    return (wins + PRIOR_GAMES * 0.5) / (games + PRIOR_GAMES)


def tier_list(conn: psycopg.Connection, patch: str | None = None, rank: str = "all") -> dict:
    """Every champion in every role it's really played in, with win, pick and ban rates and a
    tier, plus its hardest lane matchups."""
    return _cached(("tiers", patch, rank), lambda: _tier_list(conn, patch, rank))


def _tier_list(conn: psycopg.Connection, patch: str | None, rank: str) -> dict:
    scope, params = _scope(patch, rank)
    total = conn.execute(f"SELECT count(*) FROM ({scope}) s", params).fetchone()[0]
    if not total:
        return {"games": 0, "rows": []}
    picks = conn.execute(
        f"""SELECT champion_id, position, count(*), sum(win::int), avg(kills)::float8,
                   avg(deaths)::float8, avg(assists)::float8
              FROM participant_builds JOIN ({scope}) s USING (match_id)
             WHERE NOT remake AND position <> '' GROUP BY 1, 2""", params).fetchall()
    bans = dict(conn.execute(
        f"""SELECT champion_id, count(DISTINCT match_id) FROM match_bans
              JOIN ({scope}) s USING (match_id) GROUP BY 1""", params).fetchall())
    matchups = conn.execute(
        f"""SELECT champion_id, position, opponent_champion_id, count(*), sum(win::int)
              FROM participant_builds JOIN ({scope}) s USING (match_id)
             WHERE NOT remake AND position <> '' AND opponent_champion_id IS NOT NULL
             GROUP BY 1, 2, 3 HAVING count(*) >= %s""", [*params, MIN_MATCHUP]).fetchall()
    worst: dict[tuple, list] = defaultdict(list)
    for champ, pos, opp, n, w in matchups:
        worst[(champ, pos)].append((shrunk(w, n), opp, n, w))
    per_champ = Counter()
    for champ, _, n, *_ in picks:
        per_champ[champ] += n
    rows = []
    for champ, pos, n, w, k, d, a in picks:
        if n < MIN_GAMES or n < MIN_ROLE_SHARE * per_champ[champ]:
            continue
        counters = sorted(worst[(champ, pos)])[:3]
        rows.append({"champion_id": champ, "position": pos, "games": n, "wins": w,
                     "win_rate": 100 * w / n, "win_rate_shrunk": 100 * shrunk(w, n),
                     "pick_rate": 100 * n / total, "ban_rate": 100 * bans.get(champ, 0) / total,
                     "role_share": 100 * n / per_champ[champ],
                     "kda": (k + a) / max(d, 0.01),
                     "counters": [{"champion_id": opp, "games": cn,
                                   "win_rate": 100 * cw / cn} for _, opp, cn, cw in counters]})
    for pos in {r["position"] for r in rows}:
        group = [r for r in rows if r["position"] == pos]
        mid_pick = median(r["pick_rate"] for r in group)
        for r in group:
            r["score"] = (r["win_rate_shrunk"] - 50) + 1.5 * math.log(r["pick_rate"] / mid_pick)
        group.sort(key=lambda r: -r["score"])
        for i, r in enumerate(group):
            share = (i + 0.5) / len(group)
            r["tier"] = next(t for cut, t in TIER_CUTS if share < cut)
            r["role_rank"] = i + 1
    rows.sort(key=lambda r: -r["score"])
    return {"games": total, "rows": rows}


def champion(conn: psycopg.Connection, champion_id: int, position: str | None = None,
             patch: str | None = None, rank: str = "all", is_legendary=None, is_boots=None,
             is_starter=None) -> dict:
    """Builds, runes, spells, skill order and matchups for one champion in one role."""
    key = ("champion", champion_id, position, patch, rank)
    return _cached(key, lambda: _champion(conn, champion_id, position, patch, rank,
                                          is_legendary or (lambda i: False),
                                          is_boots or (lambda i: False),
                                          is_starter or (lambda i: False)))


def _champion(conn, champion_id, position, patch, rank, is_legendary, is_boots, is_starter):
    scope, params = _scope(patch, rank)
    total = conn.execute(f"SELECT count(*) FROM ({scope}) s", params).fetchone()[0] or 1
    with conn.cursor(row_factory=dict_row) as cur:
        all_rows = cur.execute(
            f"""SELECT position, win, kills, deaths, assists, cs, damage, gold, duration_s,
                       items, spells, keystone, primary_style, sub_style, perks, skill_order,
                       purchases, purchase_s, opponent_champion_id
                  FROM participant_builds JOIN ({scope}) s USING (match_id)
                 WHERE champion_id = %s AND NOT remake""", [*params, champion_id]).fetchall()
    if not all_rows:
        return {"games": 0}
    roles = Counter(r["position"] for r in all_rows if r["position"])
    position = position if position in roles else (roles.most_common(1)[0][0] if roles else "")
    rows = [r for r in all_rows if r["position"] == position] or all_rows
    n = len(rows)
    wins = sum(r["win"] for r in rows)
    bans = conn.execute(f"""SELECT count(DISTINCT match_id) FROM match_bans
                              JOIN ({scope}) s USING (match_id) WHERE champion_id = %s""",
                        [*params, champion_id]).fetchone()[0]
    minutes = sum(r["duration_s"] for r in rows) / 60 or 1

    def top(key_of, limit: int, min_games: int = 2):
        tally: dict[Any, list[int]] = defaultdict(lambda: [0, 0])
        for r in rows:
            key = key_of(r)
            if key is None:
                continue
            tally[key][0] += 1
            tally[key][1] += r["win"]
        best = sorted(((k, g, w) for k, (g, w) in tally.items() if g >= min_games),
                      key=lambda kgw: (-kgw[1], -kgw[2]))[:limit]
        return [{"key": k, "games": g, "wins": w, "win_rate": 100 * w / g,
                 "pick_rate": 100 * g / n} for k, g, w in best]

    def starting(r):
        items = [i for i, s in zip(r["purchases"] or [], r["purchase_s"] or [], strict=False)
                 if s <= 60 and is_starter(i)]
        return tuple(sorted(items)) or None

    def core(r, count=3):
        done = [i for i in (r["purchases"] or []) if is_legendary(i)]
        unique = list(dict.fromkeys(done))
        return tuple(unique[:count]) if len(unique) >= count else None

    def boots(r):
        return next((i for i in (r["purchases"] or []) if is_boots(i)), None)

    def nth_item(r, k):
        unique = list(dict.fromkeys(i for i in (r["purchases"] or []) if is_legendary(i)))
        return unique[k] if len(unique) > k else None

    def max_order(r):
        """Which of Q, W, E is maxed first, second, third (from the first 15 level-ups)."""
        order = (r["skill_order"] or "")[:15]
        if len(order) < 9:
            return None
        counts = {k: order.count(k) for k in "QWE"}
        reached = {}
        for k in "QWE":
            ups = [i for i, s in enumerate(order) if s == k]
            reached[k] = ups[4] if len(ups) >= 5 else 99 + (5 - counts[k]) * 10
        return tuple(sorted("QWE", key=lambda k: reached[k]))

    matchups: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["opponent_champion_id"]:
            matchups[r["opponent_champion_id"]][0] += 1
            matchups[r["opponent_champion_id"]][1] += r["win"]
    vs = [{"champion_id": c, "games": g, "wins": w, "win_rate": 100 * w / g,
           "shrunk": 100 * shrunk(w, g)} for c, (g, w) in matchups.items() if g >= MIN_MATCHUP]
    skill_strings = top(lambda r: (r["skill_order"] or "")[:15] if len(r["skill_order"] or "") >= 15 else None, 1)
    return {
        "games": n, "wins": wins, "win_rate": 100 * wins / n, "pick_rate": 100 * n / total,
        "ban_rate": 100 * bans / total, "position": position, "scope_games": total,
        "roles": [{"position": p, "games": g, "share": 100 * g / sum(roles.values())}
                  for p, g in roles.most_common()],
        "kda": (sum(r["kills"] for r in rows) + sum(r["assists"] for r in rows))
               / max(sum(r["deaths"] for r in rows), 1),
        "cs_min": sum(r["cs"] for r in rows) / minutes,
        "dmg_min": sum(r["damage"] for r in rows) / minutes,
        "runes": top(lambda r: (r["keystone"], r["primary_style"], r["sub_style"])
                     if r["keystone"] else None, 3),
        "rune_pages": top(lambda r: tuple(r["perks"]) if r["perks"] else None, 1),
        "spells": top(lambda r: tuple(sorted(r["spells"])) if r["spells"] else None, 2),
        "starting": top(starting, 3),
        "boots": top(boots, 3),
        "core": top(core, 5),
        "fourth": top(lambda r: nth_item(r, 3), 4),
        "fifth": top(lambda r: nth_item(r, 4), 4),
        "sixth": top(lambda r: nth_item(r, 5), 4),
        "skill_max": top(max_order, 2),
        "skill_order": skill_strings[0]["key"] if skill_strings else None,
        "counters": sorted(vs, key=lambda v: v["shrunk"])[:6],
        "best_against": sorted(vs, key=lambda v: -v["shrunk"])[:6],
    }


def leaderboard_names(conn: psycopg.Connection, puuids: list[str]) -> dict[str, tuple[str, str]]:
    """Riot IDs RiftWatch has seen for these players (from their newest stored game)."""
    rows = conn.execute(
        """SELECT DISTINCT ON (puuid) puuid, riot_name, riot_tag FROM participant_builds
            WHERE puuid = ANY(%s) AND riot_name IS NOT NULL ORDER BY puuid, game_start DESC""",
        (puuids,)).fetchall()
    return {p: (n, t) for p, n, t in rows}
