"""Game-wide statistics over stored ranked games: what each objective is worth, how much a
lead at 15 minutes decides, how long games last, and how each finished item does.

Everything is measured, not modelled: a "first dragon" win rate is the share of games won by
the team that took the first dragon. Taking an objective and winning are tangled up (strong
teams take more objectives), so these are associations, which the page says.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

import psycopg

from riftwatch.stats.champions import _scope, shrunk

FIRSTS = (("champion", "First blood"), ("tower", "First tower"), ("horde", "First voidgrub"),
          ("dragon", "First dragon"), ("riftHerald", "Rift Herald"), ("baron", "First Baron"),
          ("inhibitor", "First inhibitor"), ("atakhan", "Atakhan"))
LEAD_BUCKETS = ((-1e9, -5000, "5k+ behind"), (-5000, -3000, "3-5k behind"), (-3000, -1500, "1.5-3k behind"),
                (-1500, -500, "0.5-1.5k behind"), (-500, 500, "even"), (500, 1500, "0.5-1.5k ahead"),
                (1500, 3000, "1.5-3k ahead"), (3000, 5000, "3-5k ahead"), (5000, 1e9, "5k+ ahead"))
LENGTHS = ((0, 1200, "under 20"), (1200, 1500, "20-25"), (1500, 1800, "25-30"), (1800, 2100, "30-35"),
           (2100, 2400, "35-40"), (2400, 10**6, "40+"))
MIN_ITEM_GAMES = 30
MIN_HAPPENED = 5            # % of games: rarer objectives (a new one, say) aren't listed
CACHE_S = 3600              # reading every game's objectives takes seconds; they change slowly

_cache: dict[tuple, tuple[float, Any]] = {}
_lock = threading.Lock()


def _cached(key: tuple, build):
    with _lock:
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_S:
            return hit[1]
    value = build()
    with _lock:
        _cache[key] = (time.monotonic(), value)
    return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def objectives(conn: psycopg.Connection, rank: str = "all") -> dict[str, Any]:
    return _cached(("objectives", rank), lambda: _objectives(conn, rank))


def _objectives(conn: psycopg.Connection, rank: str) -> dict[str, Any]:
    scope, params = _scope(None, rank)
    teams = conn.execute(f"""
        SELECT m.match_id, (t->>'teamId')::int, (t->>'win')::bool, t->'objectives', m.duration_s
          FROM matches m JOIN ({scope}) s USING (match_id),
               jsonb_array_elements(m.raw->'info'->'teams') t""", params).fetchall()
    games = len({row[0] for row in teams})
    if not games:
        return {"games": 0}
    firsts = []
    for key, label in FIRSTS:
        took = [win for _, _, win, obj, _ in teams if ((obj or {}).get(key) or {}).get("first")]
        if took and 100 * len(took) / games >= MIN_HAPPENED:
            firsts.append({"key": key, "label": label, "games": len(took), "happened": round(100 * len(took) / games),
                           "win_rate": 100 * sum(took) / len(took)})
    dragons = defaultdict(lambda: [0, 0])
    for _, _, win, obj, _ in teams:
        n = min(((obj or {}).get("dragon") or {}).get("kills", 0), 4)
        dragons[n][0] += 1
        dragons[n][1] += win
    side = defaultdict(lambda: [0, 0])
    lengths = defaultdict(lambda: [0, 0])
    for _, team, win, _, duration in teams:
        side[team][0] += 1
        side[team][1] += win
        if team == 100:
            for lo, hi, label in LENGTHS:
                if lo <= duration < hi:
                    lengths[label][0] += 1
                    lengths[label][1] += win
    leads = conn.execute(f"""
        SELECT b.match_id, b.team_id, sum(f.gold), bool_or(b.win)
          FROM participant_minute_features f
          JOIN participant_builds b ON b.match_id = f.match_id AND b.participant_id = f.participant_id
          JOIN ({scope}) s ON s.match_id = b.match_id
         WHERE f.minute = 15 GROUP BY 1, 2""", params).fetchall()
    by_match: dict[str, dict[int, tuple[int, bool]]] = defaultdict(dict)
    for match_id, team, gold, win in leads:
        by_match[match_id][team] = (gold, win)
    buckets = defaultdict(lambda: [0, 0])
    comebacks = behind = 0
    for sides in by_match.values():
        if len(sides) != 2:
            continue
        (g1, w1), (g2, _) = sides[100], sides[200]
        for lead, won in ((g1 - g2, w1), (g2 - g1, not w1)):
            for lo, hi, label in LEAD_BUCKETS:
                if lo <= lead < hi:
                    buckets[label][0] += 1
                    buckets[label][1] += won
            if lead <= -3000:
                behind += 1
                comebacks += won
    return {
        "games": games,
        "firsts": sorted(firsts, key=lambda f: -f["win_rate"]),
        "dragons": [{"label": f"{n}{'+' if n == 4 else ''}", "teams": c, "win_rate": 100 * w / c}
                    for n, (c, w) in sorted(dragons.items())],
        "side": {("Blue" if t == 100 else "Red"): {"games": c, "win_rate": 100 * w / c} for t, (c, w) in side.items()},
        "lengths": [{"label": label, "games": lengths[label][0], "share": round(100 * lengths[label][0] / games),
                     "blue_win_rate": 100 * lengths[label][1] / lengths[label][0] if lengths[label][0] else None}
                    for _, _, label in LENGTHS],
        "leads": [{"label": label, "teams": buckets[label][0],
                   "win_rate": 100 * buckets[label][1] / buckets[label][0] if buckets[label][0] else None}
                  for _, _, label in LEAD_BUCKETS],
        "comeback": {"behind": behind, "won": comebacks,
                     "rate": round(100 * comebacks / behind) if behind else None},
    }


def items(conn: psycopg.Connection, rank: str = "all") -> dict[str, Any]:
    return _cached(("items", rank), lambda: _items(conn, rank))


def _items(conn: psycopg.Connection, rank: str) -> dict[str, Any]:
    """Every finished item: how many players ended with it, their win rate, and the champions
    that build it most."""
    scope, params = _scope(None, rank)
    players = conn.execute(f"""SELECT count(*) FROM participant_builds b JOIN ({scope}) s USING (match_id)
                                WHERE NOT b.remake""", params).fetchone()[0]
    rows = conn.execute(f"""
        SELECT x.item, b.champion_id, count(*), sum(b.win::int)
          FROM participant_builds b JOIN ({scope}) s USING (match_id),
               LATERAL unnest(b.items[1:6]) x(item)
         WHERE NOT b.remake AND x.item > 0
         GROUP BY 1, 2""", params).fetchall()
    per_item: dict[int, dict[str, Any]] = defaultdict(lambda: {"games": 0, "wins": 0, "champions": []})
    for item, champ, n, w in rows:
        it = per_item[item]
        it["games"] += n
        it["wins"] += w
        it["champions"].append((n, champ, w))
    out = []
    for item, it in per_item.items():
        if it["games"] < MIN_ITEM_GAMES:
            continue
        top = sorted(it["champions"], reverse=True)[:4]
        out.append({"item_id": item, "games": it["games"], "win_rate": 100 * it["wins"] / it["games"],
                    "shrunk": 100 * shrunk(it["wins"], it["games"]),
                    "pick_rate": 100 * it["games"] / max(players, 1),
                    "champions": [{"champion_id": c, "games": n, "win_rate": 100 * w / n} for n, c, w in top]})
    return {"players": players, "rows": sorted(out, key=lambda r: -r["games"])}
