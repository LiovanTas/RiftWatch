"""One player on one champion: their season numbers beside the same champion's at their rank
and in Master+ games, their build and keystone beside the high-elo ones, their record in each
matchup, and their recent games on it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from riftwatch.stats import champions as champ_stats
from riftwatch.web import insights
from riftwatch.web.gamedata import POSITIONS, GameData
from riftwatch.web.views import ago, kda_ratio

QUEUES = (420, 440, 400)
MIN_PEERS = 30            # fewer games at the player's rank: compare with every rank instead

METRICS = (  # key, label, format, higher is better
    ("win_rate", "Win rate", "{:.0f}%", True),
    ("kda", "KDA", "{:.2f}", True),
    ("cs_min", "CS per minute", "{:.1f}", True),
    ("cs10", "CS at 10 minutes", "{:.0f}", True),
    ("dmg_min", "Damage per minute", "{:,.0f}", True),
    ("gold_min", "Gold per minute", "{:,.0f}", True),
    ("vision_min", "Vision per minute", "{:.2f}", True),
    ("kp", "Kill participation", "{:.0f}%", True),
    ("deaths_10", "Deaths per 10 minutes", "{:.1f}", False),
    ("first_item", "First item finished at", "clock", False),
    ("second_item", "Second item finished at", "clock", False),
)


def _aggregate(conn: psycopg.Connection, where: str, params: list,
               legendary: list[int] | None = None) -> dict[str, float] | None:
    # The first and second legendary items finished, from the timeline's purchase times.
    row = conn.execute(f"""
        SELECT count(*), sum(b.win::int), sum(b.kills), sum(b.deaths), sum(b.assists), sum(b.cs),
               sum(b.damage), sum(b.gold), sum(b.vision_score), sum(b.duration_s),
               avg((b.kills + b.assists)::float8 / greatest(b.team_kills, 1)), avg(f.cs)::float8,
               avg(x.t1)::float8, avg(x.t2)::float8
          FROM participant_builds b
          LEFT JOIN participant_minute_features f
                 ON f.match_id = b.match_id AND f.participant_id = b.participant_id AND f.minute = 10
          LEFT JOIN LATERAL (SELECT (array_agg(u.s ORDER BY u.s))[1] t1, (array_agg(u.s ORDER BY u.s))[2] t2
                               FROM unnest(b.purchases, b.purchase_s) u(item, s) WHERE u.item = ANY(%s)) x ON true
         WHERE NOT b.remake AND {where}""", [legendary or [], *params]).fetchone()
    n, wins, k, d, a, cs, dmg, gold, vision, secs, kp, cs10, t1, t2 = row
    if not n:
        return None
    minutes = secs / 60
    return {"games": n, "win_rate": 100 * wins / n, "kda": (k + a) / max(d, 1), "cs_min": cs / minutes,
            "cs10": cs10, "dmg_min": dmg / minutes, "gold_min": gold / minutes,
            "vision_min": vision / minutes, "kp": 100 * kp, "deaths_10": d / (minutes / 10),
            "first_item": t1, "second_item": t2}


def player_champion(conn: psycopg.Connection, gd: GameData, puuid: str, champion_id: int,
                    region: str, riot_path: str, tier: str) -> dict[str, Any] | None:
    year = datetime.now(UTC).year
    start = datetime(year, 1, 1, tzinfo=UTC)
    with conn.cursor(row_factory=dict_row) as cur:
        games = cur.execute("""SELECT * FROM participant_builds WHERE puuid = %s AND champion_id = %s
                                 AND game_start >= %s AND queue_id = ANY(%s) AND NOT remake
                               ORDER BY game_start DESC""",
                            (puuid, champion_id, start, list(QUEUES))).fetchall()
    if not games:
        return None
    roles = Counter(g["position"] for g in games if g["position"])
    position = roles.most_common(1)[0][0] if roles else None

    legendary = [i for i, it in gd.items.items() if it.is_legendary]
    mine = _aggregate(conn, "b.puuid = %s AND b.champion_id = %s AND b.game_start >= %s AND b.queue_id = ANY(%s)",
                      [puuid, champion_id, start, list(QUEUES)], legendary)
    peers_label, peers = None, None
    if position:
        # Other players only: the player's own games would make "your rank" look like them.
        base = "b.champion_id = %s AND b.position = %s AND b.queue_id = 420 AND b.puuid <> %s"
        if tier != "ALL":
            peers = _aggregate(conn, base + " AND b.match_id IN (SELECT match_id FROM match_samples WHERE tier_bucket = %s)",
                               [champion_id, position, puuid, tier], legendary)
            peers_label = insights.TIER_NAME.get(tier, tier.title())
        if not peers or peers["games"] < MIN_PEERS:
            peers, peers_label = _aggregate(conn, base, [champion_id, position, puuid], legendary), "All ranks"
        high = _aggregate(conn, base + " AND b.match_id IN (SELECT match_id FROM match_samples WHERE tier_bucket = 'MASTER_PLUS')",
                          [champion_id, position, puuid], legendary)
    else:
        high = None

    rows = []
    for key, label, fmt, higher in METRICS:
        values = [g[key] if g and g.get(key) is not None else None for g in (mine, peers, high)]
        if values[0] is None:
            continue
        others = [v for v in values[1:] if v is not None]
        verdict = None
        if others:
            ref = others[-1]                                 # Master+ when there is one
            verdict = "up" if (values[0] >= ref) == higher else "down"
        shown = [(f"{int(v) // 60}:{int(v) % 60:02d}" if fmt == "clock" else fmt.format(v)) if v is not None else "-"
                 for v in values]
        rows.append({"label": label, "values": shown, "verdict": verdict})

    # Build: the player's most-finished legendary items against the high-elo core.
    def legendary(i: int) -> bool:
        it = gd.item(i)
        return bool(it and it.is_legendary)

    counts = Counter(i for g in games for i in (g["items"] or [])[:6] if i and legendary(i))
    my_items = [{"id": i, "icon": gd.item_icon(i), "name": gd.item(i).name, "share": round(100 * n / len(games))}
                for i, n in counts.most_common(3)]
    pro = champ_stats.champion(conn, champion_id, position, None, "master_plus",
                               is_legendary=legendary,
                               is_boots=lambda i: bool(gd.item(i) and gd.item(i).is_boots),
                               is_starter=lambda i: bool(gd.item(i) and gd.item(i).is_starter)) if position else {}
    pro_core = [{"id": i, "icon": gd.item_icon(i), "name": gd.item(i).name if gd.item(i) else str(i)}
                for i in (pro.get("core") or [{}])[0].get("key", ())] if pro.get("core") else []
    keystones = Counter(g["keystone"] for g in games if g["keystone"])
    my_key = keystones.most_common(1)[0][0] if keystones else None
    pro_key = pro["runes"][0]["key"][0] if pro.get("runes") else None

    # Matchups on this champion.
    vs: dict[int, list[bool]] = defaultdict(list)
    for g in games:
        if g["opponent_champion_id"]:
            vs[g["opponent_champion_id"]].append(g["win"])
    matchups = sorted(({"name": gd.champion(o).name, "slug": gd.champion(o).slug, "icon": gd.champion_icon(o),
                        "games": len(r), "wins": sum(r), "win_rate": round(100 * sum(r) / len(r))}
                       for o, r in vs.items()), key=lambda m: (-m["games"], m["win_rate"]))

    recent = [{"match_id": g["match_id"], "win": g["win"], "kda": f"{g['kills']}/{g['deaths']}/{g['assists']}",
               "cs_min": round(g["cs"] / (g["duration_s"] / 60), 1), "when": ago(g["game_start"]),
               "opponent": gd.champion(g["opponent_champion_id"]).name if g["opponent_champion_id"] else None,
               "opponent_icon": gd.champion_icon(g["opponent_champion_id"]) if g["opponent_champion_id"] else None,
               "url": f"/games/{g['match_id']}?as={region}/{riot_path}"} for g in games[:12]]
    k, d, a = (sum(g[x] for g in games) for x in ("kills", "deaths", "assists"))
    return {
        "games": len(games), "wins": sum(g["win"] for g in games), "kda": kda_ratio(k, d, a),
        "position": position, "role": POSITIONS.get(position or "", ""),
        "roles": [{"name": POSITIONS.get(r, r), "games": n} for r, n in roles.most_common()],
        "columns": ["You", peers_label or "Your rank", "Master+"], "rows": rows,
        "peer_games": peers["games"] if peers else 0, "high_games": high["games"] if high else 0,
        "my_items": my_items, "pro_core": pro_core,
        "my_key": {"icon": gd.rune_icon(my_key), "name": gd.rune_name(my_key),
                   "share": round(100 * keystones[my_key] / len(games))} if my_key else None,
        "pro_key": {"icon": gd.rune_icon(pro_key), "name": gd.rune_name(pro_key)} if pro_key else None,
        "matchups": matchups, "recent": recent,
    }
