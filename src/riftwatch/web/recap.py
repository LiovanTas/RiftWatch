"""A player's season recap and side-by-side player comparison.

Both read ``participant_builds`` only, so they cost no Riot requests: the recap covers every
stored game of the season, the comparison the two players' recent ranked games and any games
they've played together or against each other.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import psycopg
from psycopg.rows import dict_row

from riftwatch.stats.champions import shrunk
from riftwatch.web.gamedata import POSITIONS, GameData
from riftwatch.web.views import _scores, kda_ratio, path_id, played_with, rank_card

QUEUES = (420, 440, 400)
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _rows(conn: psycopg.Connection, sql: str, params: tuple) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, params).fetchall()


def _streaks(results: list[bool]) -> tuple[int, int]:
    best_w = best_l = run = 0
    last = None
    for won in results:
        run = run + 1 if won == last else 1
        last = won
        if won:
            best_w = max(best_w, run)
        else:
            best_l = max(best_l, run)
    return best_w, best_l


def season_recap(conn: psycopg.Connection, gd: GameData, puuid: str, region: str,
                 year: int | None = None) -> dict[str, Any] | None:
    """Everything the recap page shows for one season (calendar year)."""
    year = year or datetime.now(UTC).year
    start, end = datetime(year, 1, 1, tzinfo=UTC), datetime(year + 1, 1, 1, tzinfo=UTC)
    games = _rows(conn, """
        SELECT * FROM participant_builds
         WHERE puuid = %s AND game_start >= %s AND game_start < %s AND queue_id = ANY(%s) AND NOT remake
         ORDER BY game_start""", (puuid, start, end, list(QUEUES)))
    if not games:
        return None
    n = len(games)
    wins = sum(g["win"] for g in games)
    k, d, a = (sum(g[x] for g in games) for x in ("kills", "deaths", "assists"))
    minutes = sum(g["duration_s"] for g in games) / 60

    # MVP / ACE needs everyone in each game.
    everyone: dict[str, list[dict]] = defaultdict(list)
    for p in _rows(conn, "SELECT * FROM participant_builds WHERE match_id = ANY(%s)",
                   ([g["match_id"] for g in games],)):
        everyone[p["match_id"]].append(p)
    mvp = ace = 0
    scored = []
    for g in games:
        players = everyone[g["match_id"]]
        if len(players) < 10:
            continue
        scores = _scores(players)
        mine = scores[g["participant_id"]]
        scored.append((mine, g))
        same = [p for p in players if p["win"] == g["win"]]
        if max(same, key=lambda p: scores[p["participant_id"]])["participant_id"] == g["participant_id"]:
            mvp += g["win"]
            ace += not g["win"]

    by_champ: dict[int, list[dict]] = defaultdict(list)
    for g in games:
        by_champ[g["champion_id"]].append(g)

    def champ_view(cid: int, gs: list[dict]) -> dict:
        w = sum(x["win"] for x in gs)
        return {"name": gd.champion(cid).name, "slug": gd.champion(cid).slug, "icon": gd.champion_icon(cid),
                "games": len(gs), "wins": w, "win_rate": round(100 * w / len(gs)),
                "kda": kda_ratio(sum(x["kills"] for x in gs), sum(x["deaths"] for x in gs),
                                 sum(x["assists"] for x in gs))}

    top = [champ_view(c, gs) for c, gs in sorted(by_champ.items(), key=lambda kv: -len(kv[1]))[:5]]
    best = max(((c, gs) for c, gs in by_champ.items() if len(gs) >= 8),
               key=lambda kv: shrunk(sum(x["win"] for x in kv[1]), len(kv[1])), default=None)
    roles = Counter(g["position"] for g in games if g["position"])
    month = defaultdict(lambda: [0, 0])
    for g in games:
        m = month[g["game_start"].month]
        m[0] += 1
        m[1] += g["win"]
    best_w, best_l = _streaks([g["win"] for g in games])
    top_game = max(scored, key=lambda sg: sg[0], default=None)
    most_kills = max(games, key=lambda g: (g["kills"], -g["deaths"]))

    def game_view(g: dict, score: float | None = None) -> dict:
        return {"champion": gd.champion(g["champion_id"]).name, "icon": gd.champion_icon(g["champion_id"]),
                "kda": f"{g['kills']}/{g['deaths']}/{g['assists']}", "win": g["win"],
                "date": g["game_start"].strftime("%b %d"), "score": score, "match_id": g["match_id"]}

    duo = played_with(conn, puuid, region, games=n)
    return {
        "year": year, "games": n, "wins": wins, "losses": n - wins, "win_rate": round(100 * wins / n),
        "hours": round(minutes / 60, 1), "kills": k, "deaths": d, "assists": a, "kda": kda_ratio(k, d, a),
        "pentas": sum(g["penta_kills"] for g in games), "quadras": sum(g["quadra_kills"] for g in games),
        "triples": sum(g["triple_kills"] for g in games), "first_bloods": sum(g["first_blood"] for g in games),
        "mvp": mvp, "ace": ace, "champions_played": len(by_champ),
        "cs_min": round(sum(g["cs"] for g in games) / minutes, 1),
        "damage": sum(g["damage"] for g in games), "gold": sum(g["gold"] for g in games),
        "top": top, "best": champ_view(*best) if best else None,
        "roles": [{"name": POSITIONS.get(r, r), "games": c, "share": round(100 * c / n)}
                  for r, c in roles.most_common()],
        "months": [{"label": MONTHS[m - 1], "games": v[0], "win_rate": round(100 * v[1] / v[0])}
                   for m, v in sorted(month.items())],
        "win_streak": best_w, "loss_streak": best_l,
        "top_game": game_view(top_game[1], top_game[0]) if top_game else None,
        "most_kills": game_view(most_kills),
        "duo": duo[0] if duo else None,
        # For the activity heatmap, bucketed in the visitor's own time zone by the page.
        "played": [[int(g["game_start"].timestamp() * 1000), int(g["win"])] for g in games],
    }


# -- comparison ----------------------------------------------------------------------------

def locate(conn: psycopg.Connection, name: str, tag: str) -> tuple[str, str, str, str] | None:
    """(puuid, platform, name, tag) for a Riot ID RiftWatch has stored, in any region."""
    row = conn.execute("""SELECT puuid, platform, game_name, tag_line FROM accounts
                           WHERE lower(game_name) = lower(%s) AND lower(tag_line) = lower(%s)""",
                       (name, tag)).fetchone()
    if row:
        return row
    return conn.execute("""SELECT puuid, lower(split_part(match_id, '_', 1)), riot_name, riot_tag
                             FROM participant_builds
                            WHERE lower(riot_name) = lower(%s) AND lower(riot_tag) = lower(%s)
                            ORDER BY game_start DESC LIMIT 1""", (name, tag)).fetchone()


METRICS = (  # key, label, format, higher is better
    ("win_rate", "Win rate", "{:.0f}%", True),
    ("kda", "KDA", "{:.2f}", True),
    ("kp", "Kill participation", "{:.0f}%", True),
    ("cs_min", "CS per minute", "{:.1f}", True),
    ("dmg_min", "Damage per minute", "{:,.0f}", True),
    ("gold_min", "Gold per minute", "{:,.0f}", True),
    ("vision_min", "Vision per minute", "{:.2f}", True),
    ("deaths_10", "Deaths per 10 minutes", "{:.1f}", False),
)


def player_stats(conn: psycopg.Connection, gd: GameData, puuid: str, games: int = 50) -> dict | None:
    rows = _rows(conn, """SELECT * FROM participant_builds WHERE puuid = %s AND queue_id IN (420, 440)
                           AND NOT remake ORDER BY game_start DESC LIMIT %s""", (puuid, games))
    if not rows:
        return None
    n = len(rows)
    minutes = sum(r["duration_s"] for r in rows) / 60
    k, d, a = (sum(r[x] for r in rows) for x in ("kills", "deaths", "assists"))
    roles = Counter(r["position"] for r in rows if r["position"])
    champs = Counter(r["champion_id"] for r in rows)
    return {
        "games": n,
        "win_rate": 100 * sum(r["win"] for r in rows) / n,
        "kda": (k + a) / max(d, 1),
        "kp": 100 * sum((r["kills"] + r["assists"]) / max(1, r["team_kills"]) for r in rows) / n,
        "cs_min": sum(r["cs"] for r in rows) / minutes,
        "dmg_min": sum(r["damage"] for r in rows) / minutes,
        "gold_min": sum(r["gold"] for r in rows) / minutes,
        "vision_min": sum(r["vision_score"] for r in rows) / minutes,
        "deaths_10": d / (minutes / 10),
        "role": POSITIONS.get(roles.most_common(1)[0][0], "") if roles else "",
        "champions": [{"name": gd.champion(c).name, "icon": gd.champion_icon(c), "games": g}
                      for c, g in champs.most_common(4)],
        "icon": gd.profile_icon(rows[0]["profile_icon"]), "level": rows[0]["summoner_level"],
        "by_champ": {c: [sum(1 for r in rows if r["champion_id"] == c),
                         sum(r["win"] for r in rows if r["champion_id"] == c)] for c in champs},
    }


def head_to_head(conn: psycopg.Connection, a: str, b: str) -> dict[str, int]:
    """Games the two played on the same team, and against each other (and who won those)."""
    rows = conn.execute("""SELECT x.team_id = y.team_id, x.win FROM participant_builds x
                             JOIN participant_builds y ON y.match_id = x.match_id
                            WHERE x.puuid = %s AND y.puuid = %s AND NOT x.remake""", (a, b)).fetchall()
    together = [won for same, won in rows if same]
    against = [won for same, won in rows if not same]
    return {"together": len(together), "together_wins": sum(together),
            "against": len(against), "a_wins": sum(against), "b_wins": len(against) - sum(against)}


def compare(conn: psycopg.Connection, gd: GameData, ids: list[tuple[str, str]],
            platform_region: dict[str, str]) -> dict[str, Any]:
    from riftwatch.db import repo

    sides = []
    for name, tag in ids:
        found = locate(conn, name, tag)
        side: dict[str, Any] = {"name": name, "tag": tag, "found": bool(found),
                                "search": f"/search?region=auto&riot_id={quote(name + '#' + tag)}"}
        if found:
            puuid, platform, name, tag = found
            region = platform_region.get(platform, platform)
            side.update(puuid=puuid, name=name, tag=tag, region=region,
                        url=f"/players/{region}/{path_id(name, tag)}",
                        rank=rank_card(repo.latest_rank(conn, puuid), gd),
                        stats=player_stats(conn, gd, puuid))
        sides.append(side)
    rows = []
    if len(sides) == 2 and all(s.get("stats") for s in sides):
        for key, label, fmt, higher in METRICS:
            va, vb = sides[0]["stats"][key], sides[1]["stats"][key]
            better = None if round(va, 2) == round(vb, 2) else (0 if (va > vb) == higher else 1)
            top = max(va, vb) or 1
            rows.append({"label": label, "a": fmt.format(va), "b": fmt.format(vb), "better": better,
                         "a_pct": round(100 * va / top), "b_pct": round(100 * vb / top)})
    h2h = (head_to_head(conn, sides[0]["puuid"], sides[1]["puuid"])
           if len(sides) == 2 and all(s.get("puuid") for s in sides) else None)
    shared = []
    if len(sides) == 2 and all(s.get("stats") for s in sides):
        a, b = sides[0]["stats"]["by_champ"], sides[1]["stats"]["by_champ"]
        for c in sorted(set(a) & set(b), key=lambda c: -(a[c][0] + b[c][0]))[:8]:
            shared.append({"name": gd.champion(c).name, "slug": gd.champion(c).slug, "icon": gd.champion_icon(c),
                           "a_games": a[c][0], "a_wr": round(100 * a[c][1] / a[c][0]),
                           "b_games": b[c][0], "b_wr": round(100 * b[c][1] / b[c][0])})
    return {"sides": sides, "rows": rows, "h2h": h2h, "shared": shared}


# -- season badges -------------------------------------------------------------------------

def achievements(conn: psycopg.Connection, gd: GameData, puuid: str, year: int | None = None) -> list[dict]:
    """Badges earned this season, from stored games: each says what earned it."""
    year = year or datetime.now(UTC).year
    start, end = datetime(year, 1, 1, tzinfo=UTC), datetime(year + 1, 1, 1, tzinfo=UTC)
    games = _rows(conn, """
        SELECT match_id, team_id, win, deaths, duration_s, cs, vision_score, penta_kills, quadra_kills,
               first_blood, champion_id, position
          FROM participant_builds
         WHERE puuid = %s AND game_start >= %s AND game_start < %s AND queue_id = ANY(%s) AND NOT remake
         ORDER BY game_start""", (puuid, start, end, list(QUEUES)))
    if not games:
        return []
    n = len(games)
    gold15: dict[str, dict[int, int]] = defaultdict(dict)
    for match_id, team, gold in conn.execute(
            """SELECT b.match_id, b.team_id, sum(f.gold) FROM participant_minute_features f
                 JOIN participant_builds b ON b.match_id = f.match_id AND b.participant_id = f.participant_id
                WHERE f.match_id = ANY(%s) AND f.minute = 15 GROUP BY 1, 2""",
            ([g["match_id"] for g in games],)):
        gold15[match_id][team] = gold
    comebacks = sum(1 for g in games if g["win"] and len(gold15.get(g["match_id"], {})) == 2
                    and gold15[g["match_id"]][g["team_id"]] - gold15[g["match_id"]][300 - g["team_id"]] <= -2500)
    minutes = lambda g: g["duration_s"] / 60  # noqa: E731
    pentas = sum(g["penta_kills"] for g in games)
    quadras = sum(g["quadra_kills"] for g in games)
    deathless = sum(1 for g in games if g["deaths"] == 0 and minutes(g) >= 15)
    farmed = sum(1 for g in games if g["cs"] / minutes(g) >= 8)
    vision = sum(g["vision_score"] for g in games) / sum(minutes(g) for g in games)
    streak, _ = _streaks([g["win"] for g in games])
    long_games = sum(1 for g in games if minutes(g) >= 40)
    first_bloods = sum(g["first_blood"] for g in games)
    champs = Counter(g["champion_id"] for g in games)
    top_champ, top_n = champs.most_common(1)[0]
    roles = Counter(g["position"] for g in games if g["position"])
    candidates = [
        ("pentakill", "Pentakill", "sword", pentas >= 1, f"{pentas} pentakill{'s' if pentas != 1 else ''} this season"),
        ("quadra", "Quadra club", "sword", quadras >= 3, f"{quadras} quadra kills"),
        ("untouchable", "Untouchable", "star", deathless >= 5, f"{deathless} games without dying"),
        ("farmer", "Farmer", "chart", farmed >= 10, f"{farmed} games at 8+ CS a minute"),
        ("eyes", "Eyes everywhere", "eye", n >= 20 and vision >= 1.2, f"{vision:.2f} vision a minute this season"),
        ("streak", "Hot streak", "trophy", streak >= 5, f"{streak} wins in a row"),
        ("comeback", "Comeback kid", "history", comebacks >= 3, f"{comebacks} wins from 2.5k+ behind at 15 minutes"),
        ("marathon", "Marathoner", "history", long_games >= 5, f"{long_games} games of 40+ minutes"),
        ("first_strike", "First strike", "sword", first_bloods >= 15, f"{first_bloods} first bloods"),
        ("one_trick", "One-trick", "star", top_n >= 50 and top_n / n >= 0.4,
         f"{gd.champion(top_champ).name} in {round(100 * top_n / n)}% of games"),
        ("flex", "Flex player", "chart", sum(1 for c in roles.values() if c >= 10) >= 4,
         f"{sum(1 for c in roles.values() if c >= 10)} roles with 10+ games"),
        ("grinder", "Grinder" if n >= 500 else "Centurion", "trophy", n >= 100, f"{n} games this season"),
    ]
    return [{"key": key, "name": name, "icon": icon, "detail": detail}
            for key, name, icon, earned, detail in candidates if earned]


# -- personal records and champions to try ---------------------------------------------------

def records(conn: psycopg.Connection, gd: GameData, puuid: str, region: str, riot_path: str) -> list[dict]:
    """The player's best single games among everything stored."""
    # Games under 15 minutes are early surrenders (an AFK the remake vote didn't catch), not
    # real records: a normal surrender can't come sooner.
    games = _rows(conn, """SELECT * FROM participant_builds WHERE puuid = %s AND queue_id = ANY(%s)
                            AND NOT remake AND duration_s >= 900""", (puuid, list(QUEUES)))
    if not games:
        return []
    mins = lambda g: g["duration_s"] / 60  # noqa: E731

    def view(label: str, g: dict, value: str) -> dict:
        return {"label": label, "value": value, "champion": gd.champion(g["champion_id"]).name,
                "icon": gd.champion_icon(g["champion_id"]), "win": g["win"],
                "kda": f"{g['kills']}/{g['deaths']}/{g['assists']}", "date": g["game_start"].strftime("%b %d, %Y"),
                "url": f"/games/{g['match_id']}?as={region}/{riot_path}"}

    long_enough = [g for g in games if mins(g) >= 15] or games
    wins = [g for g in games if g["win"]]
    best_kda = [g for g in games if g["kills"] + g["assists"] >= 10]
    out = [
        view("Most kills", (g := max(games, key=lambda g: (g["kills"], -g["deaths"]))), f"{g['kills']} kills"),
        view("Most assists", (g := max(games, key=lambda g: g["assists"])), f"{g['assists']} assists"),
        view("Most damage", (g := max(games, key=lambda g: g["damage"])), f"{g['damage']:,} damage"),
        view("Best farm", (g := max(long_enough, key=lambda g: g["cs"] / mins(g))), f"{g['cs'] / mins(g):.1f} CS/min"),
        view("Most vision", (g := max(games, key=lambda g: g["vision_score"])), f"{g['vision_score']} vision score"),
        view("Longest game", (g := max(games, key=lambda g: g["duration_s"])),
             f"{g['duration_s'] // 60}m {g['duration_s'] % 60:02d}s"),
    ]
    if wins:
        g = min(wins, key=lambda g: g["duration_s"])
        out.append(view("Fastest win", g, f"{g['duration_s'] // 60}m {g['duration_s'] % 60:02d}s"))
    if best_kda:
        g = max(best_kda, key=lambda g: (g["kills"] + g["assists"]) / max(1, g["deaths"]))
        out.append(view("Best KDA", g, "Perfect" if g["deaths"] == 0 else
                        f"{(g['kills'] + g['assists']) / g['deaths']:.1f} KDA"))
    return out


TIER_BONUS = {"S+": 3, "S": 2, "A": 1}


def to_try(conn: psycopg.Connection, gd: GameData, puuid: str, tier_rows: list[dict], limit: int = 4) -> dict | None:
    """Champions strong in the player's main role that they've barely played, preferring ones
    that share a class with the champions they already win on."""
    games = conn.execute("""SELECT champion_id, position, count(*), sum(win::int) FROM participant_builds
                             WHERE puuid = %s AND queue_id IN (420, 440, 400) AND NOT remake
                             GROUP BY 1, 2""", (puuid,)).fetchall()
    if not games:
        return None
    roles = Counter()
    played = Counter()
    for champ, position, n, _ in games:
        played[champ] += n
        if position:
            roles[position] += n
    if not roles:
        return None
    role = roles.most_common(1)[0][0]
    best = sorted(((shrunk(w, n), c) for c, p, n, w in games if p == role and n >= 8), reverse=True)[:2]
    best_champs = [gd.champion(c) for _, c in best]
    best_tags = {t for c in best_champs for t in c.tags}
    picks = []
    for r in tier_rows:
        if r["position"] != role or r["tier"] not in TIER_BONUS or played[r["champion_id"]] >= 3:
            continue
        champ = gd.champion(r["champion_id"])
        shared = [t for t in champ.tags if t in best_tags]
        like = next((b for b in best_champs if set(b.tags) & set(champ.tags)), None)
        score = TIER_BONUS[r["tier"]] + len(shared) + r["win_rate"] / 100
        reason = (f"plays like your {like.name} ({', '.join(sorted(set(like.tags) & set(champ.tags))).lower()})"
                  if like else f"{r['win_rate']:.0f}% win rate in {POSITIONS.get(role, role).lower()}")
        picks.append((score, {"name": champ.name, "slug": champ.slug, "icon": gd.champion_icon(champ.id),
                              "tier": r["tier"], "win_rate": r["win_rate"], "games": r["games"],
                              "reason": reason, "played": played[r["champion_id"]]}))
    picks.sort(key=lambda p: -p[0])
    return {"role": POSITIONS.get(role, role), "position": role,
            "best": [c.name for c in best_champs], "picks": [p for _, p in picks[:limit]]}
