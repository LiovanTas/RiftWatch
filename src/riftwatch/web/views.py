"""View models for the website's player pages: the summoner header, ranked cards, match
history rows, a game's full scoreboard, champion statistics, recently-played-with.

Everything reads ``participant_builds`` (one row per player per game, see stats.extract),
so a profile is a handful of indexed queries; only an expanded game reads its timeline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from statistics import fmean
from typing import Any
from urllib.parse import quote

import psycopg
from psycopg.rows import dict_row

from riftwatch.riot.api import QUEUE_NAMES
from riftwatch.web.gamedata import POSITIONS, GameData

QUEUE_FILTERS = {"all": None, "solo": 420, "flex": 440, "normal": 400}
QUEUE_SHORT = {420: "Ranked Solo", 440: "Ranked Flex", 400: "Normal Draft"}
SHARDS = {5001: "Health Scaling", 5002: "Armor", 5003: "Magic Resist", 5005: "Attack Speed",
          5007: "Ability Haste", 5008: "Adaptive Force", 5010: "Move Speed", 5011: "Health",
          5013: "Tenacity and Slow Resist"}
MULTIKILL = {5: "Penta Kill", 4: "Quadra Kill", 3: "Triple Kill", 2: "Double Kill"}


def path_id(name: str, tag: str) -> str:
    return quote(f"{name}-{tag}", safe="")


def ago(when: datetime) -> str:
    seconds = (datetime.now(UTC) - when).total_seconds()
    for unit, size, word in (("y", 31_536_000, "year"), ("mo", 2_592_000, "month"),
                             ("d", 86400, "day"), ("h", 3600, "hour"), ("m", 60, "minute")):
        if seconds >= size:
            n = int(seconds // size)
            return f"{n} {word}{'s' if n != 1 else ''} ago"
    return "just now"


def clock(seconds: int) -> str:
    return f"{seconds // 60}m {seconds % 60:02d}s"


def kda_ratio(k: float, d: float, a: float) -> str:
    return "Perfect" if d == 0 else f"{(k + a) / d:.2f}"


def _rows(conn: psycopg.Connection, sql: str, params: tuple) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, params).fetchall()


# -- header and ranks ---------------------------------------------------------------------

def header(conn: psycopg.Connection, gd: GameData, account: dict) -> dict:
    latest = conn.execute(
        """SELECT profile_icon, summoner_level, game_start FROM participant_builds
            WHERE puuid = %s ORDER BY game_start DESC LIMIT 1""", (account["puuid"],)).fetchone()
    updated = conn.execute("SELECT updated_at FROM accounts WHERE puuid = %s",
                           (account["puuid"],)).fetchone()
    return {"name": account["game_name"], "tag": account["tag_line"],
            "platform": account["platform"],
            "icon": gd.profile_icon(latest[0] if latest else None),
            "level": latest[1] if latest else None,
            "updated": ago(updated[0]) if updated else None}


def rank_card(snapshot: dict | None, gd: GameData) -> dict | None:
    if not snapshot:
        return None
    games = snapshot["wins"] + snapshot["losses"]
    apex = snapshot["tier"] in ("MASTER", "GRANDMASTER", "CHALLENGER")
    return {"tier": snapshot["tier"].title(), "division": "" if apex else snapshot["division"] or "",
            "lp": snapshot["lp"], "wins": snapshot["wins"], "losses": snapshot["losses"],
            "games": games, "win_rate": round(100 * snapshot["wins"] / games) if games else 0,
            "emblem": gd.tier_emblem(snapshot["tier"]), "crest": gd.tier_crest(snapshot["tier"]),
            "key": snapshot["tier"].lower()}


# -- match history ------------------------------------------------------------------------

def _scores(players: list[dict]) -> dict[int, float]:
    """A 0-10 performance score per player, standardised within the game: KDA, kill share,
    damage, gold, farm and vision per minute, deaths against. The best on the winning team is
    the MVP; the best on the losing team, the ACE."""
    if not players:
        return {}
    minutes = max(1.0, players[0]["duration_s"] / 60)

    def feats(p: dict) -> list[float]:
        kp = (p["kills"] + p["assists"]) / max(1, p["team_kills"])
        return [min(10.0, (p["kills"] + p["assists"]) / max(1, p["deaths"])), kp,
                p["damage"] / minutes, p["gold"] / minutes, p["cs"] / minutes,
                p["vision_score"] / minutes, -p["deaths"]]

    weights = [1.4, 1.2, 1.0, 0.8, 0.5, 0.5, 0.8]
    table = [feats(p) for p in players]
    cols = list(zip(*table, strict=True))
    stats = [(fmean(c), (fmean((x - fmean(c)) ** 2 for x in c)) ** 0.5 or 1.0) for c in cols]
    raw = {p["participant_id"]: sum(w * (x - m) / s for w, x, (m, s) in zip(weights, f, stats, strict=True))
           + (0.6 if p["win"] else 0.0) for p, f in zip(players, table, strict=True)}
    lo, hi = min(raw.values()), max(raw.values())
    return {pid: round(3 + 7 * (v - lo) / (hi - lo), 1) if hi > lo else 5.0
            for pid, v in raw.items()}


def _player_view(p: dict, gd: GameData, region: str) -> dict:
    champ = gd.champion(p["champion_id"])
    minutes = max(1.0, p["duration_s"] / 60)
    items = [{"id": i, "icon": gd.item_icon(i), "name": (gd.item(i).name if gd.item(i) else ""),
              "text": (gd.item(i).text if gd.item(i) else "")} for i in p["items"]]
    return {
        "participant_id": p["participant_id"], "puuid": p["puuid"],
        "name": p["riot_name"] or "Unknown", "tag": p["riot_tag"] or "",
        "url": (f"/players/{region}/{path_id(p['riot_name'], p['riot_tag'])}"
                if p["riot_name"] and p["riot_tag"] else None),
        "team_id": p["team_id"], "position": p["position"],
        "position_name": POSITIONS.get(p["position"], ""),
        "champion": champ.name, "champion_slug": champ.slug,
        "champion_icon": gd.champion_icon(p["champion_id"]), "level": p["champ_level"],
        "kills": p["kills"], "deaths": p["deaths"], "assists": p["assists"],
        "kda": kda_ratio(p["kills"], p["deaths"], p["assists"]),
        "kp": round(100 * (p["kills"] + p["assists"]) / max(1, p["team_kills"])),
        "cs": p["cs"], "cs_min": round(p["cs"] / minutes, 1), "gold": p["gold"],
        "damage": p["damage"], "damage_taken": p["damage_taken"],
        "vision": p["vision_score"], "wards_placed": p["wards_placed"],
        "wards_killed": p["wards_killed"], "control_wards": p["control_wards"],
        "items": items[:6], "trinket": items[6] if len(items) > 6 else None,
        "spells": [{"icon": gd.spell_icon(s), "name": gd.spell_name(s),
                    "text": gd.spells.get(s, {}).get("text", "")} for s in p["spells"]],
        "keystone": {"icon": gd.rune_icon(p["keystone"]), "name": gd.rune_name(p["keystone"]),
                     "text": gd.runes.get(p["keystone"] or 0, {}).get("text", "")},
        "sub_style": {"icon": gd.rune_icon(p["sub_style"]), "name": gd.rune_name(p["sub_style"])},
        "multikill": MULTIKILL.get(min(p["largest_multikill"], 5)) if p["largest_multikill"] >= 2 else None,
        "win": p["win"],
    }


def match_rows(conn: psycopg.Connection, gd: GameData, puuid: str, region: str,
               queue: str = "all", limit: int = 20, offset: int = 0) -> list[dict]:
    """The player's games, newest first, as match-history rows."""
    queue_id = QUEUE_FILTERS.get(queue)
    mine = _rows(conn, f"""
        SELECT * FROM participant_builds WHERE puuid = %s {'AND queue_id = %s' if queue_id else ''}
         ORDER BY game_start DESC LIMIT %s OFFSET %s""",
        (puuid, queue_id, limit, offset) if queue_id else (puuid, limit, offset))
    if not mine:
        return []
    everyone: dict[str, list[dict]] = defaultdict(list)
    for p in _rows(conn, "SELECT * FROM participant_builds WHERE match_id = ANY(%s) "
                         "ORDER BY participant_id", ([m["match_id"] for m in mine],)):
        everyone[p["match_id"]].append(p)
    out = []
    for m in mine:
        players = everyone[m["match_id"]]
        scores = _scores(players)
        me = _player_view(m, gd, region)
        winners = [p for p in players if p["win"]]
        losers = [p for p in players if not p["win"]]
        mvp = max(winners, key=lambda p: scores[p["participant_id"]], default=None)
        ace = max(losers, key=lambda p: scores[p["participant_id"]], default=None)
        badge = ("MVP" if mvp and mvp["participant_id"] == m["participant_id"] else
                 "ACE" if ace and ace["participant_id"] == m["participant_id"] else None)
        rank_in_game = sorted(scores.values(), reverse=True).index(scores[m["participant_id"]]) + 1
        teams = {100: [], 200: []}
        for p in players:
            champ = gd.champion(p["champion_id"])
            teams.setdefault(p["team_id"], []).append({
                "name": p["riot_name"] or "Unknown", "champion": champ.name,
                "icon": gd.champion_icon(p["champion_id"]),
                "url": (f"/players/{region}/{path_id(p['riot_name'], p['riot_tag'])}"
                        if p["riot_name"] and p["riot_tag"] else None),
                "me": p["puuid"] == puuid})
        result = "Remake" if m["remake"] else ("Victory" if m["win"] else "Defeat")
        out.append({
            **me, "match_id": m["match_id"],
            "queue": QUEUE_SHORT.get(m["queue_id"], QUEUE_NAMES.get(m["queue_id"], "Custom")),
            "when": ago(m["game_start"]), "date": m["game_start"].strftime("%b %d, %Y %H:%M UTC"),
            "duration": clock(m["duration_s"]), "result": result,
            "result_key": "remake" if m["remake"] else ("win" if m["win"] else "loss"),
            "score": scores[m["participant_id"]], "badge": badge, "rank_in_game": rank_in_game,
            "blue": teams.get(100, []), "red": teams.get(200, []),
            "patch": m["patch"], "day": day_label(m["game_start"]),
            "score_pct": round(10 * scores[m["participant_id"]]),
        })
    return out


def day_label(when: datetime) -> str:
    today = datetime.now(UTC).date()
    d = when.date()
    if d == today:
        return "Today"
    if (today - d).days == 1:
        return "Yesterday"
    return when.strftime("%a %b %d") if d.year == today.year else when.strftime("%b %d, %Y")


def day_groups(rows: list[dict], after_day: str | None = None) -> list[dict]:
    """Match rows grouped by the day they were played, with each day's record. A page that
    continues the previous page's last day doesn't repeat its heading."""
    groups: list[dict] = []
    for r in rows:
        if not groups or groups[-1]["day"] != r["day"]:
            groups.append({"day": r["day"], "rows": [], "wins": 0, "losses": 0,
                           "continued": not groups and r["day"] == after_day})
        g = groups[-1]
        g["rows"].append(r)
        g["wins"] += r["result_key"] == "win"
        g["losses"] += r["result_key"] == "loss"
    return groups


def summary(rows: list[dict]) -> dict | None:
    """The stats box over a page of match rows (remakes left out)."""
    games = [r for r in rows if r["result_key"] != "remake"]
    if not games:
        return None
    wins = sum(r["win"] for r in games)
    k = fmean(r["kills"] for r in games)
    d = fmean(r["deaths"] for r in games)
    a = fmean(r["assists"] for r in games)
    by_champ: dict[str, list[dict]] = defaultdict(list)
    for r in games:
        by_champ[r["champion"]].append(r)
    champs = []
    for name, rs in sorted(by_champ.items(), key=lambda kv: -len(kv[1]))[:3]:
        w = sum(r["win"] for r in rs)
        champs.append({"name": name, "icon": rs[0]["champion_icon"], "games": len(rs), "wins": w,
                       "losses": len(rs) - w, "win_rate": round(100 * w / len(rs)),
                       "kda": kda_ratio(sum(r["kills"] for r in rs), sum(r["deaths"] for r in rs),
                                        sum(r["assists"] for r in rs))})
    roles = Counter(r["position"] for r in games if r["position"])
    colours = {"TOP": "var(--accent)", "JUNGLE": "var(--sky)", "MIDDLE": "var(--amber)",
               "BOTTOM": "var(--violet)", "UTILITY": "var(--loss)"}
    positions = [{"key": p, "name": POSITIONS[p], "share": round(100 * roles[p] / len(games)),
                  "colour": colours[p]} for p in POSITIONS]
    main = max(positions, key=lambda p: p["share"])
    streak_kind = games[0]["result_key"]
    streak = 0
    for r in games:
        if r["result_key"] != streak_kind:
            break
        streak += 1
    return {"games": len(games), "wins": wins, "losses": len(games) - wins,
            "main_role": main if main["share"] else None,
            "streak": f"{streak}{'W' if streak_kind == 'win' else 'L'}",
            "streak_kind": streak_kind,
            "form": [r["result_key"] for r in rows],
            "cs_min": round(fmean(r["cs_min"] for r in games), 1),
            "win_rate": round(100 * wins / len(games)), "kills": round(k, 1),
            "deaths": round(d, 1), "assists": round(a, 1), "kda": kda_ratio(k, d, a),
            "kp": round(fmean(r["kp"] for r in games)), "champions": champs,
            "positions": positions}


# -- champion statistics, played with -----------------------------------------------------

def champion_stats(conn: psycopg.Connection, gd: GameData, puuid: str,
                   queue: str = "ranked") -> list[dict]:
    """Per champion over the season (since 1 January): games, win rate, average K/D/A,
    farm, damage, gold and vision per minute."""
    queues = {"ranked": (420, 440), "solo": (420,), "flex": (440,), "all": (420, 440, 400)}[queue]
    season = datetime(datetime.now(UTC).year, 1, 1, tzinfo=UTC)
    rows = _rows(conn, """
        SELECT champion_id, count(*) games, sum(win::int) wins,
               avg(kills)::float8 k, avg(deaths)::float8 d, avg(assists)::float8 a,
               (sum(cs) / (sum(duration_s) / 60.0))::float8 cs_min,
               (sum(damage) / (sum(duration_s) / 60.0))::float8 dmg_min,
               (sum(gold) / (sum(duration_s) / 60.0))::float8 gold_min,
               avg(vision_score)::float8 vision, max(largest_multikill) best_multikill,
               sum(double_kills) doubles, sum(triple_kills) triples,
               sum(quadra_kills) quadras, sum(penta_kills) pentas,
               avg((kills + assists)::float / greatest(team_kills, 1)) kp
          FROM participant_builds
         WHERE puuid = %s AND queue_id = ANY(%s) AND game_start >= %s AND NOT remake
         GROUP BY champion_id ORDER BY count(*) DESC, sum(win::int) DESC""",
        (puuid, list(queues), season))
    out = []
    for r in rows:
        champ = gd.champion(r["champion_id"])
        out.append({
            "name": champ.name, "slug": champ.slug, "icon": gd.champion_icon(r["champion_id"]),
            "games": r["games"], "wins": r["wins"], "losses": r["games"] - r["wins"],
            "win_rate": round(100 * r["wins"] / r["games"]),
            "kills": round(r["k"], 1), "deaths": round(r["d"], 1), "assists": round(r["a"], 1),
            "kda": kda_ratio(r["k"], r["d"], r["a"]),
            "kda_value": (r["k"] + r["a"]) / max(r["d"], 0.01),
            "cs_min": round(r["cs_min"] or 0, 1), "dmg_min": round(r["dmg_min"] or 0),
            "gold_min": round(r["gold_min"] or 0), "vision": round(r["vision"] or 0, 1),
            "kp": round(100 * (r["kp"] or 0)), "doubles": r["doubles"], "triples": r["triples"],
            "quadras": r["quadras"], "pentas": r["pentas"]})
    return out


def played_with(conn: psycopg.Connection, puuid: str, region: str, games: int = 60) -> list[dict]:
    """Teammates in two or more of the player's last ``games`` games."""
    rows = conn.execute("""
        WITH mine AS (SELECT match_id, team_id, win FROM participant_builds
                       WHERE puuid = %s ORDER BY game_start DESC LIMIT %s)
        SELECT b.puuid, max(b.riot_name), max(b.riot_tag), count(*), sum(m.win::int)
          FROM participant_builds b JOIN mine m ON b.match_id = m.match_id AND b.team_id = m.team_id
         WHERE b.puuid <> %s GROUP BY b.puuid HAVING count(*) >= 2
         ORDER BY count(*) DESC, sum(m.win::int) DESC LIMIT 8""", (puuid, games, puuid)).fetchall()
    return [{"name": name, "tag": tag, "games": n, "wins": w, "losses": n - w,
             "win_rate": round(100 * w / n),
             "url": f"/players/{region}/{path_id(name, tag)}" if name and tag else None}
            for _, name, tag, n, w in rows]


# -- one game in full ---------------------------------------------------------------------

OBJECTIVES = (("baron", "Baron"), ("dragon", "Dragon"), ("horde", "Voidgrubs"),
              ("riftHerald", "Herald"), ("tower", "Tower"), ("inhibitor", "Inhibitor"),
              ("atakhan", "Atakhan"))


def _gold_series(timeline: dict | None) -> list[dict]:
    """Blue minus red total gold at each minute."""
    if not timeline:
        return []
    out = []
    for i, frame in enumerate(timeline.get("info", {}).get("frames", [])):
        blue = red = 0
        for pid, pf in (frame.get("participantFrames") or {}).items():
            if int(pid) <= 5:
                blue += pf.get("totalGold", 0)
            else:
                red += pf.get("totalGold", 0)
        out.append({"minute": i, "diff": blue - red})
    return out


def gold_chart(series: list[dict], width: int = 720, height: int = 160) -> str | None:
    """Inline SVG of the gold lead: above the line blue is ahead, below red is."""
    if len(series) < 2:
        return None
    peak = max(1000, max(abs(s["diff"]) for s in series))
    pad_x, mid = 40, height / 2

    def x(i: int) -> float:
        return pad_x + (width - pad_x - 8) * i / (len(series) - 1)

    def y(v: float) -> float:
        return mid - (mid - 12) * v / peak

    pts = " ".join(f"{x(i):.1f},{y(s['diff']):.1f}" for i, s in enumerate(series))
    area = f"{x(0):.1f},{mid} {pts} {x(len(series) - 1):.1f},{mid}"
    ticks = "".join(
        f'<text x="{x(i):.1f}" y="{height - 1}" class="tick" text-anchor="middle">{s["minute"]}</text>'
        for i, s in enumerate(series) if s["minute"] % 5 == 0)
    k = round(peak / 1000, 1)
    return (f'<svg viewBox="0 0 {width} {height}" class="gold-chart" role="img" '
            f'aria-label="Team gold difference by minute">'
            f'<defs><clipPath id="above"><rect x="0" y="0" width="{width}" height="{mid}"/></clipPath>'
            f'<clipPath id="below"><rect x="0" y="{mid}" width="{width}" height="{mid}"/></clipPath></defs>'
            f'<line x1="{pad_x}" x2="{width - 8}" y1="{mid}" y2="{mid}" class="axis"/>'
            f'<polygon points="{area}" class="area-blue" clip-path="url(#above)"/>'
            f'<polygon points="{area}" class="area-red" clip-path="url(#below)"/>'
            f'<polyline points="{pts}" class="line"/>'
            f'<text x="{pad_x - 6}" y="16" class="tick" text-anchor="end">+{k}k</text>'
            f'<text x="{pad_x - 6}" y="{height - 14}" class="tick" text-anchor="end">-{k}k</text>'
            f"{ticks}</svg>")


def _rune_page(p: dict, gd: GameData) -> dict | None:
    if not p.get("perks"):
        return None
    chosen = set(p["perks"])
    styles = []
    for style_id, rows_from in ((p["primary_style"], 0), (p["sub_style"], 1)):
        style = gd.styles.get(style_id or 0)
        if not style:
            continue
        slots = style["slots"] if rows_from == 0 else style["slots"][1:]
        styles.append({"name": style["name"], "icon": gd.rune_icon(style_id),
                       "rows": [[{"icon": gd.rune_icon(r), "name": gd.rune_name(r),
                                  "text": gd.runes.get(r, {}).get("text", ""),
                                  "on": r in chosen} for r in slot] for slot in slots]})
    shards = [{"name": SHARDS.get(s, "Stat shard")} for s in p["perks"] if s in SHARDS]
    return {"styles": styles, "shards": shards}


def _build_steps(p: dict, gd: GameData) -> list[dict]:
    """Purchases grouped by the minute they were bought."""
    steps: dict[int, list[dict]] = {}
    for item_id, second in zip(p.get("purchases") or [], p.get("purchase_s") or [], strict=False):
        it = gd.item(item_id)
        if it and ("Consumable" in it.tags and it.gold < 100):
            continue                                   # potions and wards clutter it
        steps.setdefault(second // 60, []).append(
            {"icon": gd.item_icon(item_id), "name": it.name if it else str(item_id),
             "text": it.text if it else ""})
    return [{"minute": m, "items": items} for m, items in sorted(steps.items())]


def _skill_grid(order: str | None) -> list[dict]:
    order = (order or "")[:18]
    return [{"key": k, "levels": [i + 1 for i, s in enumerate(order) if s == k]} for k in "QWER"]


def match_detail(conn: psycopg.Connection, gd: GameData, match_id: str, puuid: str | None,
                 region: str) -> dict | None:
    players = _rows(conn, "SELECT * FROM participant_builds WHERE match_id = %s "
                          "ORDER BY participant_id", (match_id,))
    if not players:
        return None
    raw = conn.execute("SELECT raw->'info'->'teams' FROM matches WHERE match_id = %s",
                       (match_id,)).fetchone()
    teams_raw = {t["teamId"]: t for t in (raw[0] if raw and raw[0] else [])}
    bans = defaultdict(list)
    for team_id, champion_id in conn.execute(
            "SELECT team_id, champion_id FROM match_bans WHERE match_id = %s ORDER BY pick_turn",
            (match_id,)):
        bans[team_id].append({"icon": gd.champion_icon(champion_id),
                              "name": gd.champion(champion_id).name})
    scores = _scores(players)
    winners = [p for p in players if p["win"]]
    losers = [p for p in players if not p["win"]]
    mvp = max(winners, key=lambda p: scores[p["participant_id"]], default=None)
    ace = max(losers, key=lambda p: scores[p["participant_id"]], default=None)
    top_damage = max((p["damage"] for p in players), default=1) or 1
    top_taken = max((p["damage_taken"] for p in players), default=1) or 1
    teams = []
    for team_id, label in ((100, "Blue team"), (200, "Red team")):
        members = [p for p in players if p["team_id"] == team_id]
        if not members:
            continue
        objectives = (teams_raw.get(team_id) or {}).get("objectives") or {}
        rows = []
        for p in members:
            v = _player_view(p, gd, region)
            v["score"] = scores[p["participant_id"]]
            v["badge"] = ("MVP" if mvp and mvp is p else "ACE" if ace and ace is p else None)
            v["damage_pct"] = round(100 * p["damage"] / top_damage)
            v["taken_pct"] = round(100 * p["damage_taken"] / top_taken)
            v["me"] = p["puuid"] == puuid
            rows.append(v)
        teams.append({
            "id": team_id, "label": label, "win": members[0]["win"],
            "kills": sum(p["kills"] for p in members),
            "deaths": sum(p["deaths"] for p in members),
            "assists": sum(p["assists"] for p in members),
            "gold": sum(p["gold"] for p in members),
            "objectives": [{"key": k, "name": name, "kills": (objectives.get(k) or {}).get("kills", 0)}
                           for k, name in OBJECTIVES if k in objectives or k in ("baron", "dragon", "tower")],
            "bans": bans.get(team_id, []), "players": rows})
    me = next((p for p in players if p["puuid"] == puuid), None)
    timeline = conn.execute("SELECT raw FROM match_timelines WHERE match_id = %s",
                            (match_id,)).fetchone()
    gold = _gold_series(timeline[0] if timeline else None)
    return {
        "match_id": match_id, "duration": clock(players[0]["duration_s"]),
        "queue": QUEUE_NAMES.get(players[0]["queue_id"], "Custom"), "patch": players[0]["patch"],
        "teams": teams, "gold_chart": gold_chart(gold),
        "gold_end": gold[-1]["diff"] if gold else None,
        "me": ({"build": _build_steps(me, gd), "skills": _skill_grid(me["skill_order"]),
                "skill_order": (me["skill_order"] or "")[:18], "runes": _rune_page(me, gd),
                "champion": gd.champion(me["champion_id"]).name}
               if me else None),
    }
