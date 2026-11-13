"""How the site reads Swing: the Win chance tab (everyone's Swing, the moments that moved the
game and why, the player's items and choices), the profile's Swing panel (where a player gains
and loses it against their role, and the costly habits that repeat) and a champion's choices.

Every number shown is from winprob: the network's chances, Swing's shares, the Shapley split of
each moment, and the choice values (winprob.choices). Nothing here is estimated anew.
"""

from __future__ import annotations

import json
from collections import Counter
from statistics import fmean
from typing import Any

import psycopg

from riftwatch.web.gamedata import POSITIONS, GameData
from riftwatch.winprob import choices
from riftwatch.winprob import swing as swing_calc
from riftwatch.winprob.state import team_of

PART_TIPS = {"draft": "how much the two drafts favoured your team before the game began",
             "kills": "kills and assists",
             "deaths": "your deaths, and half of what your team lost while you were dead",
             "objectives": "dragons, Baron, Elder, heralds, voidgrubs, Atakhan",
             "structures": "towers and inhibitors",
             "items": "finishing items, valued by what they give against this enemy team",
             "economy": "gold and XP earned between fights, and the final push"}
SKILLS = {"Q": "Q", "W": "W", "E": "E"}


def part_keys(sw: dict) -> list[str]:
    """The parts worth showing: draft and items only when the network reads them."""
    return [k for k in swing_calc.PARTS if sw.get("reads_items") or k not in ("draft", "items")]


def parts(v: dict, keys: list[str]) -> list[dict]:
    return [{"key": k, "name": swing_calc.PART_NAMES[k], "value": v.get(k, 0.0), "tip": PART_TIPS[k]} for k in keys]


def tooltip(v: dict, keys: list[str]) -> str:
    """'Kills +4.1, Deaths -3.0, ...' for a tooltip."""
    return ", ".join(f"{swing_calc.PART_NAMES[k]} {v.get(k, 0.0):+.1f}" for k in keys)


def _team_name(side: int, mine: int | None) -> str:
    if mine:
        return "your team" if side == mine else "the enemy"
    return "blue" if side == 100 else "red"


def explain(m: dict, moments: list[dict], side: int, names: dict[int, str], norms: dict,
            mine: int | None = None) -> dict[str, Any]:
    """Why one moment moved the chance as much as it did, seen from ``side``: the Shapley split
    of the jump, its leverage, and for a death who did the damage, where, the timer, the gold
    held and what the enemy took meanwhile."""
    sign = 1 if side == 100 else -1
    why = sorted(((swing_calc.group_name(g), round(sign * v, 1)) for g, v in (m.get("why") or {}).items()),
                 key=lambda x: -abs(x[1]))
    why = [{"label": label, "value": value} for label, value in why if abs(value) >= 0.05]
    lines = []
    lev = swing_calc.leverage(m, norms)
    if lev is not None and m["swing"] >= 0.5:
        stage = ("the first 10 minutes", "minutes 10-20", "minutes 20-30", "after 30 minutes")[min(int(m["t"] // 600_000), 3)]
        kind = {"kills": "kill", "objectives": "objective", "structures": "structure", "items": "finished item"}.get(m["part"], "moment")
        lines.append(f"{lev:.1f}x the swing of a typical {kind} in {stage}"
                     + (": a big moment" if lev >= 2 else ": a smaller one than usual" if lev <= 0.5 else ""))
    death = m.get("death")
    if death and m.get("lost"):
        victim = m["lost"][0]
        who = "you" if mine is not None and victim == mine else names.get(victim, "they")
        text = f"{'You' if who == 'you' else who} died at level {death['level']}"
        if death.get("zone"):
            text += f" in {death['zone'].replace('their', 'your' if who == 'you' else 'their')}"
        foes = death.get("enemies", 0)
        text += f" to {foes} enem{'y' if foes == 1 else 'ies'}"
        if death.get("sources"):
            text += " (" + ", ".join(f"{names.get(s['pid'], '?')} {s['share']}%" for s in death["sources"][:3])
            text += f" of {death['damage']:,} damage)"
        lines.append(text + ".")
        kinds = death.get("kinds") or {}
        if kinds:
            main = max(kinds, key=kinds.get)
            if kinds[main] >= 60:
                lines.append(f"{kinds[main]}% of it was {main} damage.")
        out = f"{'You were' if who == 'you' else who + ' was'} out for {death['timer']} s"
        if death.get("held", 0) >= 300:
            out += f", holding about {death['held']:,} gold"
        if death.get("allies_dead"):
            out += f"; {death['allies_dead']} teammate{'s were' if death['allies_dead'] > 1 else ' was'} already dead"
        lines.append(out + ".")
        lost = [moments[j] for j in death.get("while_dead") or [] if j < len(moments)]
        if lost:
            team = "your team" if mine is not None and victim == mine else "their team"
            lines.append(f"While {'you were' if who == 'you' else 'they were'} dead, the enemy took "
                         + ", ".join(f"{x['text'].split(' took ')[-1].split(' destroyed ')[-1]} ({x['clock']}, -{x['swing']:.1f})"
                                     for x in lost)
                         + f": {death['while_dead_cost']:.1f} points off {team}'s chance.")
    if m.get("dead"):
        down = ", ".join("you" if mine is not None and p == mine else names.get(p, "?") for p in m["dead"])
        lines.append(f"Taken while {down} {'were' if len(m['dead']) > 1 or down == 'you' else 'was'} dead.")
    return {"why": why, "lines": lines, "leverage": lev}


def swing_view(conn: psycopg.Connection, sw: dict, players: list[dict], gd: GameData, me: dict | None,
               model=None, norms: dict | None = None) -> dict:
    """The Win chance tab: the chart, everyone's Swing by team, the moments that moved the
    game most and why, and the player's plays, deaths, items and choices."""
    from riftwatch.web.views import win_chart

    norms = norms or {}
    by_pid = {p["participant_id"]: p for p in players}
    names = {pid: gd.champion(p["champion_id"]).name for pid, p in by_pid.items()}
    my_team = me["team_id"] if me else None
    my_pid = me["participant_id"] if me else None
    keys = part_keys(sw)
    moments = sw["moments"]
    events = [m for m in moments if m["part"] != "items"]
    order = sorted(sorted(events, key=lambda m: -m["swing"])[:8], key=lambda m: m["t"])
    numbered = {(m["t"], m["text"]): i + 1 for i, m in enumerate(order)}

    def how(m: dict) -> str | None:
        """The player's part in a moment: their kill, an assist, their death, or taking part."""
        if my_pid is None:
            return None
        if my_pid in m.get("lost", []):
            return "you died"
        if my_pid in m.get("dead", []):
            return "while you were dead"
        if m["part"] in ("kills", "fights") and my_pid in m["players"]:
            return None if m.get("by") == my_pid else "assist"
        if m["part"] == "items":
            return None
        if my_pid in m["players"]:
            return "you helped"
        return None

    def moment(m: dict, amount: float | None = None) -> dict:
        side = my_team or m["team"]
        return {"n": numbered.get((m["t"], m["text"])), "clock": m["clock"], "text": m["text"], "how": how(m),
                "part": m["part"], "swing": m["swing"], "team": m["team"], "before": m["before"],
                "after": m["after"], "good": (m["team"] == my_team) if my_team else None, "amount": amount,
                **explain(m, moments, side, names, norms, my_pid)}

    peak = max([abs(v["total"]) for v in sw["players"].values()] + [1])
    teams = []
    for team_id, label in ((100, "Blue"), (200, "Red")):
        rows = []
        for p in players:
            v = sw["players"].get(str(p["participant_id"]))
            if p["team_id"] != team_id or v is None:
                continue
            rows.append({"name": p["riot_name"] or "Unknown", "champion": names[p["participant_id"]],
                         "icon": gd.champion_icon(p["champion_id"]),
                         "position": POSITIONS.get(p["position"] or "", ""),
                         "total": v["total"], "bar": round(50 * abs(v["total"]) / peak),
                         "parts": parts(v, keys), "spark": spark(v.get("line") or [], 80, 20),
                         "me": me is not None and p["participant_id"] == my_pid})
        rows.sort(key=lambda r: -r["total"])
        if rows:
            teams.append({"id": team_id, "label": label, "win": sw["winner"] == team_id, "players": rows})
    mine = None
    if me is not None and str(my_pid) in sw["players"]:
        v = sw["players"][str(my_pid)]
        everyone = sorted((x["total"] for x in sw["players"].values()), reverse=True)
        mine = {"total": v["total"], "rank": everyone.index(v["total"]) + 1, "parts": parts(v, keys),
                "best": [moment(moments[b["moment"]], b["swing"]) for b in v.get("best", [])],
                "worst": [moment(moments[b["moment"]], b["swing"]) for b in v.get("worst", [])],
                "strongest": swing_calc.PART_NAMES[max(keys, key=lambda k: v.get(k, 0))],
                "weakest": swing_calc.PART_NAMES[min(keys, key=lambda k: v.get(k, 0))],
                "spark": spark(v.get("line") or []),
                "deaths": _deaths_summary(moments, my_pid),
                "items": item_view(conn, model, gd, sw, me, players) if sw.get("reads_items") else [],
                "choices": choice_view(conn, model, gd, me, players),
                "reliability": norms.get("_choices") or {},
                "opponent": _opponent(sw, me, players, names, keys)}
    marks = [{"t": m["t"], "after": m["after"], "n": i + 1} for i, m in enumerate(order)]
    values = [v for _, v in sw["series"]]
    return {"chart": win_chart(sw["series"], sw["end_s"], marks),
            "data": json.dumps({"series": sw["series"], "end": sw["end_s"], "team": my_team}),
            "teams": teams, "moments": [moment(m) for m in order], "me": mine,
            "start": sw.get("start", 50.0), "reads_items": bool(sw.get("reads_items")),
            "my_start": (sw.get("start", 50.0) if my_team != 200 else 100 - sw.get("start", 50.0)) if my_team else None,
            "peak_blue": max(values), "low_blue": min(values)}


def _opponent(sw: dict, me: dict, players: list[dict], names: dict[int, str], keys: list[str]) -> dict | None:
    """The lane opponent's Swing beside the player's, part by part."""
    opp = next((p for p in players if me.get("position") and p["team_id"] != me["team_id"]
                and p["position"] == me["position"]), None)
    v = sw["players"].get(str(opp["participant_id"])) if opp else None
    if not v:
        return None
    mine = sw["players"][str(me["participant_id"])]
    return {"champion": names[opp["participant_id"]], "total": v["total"],
            "parts": [{"name": swing_calc.PART_NAMES[k], "mine": mine.get(k, 0.0), "theirs": v.get(k, 0.0),
                       "gap": round(mine.get(k, 0.0) - v.get(k, 0.0), 1)} for k in keys]}


def _deaths_summary(moments: list[dict], pid: int) -> dict | None:
    deaths = [m for m in moments if pid in m.get("lost", [])]
    if not deaths:
        return None
    lost = sum(m["swing"] for m in deaths)
    after = sum((m.get("death") or {}).get("while_dead_cost", 0) for m in deaths)
    return {"count": len(deaths), "swing": round(lost, 1), "while_dead": round(after, 1),
            "costly": sum(1 for m in deaths if (m.get("death") or {}).get("while_dead"))}


def spark(line: list[float], width: int = 220, height: int = 44) -> str | None:
    """A player's running Swing over the game, as a small inline chart."""
    if len(line) < 2:
        return None
    top = max(max(line), 1.0)
    low = min(min(line), -1.0)

    def y(v: float) -> float:
        return 3 + (height - 6) * (top - v) / (top - low)
    pts = " ".join(f"{(width - 2) * i / (len(line) - 1) + 1:.1f},{y(v):.1f}" for i, v in enumerate(line))
    return (f'<svg viewBox="0 0 {width} {height}" class="sw-spark" role="img" aria-label="Your Swing over the game">'
            f'<line x1="0" x2="{width}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>'
            f'<polyline points="{pts}" class="{"up" if line[-1] >= 0 else "down"}"/></svg>')


def item_view(conn, model, gd: GameData, sw: dict, me: dict, players: list[dict]) -> list[dict]:
    """Each finished item: what finishing it did to the chance, what the network says the
    champion's other common items would have done at that moment, and what players on the
    champion got from it at that slot beyond the game state (winprob.choices)."""
    my_pid = me["participant_id"]
    catalog = getattr(model, "catalog", None)
    enemy = [p["champion_id"] for p in players if p["team_id"] != me["team_id"]]
    ctx = choices.enemy_context(catalog, enemy)
    out = []
    for m in sw["moments"]:
        if m["part"] != "items" or m.get("by") != my_pid:
            continue
        item, slot = m["item"], m["slot"]
        alts = [{**a, "icon": gd.item_icon(a["item"])} for a in m.get("alternatives") or []]
        best_alt = max(alts, key=lambda a: a["swing"], default=None)
        row = {"clock": m["clock"], "slot": slot, "item": item, "icon": gd.item_icon(item),
               "name": (gd.item(item).name if gd.item(item) else m["text"].split(" finished ")[-1]),
               "gain": m.get("gain", 0.0), "alternatives": alts[:3],
               "network_best": best_alt if best_alt and best_alt["swing"] > m.get("gain", 0.0) + 0.1 else None,
               "why": [{"label": swing_calc.group_name(g), "value": round(v * (1 if me["team_id"] == 100 else -1), 2)}
                       for g, v in sorted((m.get("why") or {}).items(), key=lambda kv: -abs(kv[1]))
                       if abs(v) >= 0.02]}
        if slot <= 3 and model is not None:
            found_ctx = ctx
            mine = choices.one(conn, model, me["champion_id"], "item", slot, str(item), ctx)
            if mine is None or mine["games"] < choices.MIN_SHOWN:
                found_ctx, mine = "all", choices.one(conn, model, me["champion_id"], "item", slot, str(item))
            opts = choices.options(conn, model, me["champion_id"], "item", slot, found_ctx, limit=4)
            row["stats"] = {"context": choices.CONTEXTS[found_ctx], "mine": mine,
                            "options": [{**o, "name": gd.item(int(o["choice"])).name if gd.item(int(o["choice"])) else o["choice"],
                                         "icon": gd.item_icon(int(o["choice"])), "is_mine": o["choice"] == str(item)}
                                        for o in opts]}
        out.append(row)
    return out


def choice_view(conn, model, gd: GameData, me: dict, players: list[dict]) -> list[dict]:
    """The first ability maxed, the keystone and the summoner spells: what each did beyond the
    game state for players on this champion, beside the best common options."""
    if model is None:
        return []
    champ = me["champion_id"]
    picked = {"skill": choices.first_max(me.get("skill_order")),
              "keystone": str(me["keystone"]) if me.get("keystone") else None,
              "spells": "+".join(str(s) for s in sorted(me.get("spells") or [])) or None}

    def label(kind: str, choice: str) -> str:
        if kind == "skill":
            return f"{choice} first"
        if kind == "keystone":
            return gd.rune_name(int(choice))
        return " + ".join(gd.spell_name(int(s)) for s in choice.split("+"))

    out = []
    for kind in ("skill", "keystone", "spells"):
        if not picked[kind]:
            continue
        opts = choices.options(conn, model, champ, kind, limit=4)
        if not opts:
            continue
        mine = choices.one(conn, model, champ, kind, 0, picked[kind])
        out.append({"kind": choices.KIND_NAMES[kind], "choice": label(kind, picked[kind]), "mine": mine,
                    "options": [{**o, "label": label(kind, o["choice"]), "is_mine": o["choice"] == picked[kind]}
                                for o in opts]})
    return out


# -- the profile's Swing panel ----------------------------------------------------------------

def role_parts(conn, model, fresh: bool = False) -> dict[str, dict[str, float]]:
    """Average of each part of Swing by role over every stored game (cached ten minutes)."""
    from riftwatch.stats import champions as champ_stats

    def build():
        rows = conn.execute("""
            SELECT b.position, count(*), avg(ps.total), avg(ps.draft), avg(ps.kills), avg(ps.deaths),
                   avg(ps.objectives), avg(ps.structures), avg(ps.items), avg(ps.economy)
              FROM player_swing ps JOIN participant_builds b USING (match_id, participant_id)
             WHERE ps.version = %s AND NOT b.remake AND b.position <> '' GROUP BY 1""",
                            (swing_calc.version(model),)).fetchall()
        return {r[0]: {"games": r[1], "total": r[2], **dict(zip(swing_calc.PARTS, r[3:], strict=True))} for r in rows}
    return champ_stats.cached(("swing_roles_parts", swing_calc.version(model)), build, fresh)


def swing_profile(conn: psycopg.Connection, gd: GameData, puuid: str, model, norms: dict,
                  games: int = 20) -> dict | None:
    """Where a player gains and loses Swing over their recent games against players in the
    same role, and the costly habits that keep coming up in their deaths."""
    if model is None:
        return None
    mine = conn.execute("""SELECT b.match_id, b.participant_id, b.position, b.team_id, b.champion_id
                             FROM participant_builds b JOIN match_timelines t USING (match_id)
                            WHERE b.puuid = %s AND NOT b.remake AND b.queue_id IN (420, 440)
                            ORDER BY b.game_start DESC LIMIT %s""", (puuid, games)).fetchall()
    if len(mine) < 3:
        return None
    data = swing_calc.for_matches(conn, model, [r[0] for r in mine], lambda c: gd.champion(c).name)
    rows = [(r, data[r[0]]) for r in mine if r[0] in data]
    if len(rows) < 3:
        return None
    keys = part_keys(rows[0][1])
    role = Counter(r[2] for r, _ in rows if r[2]).most_common(1)
    role = role[0][0] if role else None
    typical = role_parts(conn, model).get(role or "", {})
    avg = {k: fmean(d["players"][str(r[1])].get(k, 0.0) for r, d in rows) for k in [*keys, "total"]}
    compare = [{"key": k, "name": swing_calc.PART_NAMES[k], "tip": PART_TIPS[k], "mine": round(avg[k], 1),
                "typical": round(typical[k], 1) if k in typical else None,
                "gap": round(avg[k] - typical[k], 1) if k in typical else None} for k in keys]
    patterns = _patterns(rows, (norms.get("habits") or {}).get(role or "", {}))
    worst = min((c for c in compare if c["gap"] is not None), key=lambda c: c["gap"], default=None)
    best = max((c for c in compare if c["gap"] is not None), key=lambda c: c["gap"], default=None)
    trend = [{"total": d["players"][str(r[1])]["total"], "win": d["winner"] == r[3],
              "champion": gd.champion(r[4]).name, "match_id": r[0]} for r, d in reversed(rows)]
    top = max([abs(t["total"]) for t in trend] + [1])
    for t in trend:
        t["height"] = round(50 * abs(t["total"]) / top)
    return {"games": len(rows), "role": POSITIONS.get(role or "", ""), "avg": round(avg["total"], 1), "trend": trend,
            "typical": round(typical["total"], 1) if typical else None, "parts": compare,
            "peak": max([abs(c["mine"]) for c in compare] + [abs(c["typical"] or 0) for c in compare] + [1]),
            "worst": worst if worst and worst["gap"] < -0.5 else None,
            "best": best if best and best["gap"] > 0.5 else None, "patterns": patterns}


def _patterns(rows, typical: dict) -> list[dict]:
    """Costly habits in the player's deaths (winprob.habits): how often each comes up, what
    those deaths cost their team per game, and the same for players in their role. Ordered by
    how much more they cost this player than is typical."""
    from riftwatch.winprob import habits

    found = {key: {"deaths": 0, "games": set(), "cost": 0.0, "examples": []} for key, *_ in habits.HABITS}
    total = 0
    for (match_id, pid, _, _, _), data in rows:
        for m, cost, keys in habits.deaths(data, pid):
            total += 1
            for key in keys:
                f = found[key]
                f["deaths"] += 1
                f["games"].add(match_id)
                f["cost"] += cost
                f["examples"].append((cost, match_id, m["clock"], m["text"]))
    out, used = [], set()
    for key, title, detail, _ in habits.HABITS:
        f = found[key]
        if f["deaths"] < 2:
            continue
        per_game = f["cost"] / len(rows)
        norm = typical.get(key) or {}
        examples = sorted(f["examples"], reverse=True)
        ex = next((e for e in examples if (e[1], e[2]) not in used), examples[0])
        used.add((ex[1], ex[2]))
        out.append({"key": key, "title": title, "detail": detail, "deaths": f["deaths"], "of": total,
                    "games": len(f["games"]), "cost": round(f["cost"], 1), "per_game": round(per_game, 1),
                    "share": round(100 * f["deaths"] / max(1, total)),
                    "typical": round(norm["per_game"], 1) if "per_game" in norm else None,
                    "typical_share": round(100 * norm["share"]) if "share" in norm else None,
                    "excess": round(per_game - norm["per_game"], 1) if "per_game" in norm else None,
                    "example": {"match_id": ex[1], "clock": ex[2], "text": ex[3], "cost": round(ex[0], 1)}})
    out.sort(key=lambda p: -(p["excess"] if p["excess"] is not None else p["per_game"]))
    return out[:4]


# -- a champion's choices ------------------------------------------------------------------

def champion_choices(conn, model, gd: GameData, champion_id: int) -> list[dict] | None:
    """For a champion page: the items at each slot, the first ability maxed, keystones and
    summoner spells that did best beyond the game state."""
    if model is None:
        return None
    out = []
    for slot in (1, 2, 3):
        opts = choices.options(conn, model, champion_id, "item", slot, limit=5)
        if opts:
            out.append({"title": f"{('1st', '2nd', '3rd')[slot - 1]} item", "kind": "item",
                        "options": [{**o, "label": gd.item(int(o["choice"])).name if gd.item(int(o["choice"])) else o["choice"],
                                     "icon": gd.item_icon(int(o["choice"]))} for o in opts]})
    labels = {"skill": lambda c: f"{c} first", "keystone": lambda c: gd.rune_name(int(c)),
              "spells": lambda c: " + ".join(gd.spell_name(int(s)) for s in c.split("+"))}
    icons = {"keystone": lambda c: gd.rune_icon(int(c)), "skill": lambda c: None,
             "spells": lambda c: gd.spell_icon(int(c.split("+")[0]))}
    for kind in ("skill", "keystone", "spells"):
        opts = choices.options(conn, model, champion_id, kind, limit=4)
        if opts:
            out.append({"title": choices.KIND_NAMES[kind], "kind": kind,
                        "options": [{**o, "label": labels[kind](o["choice"]), "icon": icons[kind](o["choice"])}
                                    for o in opts]})
    return out or None


def team_of_pid(pid: int) -> int:
    return team_of(pid)


# -- what the logistic layer learned, for the Swing page -----------------------------------

def context_view(weights: dict[str, float] | None) -> list[dict] | None:
    """The context layer's weights grouped and in plain words, each also as roughly how many
    percentage points it moves an even game (log-odds x 25 near 50%)."""
    if not weights:
        return None
    from riftwatch.winprob.state import CLASSES, ITEM_FLAG_NAMES, ITEM_FLAGS

    def row(key: str, label: str) -> dict | None:
        if key not in weights:
            return None
        v = weights[key]
        return {"label": label, "value": round(v, 3), "pts": round(25 * v, 1)}
    items = [row("items_diff", "One more finished item than the enemy (beyond its gold)")]
    items += [row(f"{f}_diff", f"...if it gives {ITEM_FLAG_NAMES[f]}, compared with an average item")
              for f in ITEM_FLAGS if f != "items"]
    matchups = [row("armor vs physical", "Armor item, per share of enemy damage that's physical"),
                row("magic resist vs magic", "Magic resist item, per share of enemy damage that's magic"),
                row("armor pen vs armor", "Armor penetration item, per enemy armor item"),
                row("magic pen vs magic resist", "Magic penetration item, per enemy magic resist item"),
                row("anti-heal vs lifesteal", "Anti-heal item, per enemy lifesteal item")]
    names = {"Marksman": "marksmen"}
    early = [row(f"{c.lower()}s_diff", f"One more {c.lower()} than the enemy, at the start") for c in CLASSES]
    late = [row(f"{c.lower()}s_diff x minute", f"...for every 30 minutes the game goes on ({names.get(c, c.lower() + 's')})")
            for c in CLASSES]
    groups = [("Items", items), ("Items against the enemy team", matchups),
              ("Team classes", early), ("How classes scale", late)]
    return [{"title": t, "rows": [r for r in rows if r]} for t, rows in groups if any(rows)]
