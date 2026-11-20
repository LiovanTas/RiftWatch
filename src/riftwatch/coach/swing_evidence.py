"""Swing as coaching evidence: what the win-probability model says each player's moments did.

When the win-probability model is loaded (``use``, once per process: the web app and the CLI
do it), a game's coaching gets the player's Swing and its parts, their costliest moments and
biggest plays (with the exact split of each jump among what changed, the death's details and
what the enemy took while they were down), and recent-games coaching gets their Swing rating
against their role, each part against the role's average and the death habits that cost them
more than is typical.

Every number is written here exactly as the coach may cite it (the grounding check compares
numbers literally). Without a model, or without the game's Swing, nothing is added and
coaching is exactly as before.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import psycopg

from riftwatch.coach.evidence import Evidence, EvidenceSet
from riftwatch.features.extract import GameFeatures, ParticipantFeatures

_state: dict[str, Any] = {"model": None, "models_dir": None}

WHAT_SWING_IS = ("Swing is RiftWatch's measure of how much a player changed their team's chance "
                 "to win, in percentage points of win chance; the winning team's Swing adds up "
                 "to +50 and the losing team's to -50")
PART_WORDS = {"draft": "draft", "kills": "kills and assists", "deaths": "deaths",
              "objectives": "objectives", "structures": "towers and inhibitors", "items": "finished items",
              "economy": "gold and XP between fights"}
ROLE_WORDS = {"TOP": "top laner", "JUNGLE": "jungler", "MIDDLE": "mid laner", "BOTTOM": "ADC",
              "UTILITY": "support"}
MOMENTS = 3          # costliest moments and biggest plays itemised for one game
HABITS = 3           # costly death habits itemised for recent games


def use(model, models_dir: str | Path | None) -> None:
    """Turn Swing evidence on for this process (``model`` None turns it off)."""
    _state["model"], _state["models_dir"] = model, models_dir


def active():
    return _state["model"]


def _pts(x: float) -> str:
    return f"{x:+.1f}"


def _append(evidence: EvidenceSet, kind: str, area: str, polarity: str, text: str,
            severity: float = 0.0, **data) -> None:
    evidence.items.append(Evidence(f"E{len(evidence.items) + 1}", kind, area, polarity, text, severity,
                                   {"source": "swing", **data}))


def _names(games: list[GameFeatures]) -> dict[int, str]:
    return {p.champion_id: p.champion_name for g in games for p in g.participants.values()}


def _swing_for(conn, model, games: list[GameFeatures]) -> dict[str, dict]:
    from riftwatch.winprob import swing

    names = _names(games)
    return swing.for_matches(conn, model, [g.match_id for g in games],
                             lambda c: names.get(c, str(c)))


# -- one game ------------------------------------------------------------------------------

def _why(m: dict, sign: int) -> str:
    """The exact split of a jump among what changed, from the player's team's side."""
    from riftwatch.winprob import swing

    parts = sorted(((swing.group_name(g), sign * v) for g, v in (m.get("why") or {}).items()),
                   key=lambda x: -abs(x[1]))
    parts = [(label, round(v, 1)) for label, v in parts if abs(round(v, 1)) >= 0.1]
    return ("; the jump splits into " + ", ".join(f"{label} {v:+.1f}" for label, v in parts)) if parts else ""


def _death_detail(m: dict, moments: list[dict], names: dict[int, str], mine: bool) -> str:
    d = m.get("death") or {}
    if not d:
        return ""
    victim = m["lost"][0]
    who = "You" if mine else names.get(victim, "They")
    text = f" {who} died at level {d.get('level', 1)}"
    if d.get("zone"):
        text += f" in {d['zone'].replace('their', 'your' if mine else 'their')}"
    foes = d.get("enemies", 0)
    text += f" to {foes} enem{'y' if foes == 1 else 'ies'}"
    sources = d.get("sources") or []
    if sources:
        text += (" (" + ", ".join(f"{names.get(s['pid'], 'an enemy')} dealt {s['share']}%" for s in sources[:3])
                 + f" of {d.get('damage', 0):,} damage)")
    text += f" and {'were' if mine else 'was'} out for {d.get('timer', 0)} seconds"
    if d.get("held", 0) >= 300:
        text += f", holding about {d['held']:,} unspent gold"
    if d.get("allies_dead"):
        text += f"; {d['allies_dead']} teammate{'s were' if d['allies_dead'] > 1 else ' was'} already dead"
    text += "."
    lost = [moments[j] for j in d.get("while_dead") or [] if j < len(moments)]
    if lost:
        text += (" Before they respawned, the enemy also took "
                 + ", ".join(f"{x['text'].split(' took ')[-1].split(' destroyed ')[-1]} at {x['clock']}"
                             for x in lost)
                 + f", worth {d.get('while_dead_cost', 0):.1f} more points of win chance.")
    return text


def _moment_text(m: dict, share: float, team: int, pid: int, moments: list[dict], names: dict[int, str],
                 norms: dict) -> str:
    from riftwatch.winprob import swing

    sign = 1 if team == 100 else -1
    before = m["before"] if team == 100 else 100 - m["before"]
    after = m["after"] if team == 100 else 100 - m["after"]
    if pid in m.get("lost", []):
        role = "your death"
    elif m.get("part") == "kills" and pid in m.get("players", []):
        role = "your kill" if m.get("by") == pid else "your assist"
    elif pid in m.get("dead", []):
        role = "taken while you were dead"
    elif pid in m.get("players", []):
        role = "you took part"
    else:
        role = "your team's"
    text = (f"At {m['clock']}, {m['text']} ({role}): your team's chance to win went from "
            f"{before:.0f}% to {after:.0f}%, and {share:+.1f} points of it are your Swing"
            f"{_why(m, sign)}.")
    lev = swing.leverage(m, norms)
    if lev is not None and m["swing"] >= 0.5 and (lev >= 1.5 or lev <= 0.5):
        stage = ("the first 10 minutes", "minutes 10-20", "minutes 20-30",
                 "after minute 30")[min(int(m["t"] // 600_000), 3)]
        text += f" That was {lev:.1f} times the swing of a typical moment like it in {stage}."
    if m.get("death"):
        text += _death_detail(m, moments, names, pid in m.get("lost", []))
    mine = next((f for f in m.get("farm") or [] if f["pid"] == pid), None)
    if mine and (mine["cs"] >= 1 or abs(mine["cost"]) >= 0.05):
        text += (f" {'Dying' if pid in m.get('lost', []) else 'Making this play'} cost you about {mine['cs']:.0f} CS of farm "
                 f"({mine['gold']:,} gold, {mine['xp']:,} XP) against your own farming pace, worth "
                 f"{mine.get('worth', mine['cost']):.1f} points of win chance")
        if "typical_cs" in mine:
            text += (f"; a typical {'death' if pid in m.get('lost', []) else 'play'} like it costs {mine['typical_cs']:.0f} CS, "
                     + (f"so {mine['cost']:.1f} more is taken out of your Swing for it."
                        if mine["cost"] >= 0.05 else
                        f"so you're credited {-mine['cost']:.1f} for costing less." if mine["cost"] <= -0.05
                        else "about the same."))
        else:
            text += "."
    return text


def _area(m: dict, pid: int) -> str:
    if pid in m.get("lost", []) or pid in m.get("dead", []):
        return "survival"
    return {"kills": "fighting", "objectives": "objectives", "structures": "objectives",
            "items": "economy"}.get(m.get("part"), "fighting")


def add_game_swing(conn: psycopg.Connection, evidence: EvidenceSet, game: GameFeatures,
                   p: ParticipantFeatures) -> None:
    """The player's Swing in this game, its parts, and their biggest and costliest moments."""
    from riftwatch.winprob import swing

    model = active()
    if model is None:
        return
    data = _swing_for(conn, model, [game]).get(game.match_id)
    v = (data or {}).get("players", {}).get(str(p.participant_id))
    if not data or not v:
        return
    names = {q.participant_id: q.champion_name for q in game.participants.values()}
    everyone = sorted((x["total"] for x in data["players"].values()), reverse=True)
    rank = everyone.index(v["total"]) + 1
    keys = [k for k in swing.PARTS if data.get("reads_items") or k not in ("draft", "items")]
    parts = ", ".join(f"{PART_WORDS[k]} {_pts(v.get(k, 0.0))}" for k in keys)
    polarity = "strength" if rank <= 3 and v["total"] > 0 else "weakness" if rank >= 8 and v["total"] < 0 else "neutral"
    _append(evidence, "swing", "macro", polarity,
            f"{WHAT_SWING_IS}. Your Swing in this game: {_pts(v['total'])}, {rank} of the 10 players. "
            f"By part: {parts}.", severity=abs(v["total"]) / 2, total=v["total"], rank=rank)
    norms = swing.norms(_state["models_dir"]) if _state["models_dir"] else {}
    moments = data.get("moments") or []
    for kind, picks in (("weakness", v.get("worst", [])), ("strength", v.get("best", []))):
        for b in picks[:MOMENTS]:
            if b["moment"] >= len(moments) or abs(b["swing"]) < 0.5:
                continue
            m = moments[b["moment"]]
            _append(evidence, "swing_moment", _area(m, p.participant_id), kind,
                    _moment_text(m, b["swing"], p.team_id, p.participant_id, moments, names, norms),
                    severity=abs(b["swing"]) * 3, minute=round(m["t"] / 60000, 1), share=b["swing"])


# -- recent games --------------------------------------------------------------------------

def add_recent_swing(conn: psycopg.Connection, evidence: EvidenceSet, puuid: str,
                     games: list[tuple[GameFeatures, ParticipantFeatures]]) -> None:
    """The player's Swing rating against their role, each part against the role's average,
    and the death habits costing them more than is typical."""
    from riftwatch.web import swingview
    from riftwatch.winprob import habits, swing, validate

    model = active()
    if model is None or not games:
        return
    data = _swing_for(conn, model, [g for g, _ in games])
    rows = [(g, p, data[g.match_id]) for g, p in games if g.match_id in data
            and str(p.participant_id) in data[g.match_id].get("players", {})]
    if len(rows) < 3:
        return
    role = Counter(p.role for _, p, _ in rows if p.role).most_common(1)
    role = role[0][0] if role else ""
    who = ROLE_WORDS.get(role, "player")
    models_dir = _state["models_dir"]
    validation = validate.load(models_dir) if models_dir else None
    said = []

    def intro() -> str:
        """The definition, once, at the start of the first Swing item."""
        if said:
            return ""
        said.append(True)
        return f"{WHAT_SWING_IS}. "

    rating = swingview.player_rating(conn, model, puuid, validation)
    if rating:
        lo, hi = rating["value"] - rating["sd"], rating["value"] + rating["sd"]
        polarity = "strength" if lo > 0 else "weakness" if hi < 0 else "neutral"
        _append(evidence, "swing", "macro", polarity,
                f"{intro()}Swing rating: {_pts(rating['value'])} points of win chance per game "
                f"against a typical {who}, "
                f"give or take {rating['sd']:.1f}, from {rating['games']} games (your raw average is "
                f"{_pts(rating['raw'])}; a rating is pulled towards 0 until enough games say otherwise).",
                severity=abs(rating["value"]) * 2, rating=rating["value"])

    keys = [k for k in swing.PARTS if any(d.get("reads_items") for _, _, d in rows) or k not in ("draft", "items")]
    mine = {k: fmean(d["players"][str(p.participant_id)].get(k, 0.0) for _, p, d in rows) for k in [*keys, "total"]}
    typical = swingview.role_parts(conn, model).get(role, {}) if role else {}
    if typical:
        gaps = sorted(((k, mine[k] - typical[k]) for k in keys if k in typical), key=lambda kv: kv[1])
        lines = ", ".join(f"{PART_WORDS[k]} {_pts(mine[k])} (typical {_pts(typical[k])})" for k in keys if k in typical)
        worst_k, worst_gap = gaps[0]
        best_k, best_gap = gaps[-1]
        _append(evidence, "swing", "macro", "neutral",
                f"{intro()}Swing per game over the last {len(rows)} games, by part, against the average {who}: "
                f"{lines}. Total {_pts(mine['total'])} against {_pts(typical['total'])}.", severity=1)
        if worst_gap <= -1.0:
            _append(evidence, "swing", _part_area(worst_k), "weakness",
                    f"Your biggest Swing gap to the average {who} is {PART_WORDS[worst_k]}: "
                    f"{_pts(mine[worst_k])} a game against {_pts(typical[worst_k])}, "
                    f"{abs(worst_gap):.1f} points of win chance a game worse.",
                    severity=abs(worst_gap) * 3, part=worst_k)
        if best_gap >= 1.0:
            _append(evidence, "swing", _part_area(best_k), "strength",
                    f"Your biggest Swing edge over the average {who} is {PART_WORDS[best_k]}: "
                    f"{_pts(mine[best_k])} a game against {_pts(typical[best_k])}, "
                    f"{best_gap:.1f} points of win chance a game better.",
                    severity=best_gap * 2, part=best_k)

    # Costly death habits, against the same role's typical cost.
    role_norms = ((swing.norms(models_dir) if models_dir else {}).get("habits") or {}).get(role, {})
    found = {key: [0, 0.0, set()] for key, *_ in habits.HABITS}
    total_deaths = 0
    for g, p, d in rows:
        for _, cost, keys_ in habits.deaths(d, p.participant_id):
            total_deaths += 1
            for key in keys_:
                found[key][0] += 1
                found[key][1] += cost
                found[key][2].add(g.match_id)
    ranked = []
    for key, title, detail, _ in habits.HABITS:
        n, cost, matches = found[key]
        if n < 2:
            continue
        per_game = cost / len(rows)
        typical_pg = (role_norms.get(key) or {}).get("per_game")
        excess = per_game - typical_pg if typical_pg is not None else per_game
        ranked.append((excess, key, title, detail, n, per_game, typical_pg, len(matches)))
    for excess, key, title, detail, n, per_game, typical_pg, n_games in sorted(ranked, reverse=True)[:HABITS]:
        text = (f"{title} ({detail}): {n} of your {total_deaths} deaths, in {n_games} of these "
                f"{len(rows)} games, cost your team {per_game:.1f} points of win chance a game, "
                f"counting each death's own swing and what the enemy took before you respawned")
        polarity = "neutral"
        if typical_pg is not None:
            text += f"; a typical {who} loses {typical_pg:.1f} a game this way"
            polarity = "weakness" if excess >= 1.0 else "strength" if excess <= -1.0 else "neutral"
        _append(evidence, "swing_habit", "survival", polarity, text + ".",
                severity=max(0.0, excess) * 3, habit=key)


def _part_area(part: str) -> str:
    return {"kills": "fighting", "deaths": "survival", "objectives": "objectives",
            "structures": "objectives", "items": "economy", "economy": "economy",
            "draft": "macro"}.get(part, "macro")
