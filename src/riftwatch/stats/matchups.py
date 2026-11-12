"""Why a lane matchup is good or bad for a champion.

The evidence has two halves:
  * measured lane numbers from ranked games where the two champions met in the same role
    (participant_builds for results, participant_minute_features for the lane: gold, CS and
    XP against the opponent at 3, 6, 10 and 15 minutes, early deaths, results by game length);
  * both kits from Data Dragon: base stats, attack range, each ability's cooldown, range and
    description.

The LLM turns that into a verdict, three or four reasons that connect the kits to how trades
go in lane, and a few tips. The coach's grounding rule applies: every number it writes must
appear in the evidence it cites, and a point that breaks the rule after one retry is dropped.
Answers are stored per champion pair, role and game version, so each is written once.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg
from pydantic import BaseModel, Field

from riftwatch.coach.grounding import numbers, supported
from riftwatch.web.gamedata import POSITIONS, GameData, plain

QUEUES = (420, 440)
SMALL_SAMPLE = 30


class MatchupPoint(BaseModel):
    title: str = Field(description="A short claim, e.g. 'Renekton wins the level 3 all-in'.")
    explanation: str = Field(description="Two or three sentences tying the kits to how trades go.")
    evidence_ids: list[str] = Field(description="Ids of the evidence items this point relies on.")


class MatchupOutput(BaseModel):
    verdict: str = Field(description="One sentence: who the matchup favours, how much, and why.")
    points: list[MatchupPoint]
    tips: list[str] = Field(description="Two or three concrete things to do in this lane.")


SYSTEM_PROMPT = """You explain League of Legends lane matchups for a stats website. You get \
evidence items: measured lane numbers from ranked games between two champions in one role, and \
each champion's kit from the game data (base stats, attack range, and each ability's cooldown, \
range and description).

Write from the FIRST champion's point of view.
- verdict: one sentence on whether the matchup favours them, how strongly, and the main reason.
- points: three or four reasons. Each connects the two kits to how lane trades actually play \
out: range and who can poke, cooldown windows to punish, short trades against all-ins, crowd \
control that sets up kills or ganks, mobility to engage or escape, sustain, and when each side \
spikes (levels, scaling). Use the lane numbers to show each reason is real: who is ahead at 10 \
minutes, who dies early, how the result changes with game length.
- tips: two or three concrete things the first champion's player should do in this lane.

Rules:
- Every number you write must appear in an evidence item you cite (rounding is fine). The \
verdict and tips cite nothing, so their numbers must appear somewhere in the evidence.
- Only mention abilities, stats and effects that are in the evidence. No items, runes or patch \
changes.
- If the games number fewer than 30, say the record is a small sample and lean on the kits.
- Plain sentences, no markdown, second person ("you") for the first champion's player."""


# -- evidence ------------------------------------------------------------------------------

def _lane(conn: psycopg.Connection, a: int, b: int | None, role: str) -> dict[str, Any]:
    """Results and lane numbers for champion ``a`` in ``role`` against ``b`` (None: anyone)."""
    params: dict[str, Any] = {"a": a, "b": b, "role": role, "queues": list(QUEUES)}
    opp = "AND o.champion_id = %(b)s" if b is not None else ""
    games = f"""
        SELECT x.match_id, x.participant_id me, o.participant_id opp, x.win, x.duration_s
          FROM participant_builds x
          JOIN participant_builds o ON o.match_id = x.match_id AND o.team_id <> x.team_id
                                   AND o.position = x.position
         WHERE x.champion_id = %(a)s AND x.position = %(role)s {opp}
           AND NOT x.remake AND x.queue_id = ANY(%(queues)s)"""
    n, wins, short_n, short_w, long_n, long_w = conn.execute(f"""
        WITH g AS ({games})
        SELECT count(*), coalesce(sum(win::int), 0),
               count(*) FILTER (WHERE duration_s < 1500), coalesce(sum(win::int) FILTER (WHERE duration_s < 1500), 0),
               count(*) FILTER (WHERE duration_s > 1920), coalesce(sum(win::int) FILTER (WHERE duration_s > 1920), 0)
          FROM g""", params).fetchone()
    minutes = conn.execute(f"""
        WITH g AS ({games})
        SELECT f.minute, count(*), avg(f.gold_diff)::float8, avg(f.cs_diff)::float8, avg(f.xp_diff)::float8,
               avg((f.gold_diff > 0)::int)::float8, avg((f.deaths > 0)::int)::float8,
               avg((fo.deaths > 0)::int)::float8,
               avg(g.win::int) FILTER (WHERE f.gold_diff > 0)::float8,
               avg(g.win::int) FILTER (WHERE f.gold_diff < 0)::float8
          FROM g JOIN participant_minute_features f ON f.match_id = g.match_id AND f.participant_id = g.me
                 JOIN participant_minute_features fo ON fo.match_id = g.match_id AND fo.participant_id = g.opp
                                                    AND fo.minute = f.minute
         WHERE f.minute IN (3, 6, 10, 15) GROUP BY f.minute ORDER BY f.minute""", params).fetchall()
    return {"games": n, "wins": wins, "short": (short_n, short_w), "long": (long_n, long_w),
            "minutes": {m: {"n": c, "gold": g, "cs": cs, "xp": xp, "ahead": ahead, "died": died,
                            "opp_died": odied, "win_ahead": wa, "win_behind": wb}
                        for m, c, g, cs, xp, ahead, died, odied, wa, wb in minutes}}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{round(100 * x)}%"


def _signed(x: float) -> str:
    return f"{round(x):+d}"


CC_WORDS = ("stun", "root", "knock", "airborne", "slow", "charm", "fear", "flee", "taunt",
            "silence", "suppress", "sleep", "grounded", "blind", "pull")
MOBILITY_WORDS = ("dash", "blink", "leap", "jump", "lunge", "teleport", "untargetable")
SUSTAIN_WORDS = ("heal", "restore", "shield", "life steal", "omnivamp", "regenerat")


def _short(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def kit_lines(gd: GameData, champion_id: int) -> list[str]:
    champ = gd.champion(champion_id)
    detail = gd.champion_detail(champ) if champ.key else None
    if not detail:
        return [f"{champ.name}: no kit data available."]
    st = detail.get("stats", {})
    reach = st.get("attackrange", 0)
    words = " ".join(plain(s.get("description")) for s in detail.get("spells", [])).lower()
    words += " " + plain(detail.get("passive", {}).get("description")).lower()
    cc = sorted({w for w in CC_WORDS if w in words})
    moves = sorted({w for w in MOBILITY_WORDS if w in words})
    sustain = sorted({w for w in SUSTAIN_WORDS if w in words})
    lines = [
        f"{champ.name} base stats: {'melee' if reach <= 250 else 'ranged'}, attack range "
        f"{reach:g}, move speed {st.get('movespeed', 0):g}, health {st.get('hp', 0):g} "
        f"(+{st.get('hpperlevel', 0):g} per level), armor {st.get('armor', 0):g}, attack damage "
        f"{st.get('attackdamage', 0):g}; resource {detail.get('partype') or 'none'}; class "
        f"{', '.join(detail.get('tags', [])) or 'unknown'}.",
        f"{champ.name} ability text mentions: crowd control ({', '.join(cc) or 'none found'}); "
        f"mobility ({', '.join(moves) or 'none found'}); sustain ({', '.join(sustain) or 'none found'}).",
        f"{champ.name} passive {detail.get('passive', {}).get('name', '')}: "
        f"{_short(plain(detail.get('passive', {}).get('description')))}",
    ]
    for key, sp in zip("QWER", detail.get("spells", []), strict=False):
        lines.append(f"{champ.name} {key} {sp.get('name', '')}: cooldown {sp.get('cooldownBurn', '?')} s "
                     f"(ranks 1 up), range {sp.get('rangeBurn', '?')}. {_short(plain(sp.get('description')))}")
    return lines


def evidence(conn: psycopg.Connection, gd: GameData, a: int, b: int, role: str) -> list[dict[str, str]]:
    """Evidence items (id, text) for champion ``a`` against ``b`` in ``role``."""
    an, bn = gd.champion(a).name, gd.champion(b).name
    where = POSITIONS.get(role, role.title()).lower()
    vs, overall = _lane(conn, a, b, role), _lane(conn, a, None, role)
    items: list[str] = []
    if vs["games"]:
        wr = vs["wins"] / vs["games"]
        base = overall["wins"] / max(1, overall["games"])
        items.append(f"Record: {an} {where} against {bn}: {vs['games']} ranked games, {an} won "
                     f"{vs['wins']} ({_pct(wr)}). Against every opponent {an} {where} wins "
                     f"{_pct(base)} of {overall['games']} games, so this matchup is "
                     f"{_signed(100 * (wr - base))} points from usual.")
        sn, sw = vs["short"]
        ln, lw = vs["long"]
        if sn >= 3 or ln >= 3:
            items.append(f"Game length: {an} won {sw} of {sn} games under 25 minutes "
                         f"({_pct(sw / sn) if sn else 'n/a'}) and {lw} of {ln} games over 32 minutes "
                         f"({_pct(lw / ln) if ln else 'n/a'}).")
    for minute in (3, 6, 10, 15):
        m, o = vs["minutes"].get(minute), overall["minutes"].get(minute)
        if not m:
            continue
        text = (f"At {minute} minutes ({m['n']} games): {an} averaged {_signed(m['gold'] or 0)} "
                f"gold, {_signed(m['cs'] or 0)} CS and {_signed(m['xp'] or 0)} XP against {bn}, "
                f"and was ahead in gold in {_pct(m['ahead'])} of games")
        if o:
            text += f" (against every opponent: {_signed(o['gold'] or 0)} gold, ahead in {_pct(o['ahead'])})"
        items.append(text + ".")
        if minute in (6, 10):
            items.append(f"Early deaths by {minute} minutes: {an} had died at least once in "
                         f"{_pct(m['died'])} of games and {bn} in {_pct(m['opp_died'])}"
                         + (f" ({an} against every opponent: {_pct(o['died'])})" if o else "") + ".")
        if minute == 15 and m["win_ahead"] is not None and m["win_behind"] is not None:
            items.append(f"Converting the lane: {an} won {_pct(m['win_ahead'])} of games when ahead "
                         f"in gold at 15 minutes against {bn}, and {_pct(m['win_behind'])} when behind.")
    items += kit_lines(gd, a) + kit_lines(gd, b)
    items.append("Lane power spikes: levels 2 and 3 add a second and third basic ability, and "
                 "level 6 unlocks the ultimate (R).")
    return [{"id": f"E{i}", "text": t} for i, t in enumerate(items, 1)]


# -- grounding -----------------------------------------------------------------------------

def _bad_numbers(text: str, pool: list[float]) -> list[str]:
    return [raw for value, decimals, raw in numbers(text) if not supported(value, decimals, pool)]


def check(output: MatchupOutput, items: list[dict[str, str]]) -> list[str]:
    """Grounding problems: numbers not in the evidence, citations that don't exist."""
    by_id = {e["id"]: e["text"] for e in items}
    every = [v for e in items for v, _, _ in numbers(e["text"])]
    problems = []
    for raw in _bad_numbers(output.verdict, every):
        problems.append(f"verdict: {raw} is not in the evidence")
    for i, tip in enumerate(output.tips, 1):
        for raw in _bad_numbers(tip, every):
            problems.append(f"tip {i}: {raw} is not in the evidence")
    for i, p in enumerate(output.points, 1):
        missing = [e for e in p.evidence_ids if e not in by_id]
        if not p.evidence_ids or missing:
            problems.append(f"point {i}: cites {missing or 'nothing'}; cite existing evidence ids")
            continue
        cited = [v for e in p.evidence_ids for v, _, _ in numbers(by_id[e])]
        for raw in _bad_numbers(p.title + " " + p.explanation, cited):
            problems.append(f"point {i}: {raw} is not in the evidence it cites")
    return problems


def _point_ok(p: MatchupPoint, items: list[dict[str, str]]) -> bool:
    return not check(MatchupOutput(verdict="", points=[p], tips=[]), items)


# -- writing and caching -------------------------------------------------------------------

def _request(coach, messages: list[dict]) -> dict[str, Any]:
    from riftwatch.coach.llm import FALLBACK_BETA

    return {"model": coach.model, "max_tokens": coach.max_tokens, "betas": [FALLBACK_BETA],
            "fallbacks": "default", "thinking": {"type": coach.thinking},
            "output_config": {"effort": coach.effort},
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "output_format": MatchupOutput, "messages": messages}


def write(coach, items: list[dict[str, str]], a_name: str, b_name: str, role: str) -> tuple[MatchupOutput, list[dict], dict]:
    """Ask the model, retry once on grounding problems, then drop any point still failing."""
    from riftwatch.coach.llm import _accumulate, _echo, cost_usd

    prompt = (f"Explain the {role.lower()} matchup {a_name} (first champion) against {b_name}.\n\n"
              "Evidence:\n" + "\n".join(f"[{e['id']}] {e['text']}" for e in items))
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    usage: dict[str, Any] = {}
    resp = coach.client.beta.messages.parse(**_request(coach, messages))
    _accumulate(usage, resp)
    output: MatchupOutput = resp.parsed_output
    problems = check(output, items) if output else ["no answer"]
    if problems and output is not None:
        messages.append({"role": "assistant", "content": _echo(resp.content)})
        messages.append({"role": "user", "content": "Your answer failed the grounding check:\n"
                         + "\n".join(f"- {p}" for p in problems)
                         + "\n\nReturn the full answer again with these fixed. Remove a number "
                           "rather than guess one."})
        resp = coach.client.beta.messages.parse(**_request(coach, messages))
        _accumulate(usage, resp)
        output = resp.parsed_output or output
    if output is None:
        raise RuntimeError("the model returned no answer")
    every = [v for e in items for v, _, _ in numbers(e["text"])]
    kept = [p for p in output.points if _point_ok(p, items)]
    dropped = [p.model_dump() for p in output.points if not _point_ok(p, items)]
    tips = [t for t in output.tips if not _bad_numbers(t, every)]
    verdict = output.verdict if not _bad_numbers(output.verdict, every) else ""
    usage["cost_usd"] = cost_usd(coach.model, usage)
    return MatchupOutput(verdict=verdict, points=kept, tips=tips), dropped, usage


def cached(conn: psycopg.Connection, a: int, b: int, role: str, version: str) -> dict | None:
    row = conn.execute("SELECT output, evidence, model, created_at FROM matchup_notes WHERE "
                       "champion_id = %s AND opponent_id = %s AND position = %s AND version = %s",
                       (a, b, role, version)).fetchone()
    if row is None:
        return None
    return {"output": row[0], "evidence": row[1], "model": row[2], "written": row[3].isoformat(),
            "cached": True}


def explain(conn: psycopg.Connection, gd: GameData, coach, a: int, b: int, role: str) -> dict:
    """The stored explanation, or a new one written now (and stored)."""
    hit = cached(conn, a, b, role, gd.version)
    if hit:
        return hit
    items = evidence(conn, gd, a, b, role)
    output, dropped, usage = write(coach, items, gd.champion(a).name, gd.champion(b).name,
                                   POSITIONS.get(role, role.title()))
    conn.execute(
        """INSERT INTO matchup_notes (champion_id, opponent_id, position, version, model, evidence,
                                      output, dropped, usage)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
        (a, b, role, gd.version, coach.model, json.dumps(items), json.dumps(output.model_dump()),
         json.dumps(dropped), json.dumps(usage)))
    return {"output": output.model_dump(), "evidence": items, "model": coach.model,
            "written": None, "cached": False}
