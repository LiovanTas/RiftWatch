"""Ask the coach about one minute of the High Elo Comparison.

The evidence is that minute as the tab shows it (web.mapview): the game state, the player's
own state, what they did and how often high-elo players did the same in similar spots, the
most common choice, the facts that fit it, and what actually happened over the next minutes.
The model writes a few sentences and one tip; every number it uses must be in the evidence
(the coach's grounding rule), with one retry before numbers that still fail are removed.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg
from pydantic import BaseModel, Field

from riftwatch.coach.grounding import numbers, supported

PROMPT_VERSION = "m1"


class MomentNote(BaseModel):
    explanation: str = Field(description="At most three short sentences, under 70 words.")
    tip: str = Field(description="One concrete tip for the next time this comes up, under 20 words.")
    evidence_ids: list[str] = Field(description="Ids of the evidence items used.")


SYSTEM_PROMPT = """You are a League of Legends coach explaining one minute of a player's game. \
You get evidence items describing that minute: the game state, the player's own state, what \
they did, how often Grandmaster and Challenger players in similar spots made each choice, the \
facts that fit the most common choice, and what actually happened over the next minutes.

Write:
- explanation: at most three short sentences (under 70 words). Say why most high-elo players \
chose differently here, tied to the facts of this moment, and what the player's own choice \
risked or gained, using what happened next where it shows that.
- tip: one concrete thing to do next time this situation comes up (under 20 words).

Rules: second person ("you"). The shares are how often top players chose each option in \
similar spots, not proof it would have worked, so never claim certainty. Every number you \
write must appear in the evidence. No markdown."""


def evidence(minute: dict[str, Any], game: dict[str, Any]) -> list[dict[str, str]]:
    """Evidence items for one minute of map_state() output."""
    a, st = minute["advice"], minute["state"]
    lines = [f"Minute {minute['minute']} of the game; you were playing {game['champion']} "
             f"({game['role'].lower()})" + (f" against {st['opp']}." if st.get("opp") else ".")]
    dragon = ("up" if st["dragon_in"] <= 0 else f"spawning in {st['dragon_in']} minutes")
    state = (f"Game state: your team was {abs(st['lead']):,} gold {'ahead' if st['lead'] >= 0 else 'behind'}; "
             f"dragons {st['dragons'][0]} for your team and {st['dragons'][1]} for the enemy; dragon {dragon}")
    if minute["minute"] >= 19:
        state += "; Baron " + ("up" if st["baron_in"] <= 0 else f"spawning in {st['baron_in']} minutes")
    lines.append(state + ".")
    if st.get("dead"):
        lines.append("You were dead at the start of this minute.")
    else:
        you = (f"You: level {st['level']}, {st['hp']}% health, {st['unspent']:,} gold unspent")
        if st.get("cs_diff") is not None and st.get("opp"):
            you += f", {st['cs_diff']:+d} CS against {st['opp']}"
        lines.append(you + ".")
    allies, enemies = st["allies_near"], st["enemies_near"]
    lines.append(f"On the replay, {allies} {'ally' if allies == 1 else 'allies'} and {enemies} "
                 f"{'enemy' if enemies == 1 else 'enemies'} were within 2,500 units of you.")
    lines.append(f"Your choice this minute: you {a['did']}. "
                 + (f"{a['did_share']}% of high-elo players in similar spots did the same." if a.get("did_share") is not None else ""))
    others = ", ".join(f"{o['label'].lower()} {o['share']}%" for o in a["options"])
    lines.append(f"Most common high-elo choice: {a['best'].lower()} ({a['best_share']}%). All options: {others}.")
    if a.get("why"):
        lines.append("Facts that fit: " + " ".join(a["why"]))
    if a.get("next"):
        lines.append("What happened next: " + "; ".join(a["next"]) + ".")
    lc = a.get("lead_change")
    if lc:
        lines.append(f"Your team's gold lead went from {lc['from']:+,} to {lc['to']:+,} over the next "
                     f"{lc['minutes']} minutes.")
    return [{"id": f"E{i}", "text": t} for i, t in enumerate(lines, 1)]


def _bad(output: MomentNote, items: list[dict[str, str]]) -> list[str]:
    pool = [v for e in items for v, _, _ in numbers(e["text"])]
    return [raw for text in (output.explanation, output.tip)
            for value, decimals, raw in numbers(text) if not supported(value, decimals, pool)]


def _request(coach, messages: list[dict]) -> dict[str, Any]:
    from riftwatch.coach.llm import FALLBACK_BETA

    return {"model": coach.model, "max_tokens": coach.max_tokens, "betas": [FALLBACK_BETA],
            "fallbacks": "default", "thinking": {"type": coach.thinking},
            "output_config": {"effort": coach.effort},
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "output_format": MomentNote, "messages": messages}


def write(coach, items: list[dict[str, str]]) -> tuple[MomentNote, dict]:
    from riftwatch.coach.llm import _accumulate, _echo, cost_usd

    messages: list[dict[str, Any]] = [{"role": "user", "content": "Explain this minute.\n\nEvidence:\n"
                                       + "\n".join(f"[{e['id']}] {e['text']}" for e in items)}]
    usage: dict[str, Any] = {}
    resp = coach.client.beta.messages.parse(**_request(coach, messages))
    _accumulate(usage, resp)
    out: MomentNote = resp.parsed_output
    if out is not None and _bad(out, items):
        messages += [{"role": "assistant", "content": _echo(resp.content)},
                     {"role": "user", "content": "These numbers are not in the evidence: "
                      + ", ".join(_bad(out, items)) + ". Return the answer again without them."}]
        resp = coach.client.beta.messages.parse(**_request(coach, messages))
        _accumulate(usage, resp)
        out = resp.parsed_output or out
    if out is None:
        raise RuntimeError("the model returned no answer")
    if _bad(out, items):                          # still ungrounded: keep only the tip if it's clean
        out = MomentNote(explanation="", tip=out.tip if not _bad(MomentNote(explanation="", tip=out.tip,
                                                                             evidence_ids=[]), items) else "",
                         evidence_ids=out.evidence_ids)
    usage["cost_usd"] = cost_usd(coach.model, usage)
    return out, usage


def explain(conn: psycopg.Connection, coach, game: dict[str, Any], minute: int, puuid: str) -> dict:
    """The stored note for this minute, or a new one (stored)."""
    row = conn.execute("SELECT output FROM moment_notes WHERE match_id = %s AND puuid = %s AND minute = %s "
                       "AND version = %s", (game["match_id"], puuid, minute, PROMPT_VERSION)).fetchone()
    if row:
        return {**row[0], "cached": True}
    m = game["minutes"][minute]
    if not m.get("advice"):
        raise ValueError("no high-elo comparison for this minute")
    items = evidence(m, game)
    out, usage = write(coach, items)
    conn.execute("""INSERT INTO moment_notes (match_id, puuid, minute, version, model, evidence, output, usage)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                 (game["match_id"], puuid, minute, PROMPT_VERSION, coach.model, json.dumps(items),
                  json.dumps(out.model_dump()), json.dumps(usage)))
    return {**out.model_dump(), "cached": False}
