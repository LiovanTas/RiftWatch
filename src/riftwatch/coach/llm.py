"""The LLM coach: Claude turns evidence into advice, then the grounding check audits it.

Flow per request:
  1. Claude gets the numbered evidence and nothing else about the game, and must answer in
     the CoachOutput JSON schema (structured outputs), citing evidence ids.
  2. :func:`grounding.validate` checks every id and every number.
  3. If anything fails, Claude sees the exact violations once and answers again.
  4. Points that still fail are dropped, never shown.

Token economy (default model: Claude Sonnet 5.5):
  * The system prompt is byte-identical for every player and game, and long enough to
    clear the model's 512-token caching minimum (853 tokens; 1,513 with the output
    schema), so after the first request it is read from cache at a tenth of the input
    price. Measured across different games: 1,513 cache-read tokens on every call.
  * There is deliberately no second, conversation-level breakpoint. It would write each
    game's evidence to cache at 1.25x the input price, which only pays off if a grounding
    retry re-reads it -- break-even is a ~28% retry rate, and measured retries were 0 of
    10. (IncidentPilot's agent does want one: it re-sends a growing history every turn.)
  * Answers are short by instruction (at most five points, one or two sentences each) and
    advice carries no numbers, which removes the most common cause of a retry.
  * Thinking mode and effort are configurable; see the measurements in the README.
  * Finished answers are cached in Postgres by the caller, so a repeat view costs nothing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from riftwatch.coach.evidence import EvidenceSet
from riftwatch.coach.grounding import (
    CoachOutput,
    CoachPoint,
    Violation,
    validate,
    without_violations,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_POINTS = 5  # the prompt asks for at most five; enforced here because "high" effort wrote six

# USD per million tokens: input, output, cache read, cache write (5-minute TTL).
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
}

SYSTEM_PROMPT = """\
You are a League of Legends coach reviewing a player's ranked and normal draft games.

You receive a numbered list of evidence items. They are the only facts you know about the
player and their games. Each comes from the player's match timeline and is compared with
ranked solo/duo players of the same rank tier and role (the comparison group named in the
first item), whatever mode the game itself was.

How to read the evidence:
- "better than N% of comparable players" ranks the player within the comparison group, with
  direction already accounted for: for deaths, "better than 10%" means more deaths than 90%
  of those players. Higher is always better. "Bottom quarter" / "top quarter" mean worse
  than 75% / better than 75% of them.
- "median" is the comparison group's typical value; "below"/"above" give the gap to it.
  "Master+ median" is the typical value among Master, Grandmaster and Challenger players in
  the same role -- a reference for where high elo sits, not the player's comparison group.
- "lane opponent" means the enemy player in the same role. For a jungler that is the enemy
  jungler; for a support, the enemy support.
- Per-minute patterns ("from minute A to minute B ...") mean the gap persisted the whole
  stretch, which matters more than a single moment.
- Death items describe one death: time, map area relative to the player's team, who got
  the kill, how many enemies helped, and the gold gap to the lane opponent at the time.
- "High-elo comparison" items compare the player's move at a minute with what
  Grandmaster/Challenger players in the same role did in similar situations, and with what
  followed each choice in those games. They describe how high-elo games tended to go, not
  certainties: say "in similar high-elo situations, most players..." or "that choice was
  followed by...", never "you would have" or "you should have".
- "Live recording" items come from the player's own health and gold sampled every second
  during the game. A large health loss in a short window is the player's side of a trade
  they lost; whether a recall or a death followed shows what it cost them.
- Measures: CS is minions plus jungle monsters killed. Kill participation is the share of
  the team's kills the player had a kill or assist in. Damage share is the player's share
  of the team's damage to champions. Vision score is Riot's measure of wards placed,
  cleared and the value of the vision they gave. Objective participation is the share of
  the team's dragons, grubs, heralds and barons the player helped take.

Rules:
- Every point cites, in evidence_ids, the ids of the items it is based on.
- Every number in title or explanation must appear in an item that point cites. Reuse
  numbers exactly as written; never compute new ones (differences, totals, averages).
- advice contains no numbers at all, not even minute marks or counts. Describe the habit
  to build in words.
- Never state facts the evidence doesn't contain (items, runes, matchups, teammates'
  play). General League knowledge belongs in advice, phrased as advice.
- At most five points, most important first: sustained patterns and large gaps before
  small ones, weaknesses before strengths, and include one strength if the evidence has
  any. Fewer points if the evidence is thin.
- title: a short phrase. explanation: one or two sentences. advice: one or two sentences
  of concrete things to do next game.
- headline: one sentence on the overall picture, with no numbers.
- Plain, direct language. Address the player as "you".
"""


class CoachError(RuntimeError):
    pass


@dataclass
class CoachRun:
    output: CoachOutput
    dropped: list[CoachPoint]
    first_violations: list[Violation]
    final_violations: list[Violation]
    attempts: int
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


def cost_usd(model: str, usage: dict[str, Any]) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    inp, out, read, write = price
    return round((usage.get("input_tokens", 0) * inp + usage.get("output_tokens", 0) * out
                  + usage.get("cache_read_input_tokens", 0) * read
                  + usage.get("cache_creation_input_tokens", 0) * write) / 1e6, 5)


def _accumulate(usage: dict[str, Any], resp: Any) -> None:
    u = resp.usage
    for name in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                 "cache_creation_input_tokens"):
        usage[name] = usage.get(name, 0) + (getattr(u, name, 0) or 0)
    # From IncidentPilot: once the system prompt has been cached, every request should read
    # it. One that reads nothing means the prefix changed (or the cache expired) -- a silent
    # cost leak a run total would hide.
    usage["requests"] = usage.get("requests", 0) + 1
    if usage["requests"] > 1 and not (getattr(u, "cache_read_input_tokens", 0) or 0):
        usage["uncached_followups"] = usage.get("uncached_followups", 0) + 1


def _echo(content: list[Any]) -> list[dict[str, Any]]:
    """Assistant content to send back on a retry. Thinking (and fallback) blocks go back
    unchanged; parsed text blocks go back as plain text."""
    out = []
    for block in content:
        if block.type == "text":
            out.append({"type": "text", "text": block.text})
        else:
            out.append(block.to_dict())
    return out


class Coach:
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        *,
        api_key: str | None = None,
        effort: str = "low",
        thinking: str = "adaptive",
        max_retries: int = 1,
        max_tokens: int | None = None,
        client: Any = None,
    ) -> None:
        if thinking == "between_tools" and not model.startswith("claude-sonnet-5-5"):
            thinking = "adaptive"   # between_tools exists only on Sonnet 5.5
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.max_retries = max_retries
        # Thinking counts toward max_tokens; the answer itself is well under 2,000.
        self.max_tokens = max_tokens or (4000 if thinking == "between_tools" else 12000)
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        # Created on first use, not in __init__: the SDK takes ~3 s to import, and a view
        # served from the coach_reports cache never calls the API at all.
        if self._client is None:
            import anthropic

            self._client = (anthropic.Anthropic(api_key=self._api_key) if self._api_key
                            else anthropic.Anthropic())
        return self._client

    @property
    def label(self) -> str:
        """Identifies the configuration a cached answer came from."""
        return f"{self.model}/{self.thinking}/{self.effort}"

    def _request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
            "thinking": {"type": self.thinking},
            "output_config": {"effort": self.effort},
            # One breakpoint, on the system prompt shared by every player and game.
            "system": [{"type": "text", "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}}],
            "output_format": CoachOutput,
            "messages": messages,
        }

    def _call(self, messages: list[dict[str, Any]]) -> Any:
        import anthropic

        try:
            resp = self.client.beta.messages.parse(**self._request(messages))
        except anthropic.AuthenticationError as exc:
            raise CoachError("Anthropic API key rejected -- check ANTHROPIC_API_KEY in .env") from exc
        except anthropic.RateLimitError as exc:
            raise CoachError("Anthropic rate limit hit; try again shortly") from exc
        except anthropic.BadRequestError as exc:
            raise CoachError(f"Anthropic rejected the request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise CoachError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise CoachError("could not reach the Anthropic API (network)") from exc

        if resp.stop_reason == "refusal":
            raise CoachError("the model declined to write this coaching")
        if resp.stop_reason == "max_tokens":
            raise CoachError("the coaching answer was cut off (max_tokens)")
        if resp.parsed_output is None:
            raise CoachError("the model returned no structured answer")
        return resp

    def write(self, evidence: EvidenceSet, task: str) -> CoachRun:
        """``task`` says what to coach, e.g. "this single game" or "the recent games"."""
        user = f"Coach the player on {task}.\n\nEvidence:\n{evidence.to_prompt()}"
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        usage: dict[str, Any] = {}

        resp = self._call(messages)
        _accumulate(usage, resp)
        attempts = 1
        output: CoachOutput = resp.parsed_output
        first = problems = validate(output, evidence)

        while problems and attempts <= self.max_retries:
            messages.append({"role": "assistant", "content": _echo(resp.content)})
            messages.append({"role": "user", "content": (
                "Your answer failed the grounding check:\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\n\nReturn the full answer again with these fixed. Remove a number rather "
                  "than guess one, and cite the evidence that contains each number you keep."
            )})
            resp = self._call(messages)
            _accumulate(usage, resp)
            attempts += 1
            output = resp.parsed_output
            problems = validate(output, evidence)

        usage["cost_usd"] = cost_usd(self.model, usage)
        clean, dropped = without_violations(output, problems)
        clean.points = clean.points[:MAX_POINTS]
        return CoachRun(clean, dropped, first, problems, attempts, resp.model, usage)


# -- streaming ------------------------------------------------------------------------------------

def completed_points(text: str) -> list[dict[str, Any]]:
    """Point objects that are fully written in a partial JSON answer.

    Scans the "points" array tracking brackets and strings, so a point is returned only once
    its closing brace has arrived. Anything unparseable is skipped; the final answer is always
    parsed and validated in full anyway.
    """
    start = text.find('"points"')
    if start < 0:
        return []
    start = text.find("[", start)
    if start < 0:
        return []
    out, depth, in_str, esc, obj_start = [], 0, False, False, -1
    for i in range(start + 1, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                try:
                    out.append(json.loads(text[obj_start:i + 1]))
                except json.JSONDecodeError:
                    pass
        elif c == "]" and depth == 0:
            break
    return out


def _stream_events(coach: "Coach", evidence: EvidenceSet, task: str) -> Iterator[dict[str, Any]]:
    """First attempt, streamed: yields {"type": "point", ...} for each point the moment it is
    complete *and* passes the grounding check on its own; returns the final response."""
    import anthropic

    user = f"Coach the player on {task}.\n\nEvidence:\n{evidence.to_prompt()}"
    messages = [{"role": "user", "content": user}]
    sent = 0
    text = ""
    try:
        with coach.client.beta.messages.stream(**coach._request(messages)) as stream:
            for chunk in stream.text_stream:
                text += chunk
                points = completed_points(text)
                for raw in points[sent:]:
                    sent += 1
                    try:
                        point = CoachPoint(**raw)
                    except (TypeError, ValueError):
                        continue
                    if not validate(CoachOutput(headline="", points=[point]), evidence):
                        yield {"type": "point", "point": point.model_dump()}
            resp = stream.get_final_message()
    except anthropic.APIStatusError as exc:
        raise CoachError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise CoachError("could not reach the Anthropic API (network)") from exc
    return messages, resp


def write_stream(coach: "Coach", evidence: EvidenceSet, task: str) -> Iterator[dict[str, Any]]:
    """Stream coaching: grounded points as they complete, then {"type": "done", "run": CoachRun}.

    The final answer goes through exactly the same checks as Coach.write -- full validation,
    one retry with the violations, failing points dropped -- so the "done" event is the
    authority; points streamed before it are a preview that already passed on their own.
    """
    messages, resp = yield from _stream_events(coach, evidence, task)
    if resp.stop_reason == "refusal":
        raise CoachError("the model declined to write this coaching")
    if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
        raise CoachError("the coaching answer was incomplete")
    usage: dict[str, Any] = {}
    _accumulate(usage, resp)
    output: CoachOutput = resp.parsed_output
    first = problems = validate(output, evidence)
    attempts = 1
    while problems and attempts <= coach.max_retries:
        yield {"type": "retrying", "violations": [str(v) for v in problems]}
        messages.append({"role": "assistant", "content": _echo(resp.content)})
        messages.append({"role": "user", "content": (
            "Your answer failed the grounding check:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nReturn the full answer again with these fixed. Remove a number rather "
              "than guess one, and cite the evidence that contains each number you keep."
        )})
        resp = coach._call(messages)
        _accumulate(usage, resp)
        attempts += 1
        output = resp.parsed_output
        problems = validate(output, evidence)
    usage["cost_usd"] = cost_usd(coach.model, usage)
    clean, dropped = without_violations(output, problems)
    clean.points = clean.points[:MAX_POINTS]
    yield {"type": "done", "run": CoachRun(clean, dropped, first, problems, attempts, resp.model, usage)}
