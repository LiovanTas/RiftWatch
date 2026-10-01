"""The LLM coach: Claude turns evidence into advice, then the grounding check audits it.

Flow per request:
  1. Claude gets the numbered evidence and nothing else about the game, and must answer in
     the CoachOutput JSON schema (structured outputs), citing evidence ids.
  2. :func:`grounding.validate` checks every id and every number.
  3. If anything fails, Claude sees the exact violations once and answers again.
  4. Points that still fail are dropped, never shown.

Speed: the system prompt is identical for every request and marked for prompt caching;
callers cache finished answers in Postgres (see coach.pipeline), so this runs once per
distinct evidence set.
"""

from __future__ import annotations

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

SYSTEM_PROMPT = """\
You are a League of Legends coach reviewing a ranked player's performance.

You will receive a numbered list of evidence items. They are the only facts you know about
the player and their games: measured values, percentiles against players of the same rank
and role, and details of each death. Write coaching points from them.

Rules:
- Every point cites the ids of the evidence it is based on in evidence_ids.
- Every number you write must appear in the evidence you cite for that point. Do not
  compute new numbers (differences, ratios, totals, averages) and do not round to new
  values -- reuse the numbers exactly as written in the evidence. If you want to say
  something that needs a number the evidence doesn't contain, say it without the number.
- Do not invent facts about the game that the evidence doesn't contain (items, matchups,
  enemy champions, teammates' play). General League knowledge is fine in advice, as long
  as it is phrased as advice and not as a claim about this player's game.
- Prefer the most important issues: sustained patterns and large gaps from the comparison
  group over small ones. Weaknesses first, then at least one strength if the evidence has
  any. Usually 3 to 6 points; fewer if the evidence is thin.
- advice is concrete and actionable: what to do differently next game.
- headline is one sentence summarising the overall picture.
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


def _usage(resp: Any) -> dict[str, int]:
    u = resp.usage
    return {
        "input_tokens": u.input_tokens or 0,
        "output_tokens": u.output_tokens or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


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
        model: str,
        *,
        api_key: str | None = None,
        effort: str = "medium",
        max_retries: int = 1,
        client: Any = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_retries = max_retries
        if client is None:
            # Imported here, not at module load: the SDK takes ~3 s to import, and most
            # commands (and the offline coach) never call the API.
            import anthropic

            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.client = client

    def _call(self, messages: list[dict[str, Any]]) -> Any:
        import anthropic

        try:
            resp = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": self.effort},
                output_format=CoachOutput,
                messages=messages,
            )
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
        user = (f"Coach the player on {task}.\n\nEvidence:\n{evidence.to_prompt()}")
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        usage: dict[str, int] = {}

        resp = self._call(messages)
        attempts = 1
        for k, v in _usage(resp).items():
            usage[k] = usage.get(k, 0) + v
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
            attempts += 1
            for k, v in _usage(resp).items():
                usage[k] = usage.get(k, 0) + v
            output = resp.parsed_output
            problems = validate(output, evidence)

        clean, dropped = without_violations(output, problems)
        return CoachRun(clean, dropped, first, problems, attempts, resp.model, usage)
