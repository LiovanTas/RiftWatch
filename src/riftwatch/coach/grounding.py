"""The grounding check: every claim must trace back to cited evidence.

Rules for each coaching point:

1. It cites at least one evidence id, and every cited id exists.
2. Every number it writes (in title, explanation or advice) appears in the text of an
   evidence item it cites -- after rounding the evidence value to the precision the coach
   wrote ("6.94" supports "6.9" and "7", but not "8").

The headline cites nothing, so its numbers must appear somewhere in the evidence.
A number is anything the regex below finds, so "62nd percentile", "+340 gold", "13:20"
and "45%" are all checked; evidence ids like "E3" and matchup shorthand like "2v2" are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from riftwatch.coach.evidence import EvidenceSet
from riftwatch.features.metrics import AREAS

_IGNORED = re.compile(r"\bE\d+\b|\b\d+v\d+\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")

Area = Literal["farming", "laning", "fighting", "survival", "vision", "objectives", "economy",
               "macro"]
assert set(Area.__args__) == set(AREAS)


class CoachPoint(BaseModel):
    area: Area
    kind: Literal["weakness", "strength"]
    title: str
    explanation: str
    advice: str
    evidence_ids: list[str]


class CoachOutput(BaseModel):
    headline: str
    points: list[CoachPoint]


@dataclass(frozen=True)
class Violation:
    point: int | None      # index into points; None = headline
    problem: str

    def __str__(self) -> str:
        where = "headline" if self.point is None else f"point {self.point + 1}"
        return f"{where}: {self.problem}"


def numbers(text: str) -> list[tuple[float, int, str]]:
    """(value, decimals written, raw text) for every number in ``text``."""
    out = []
    for m in _NUMBER.finditer(_IGNORED.sub(" ", text)):
        raw = m.group(0)
        clean = raw.replace(",", "")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        out.append((float(clean), decimals, raw))
    return out


def supported(value: float, decimals: int, pool: list[float]) -> bool:
    target = round(value, decimals)
    return any(round(e, decimals) == target for e in pool)


def validate(output: CoachOutput, evidence: EvidenceSet) -> list[Violation]:
    by_id = evidence.by_id()
    all_numbers = [v for e in evidence.items for v, _, _ in numbers(e.text)]
    problems: list[Violation] = []

    for value, decimals, raw in numbers(output.headline):
        if not supported(value, decimals, all_numbers):
            problems.append(Violation(None, f"number {raw!r} is not in any evidence"))

    for i, point in enumerate(output.points):
        if not point.evidence_ids:
            problems.append(Violation(i, "cites no evidence"))
            continue
        unknown = [eid for eid in point.evidence_ids if eid not in by_id]
        if unknown:
            problems.append(Violation(i, f"cites unknown evidence {', '.join(unknown)}"))
        cited = [by_id[eid] for eid in point.evidence_ids if eid in by_id]
        pool = [v for e in cited for v, _, _ in numbers(e.text)]
        for field_name in ("title", "explanation", "advice"):
            for value, decimals, raw in numbers(getattr(point, field_name)):
                if not supported(value, decimals, pool):
                    problems.append(Violation(
                        i, f"{field_name} states {raw!r}, which is not in its cited evidence "
                           f"({', '.join(point.evidence_ids)})"))
    return problems


def without_violations(output: CoachOutput, problems: list[Violation]) -> tuple[CoachOutput, list[CoachPoint]]:
    """Drop every point with a violation. A bad headline is replaced by a neutral one."""
    bad = {v.point for v in problems if v.point is not None}
    kept = [p for i, p in enumerate(output.points) if i not in bad]
    dropped = [p for i, p in enumerate(output.points) if i in bad]
    headline = output.headline
    if any(v.point is None for v in problems):
        headline = "Coaching notes for this game, based on the measured evidence below."
    return CoachOutput(headline=headline, points=kept), dropped
