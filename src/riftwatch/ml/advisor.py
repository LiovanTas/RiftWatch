"""Compare a player's decisions with high-elo play, minute by minute.

For each minute of a game the advisor asks the role's models:
  1. how often high-elo players chose each option in situations like this (decision model);
  2. what tended to follow each option within three minutes (outcome models).

A *key moment* is a minute where the player's choice was rare in high-elo play and most
high-elo players did one particular other thing -- a consensus the player went against.

The outcome models predict well *from the situation* (AUC ~0.78) but barely separate the
options within a situation: measured on real games, the predicted gap between a player's
choice and the best common alternative was typically under one point. Each minute's decision
is a coarse label, and in real games the choice is tangled up with the situation (players
gank when a gank was already likely to work). So outcome numbers are quoted only when the gap
is large enough to mean something (OUTCOME_GAP); otherwise a moment says what high-elo players
did, without implying a measured payoff. Evidence says "in similar high-elo situations",
never "would have".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from riftwatch.ml.situations import DECISIONS, Example, examples

# How each decision reads in a sentence ("you farmed", "players chose to ...").
DID = {
    "farm": "farmed camps", "gank_top": "went to gank top", "gank_mid": "went to gank mid",
    "gank_bot": "went to gank bot", "objective": "took part in an objective",
    "invade": "invaded the enemy jungle", "rotate": "rotated through a lane",
    "base": "went back to base", "other": "moved between areas", "lane": "stayed in lane",
    "push": "pushed for plates or the tower", "fight": "joined a fight away from lane",
    "roam": "roamed away from lane", "roam_top": "roamed top", "roam_bot": "roamed bot",
    "roam_mid": "roamed mid", "with_adc": "stayed with the ADC", "ward": "went warding",
}

RARE = 0.10             # the player's choice was this rare among high-elo players here...
CONSENSUS = 0.50        # ...while at least this share of them did one particular other thing
MATCHED = 0.50          # the player matched a consensus this strong: a strength
OUTCOME_GAP = 0.05      # quote outcome numbers only when the value gap is at least this
# Value of an option, for comparing outcomes; the evidence reports each outcome separately.
W_OBJECTIVE, W_DEATH, W_GOLD_PER_1000 = 1.0, 0.6, 0.15


@dataclass
class Option:
    decision: str
    share: float                    # how often high-elo players chose it here
    objective: float                # P(team takes an objective within 3 min)
    death: float                    # P(player dies within 3 min)
    gold: float                     # expected team gold swing over 3 min

    @property
    def value(self) -> float:
        return (W_OBJECTIVE * self.objective - W_DEATH * self.death
                + W_GOLD_PER_1000 * self.gold / 1000)


@dataclass
class Moment:
    minute: int
    role: str
    did: Option
    best: Option
    options: list[Option] = field(default_factory=list)

    @property
    def gain(self) -> float:
        """How strongly high-elo play pointed the other way (used for ranking)."""
        return self.best.share - self.did.share

    @property
    def outcome_gap(self) -> float:
        return self.best.value - self.did.value

    @property
    def outcomes_meaningful(self) -> bool:
        return abs(self.outcome_gap) >= OUTCOME_GAP


@dataclass
class Review:
    role: str
    minutes: int
    agreement: float                 # share of minutes the player picked the top-rated option
    moments: list[Moment]
    good: list[Moment]               # minutes the player matched high-elo play well


class Advisor:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._models: dict[str, Any] = {}

    def available(self, role: str) -> bool:
        return (self.models_dir / f"{role}.joblib").exists()

    def model(self, role: str):
        if role not in self._models:
            from riftwatch.ml.train import load

            self._models[role] = load(self.models_dir, role)
        return self._models[role]

    def review(self, match: dict[str, Any], timeline: dict[str, Any], puuid: str,
               max_moments: int = 3) -> Review | None:
        me = next((p for p in match["info"]["participants"] if p["puuid"] == puuid), None)
        role = (me or {}).get("teamPosition") or ""
        if me is None or role not in DECISIONS or not self.available(role):
            return None
        mine = [e for e in examples(match, timeline, role)
                if e.participant_id == me["participantId"]]
        if not mine:
            return None
        trained = self.model(role)
        all_minutes = self._options(trained, mine)

        moments, good, agree = [], [], 0
        for e, options in zip(mine, all_minutes, strict=True):
            by_name = {o.decision: o for o in options}
            did = by_name.get(e.decision)
            if did is None:
                continue
            # "other" (moving between areas) has nothing specific to coach.
            coachable = e.decision != "other"
            top = max(options, key=lambda o: o.share)
            agree += top.decision == e.decision
            if not coachable:
                continue
            if top.decision != e.decision and did.share < RARE and top.share >= CONSENSUS:
                moments.append(Moment(e.minute, role, did, top, options))
            elif top.decision == e.decision and did.share >= MATCHED:
                runner_up = max((o for o in options if o.decision != e.decision),
                                key=lambda o: o.share)
                good.append(Moment(e.minute, role, did, runner_up, options))
        moments.sort(key=lambda m: -m.gain)
        good.sort(key=lambda m: -m.did.share)
        return Review(role, len(mine), agree / len(mine), moments[:max_moments], good[:1])

    @staticmethod
    def _options(trained, minutes: list[Example]) -> list[list[Option]]:
        """Options for every minute, in one call per model: the decision model over all
        minutes, then each outcome model over every (minute, decision) pair."""
        import pandas as pd

        from riftwatch.ml.train import _matrix

        rows = pd.DataFrame([e.row() for e in minutes])
        for col in trained.features:
            if col not in rows.columns:
                rows[col] = np.nan
        shares = trained.decision_model.predict_proba(_matrix(rows, trained.features, trained.champions))
        classes = list(trained.decision_model.classes_)
        decisions = trained.decisions
        what_if = pd.concat([rows.assign(decision=d) for d in decisions], ignore_index=True)
        xo = _matrix(what_if, trained.features, trained.champions, decisions)
        om = trained.outcome_models
        n = len(minutes)
        objective = om["team_objective"].predict_proba(xo)[:, 1].reshape(len(decisions), n)
        death = om["player_died"].predict_proba(xo)[:, 1].reshape(len(decisions), n)
        gold = om["gold_swing"].predict(xo).reshape(len(decisions), n)
        return [
            [Option(d, float(shares[i, classes.index(d)]) if d in classes else 0.0,
                    float(objective[k, i]), float(death[k, i]), float(gold[k, i]))
             for k, d in enumerate(decisions)]
            for i in range(n)
        ]
