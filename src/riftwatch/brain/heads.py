"""The questions the brain answers about a situation, one model ("head") each.

``trade`` is the policy: what high-elo players do from a spot like this. ``traded_on`` is the
threat: what their opponents do to them from it. The rest are what follows, and the trade
heads learn only from situations where the player started a (clean) trade -- so asked about
any spot, they answer "if a high-elo player traded from here, how did it tend to go".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Head:
    name: str
    kind: str              # "binary" or "regression"
    about: str             # what it predicts, finishing "the chance that ..." / "the ..."
    unit: str = ""         # for regression heads: what the number is in


HEADS: tuple[Head, ...] = (
    Head("trade", "binary", "a high-elo player starts a trade within the next second"),
    Head("traded_on", "binary", "the opponent starts a trade on them within the next second"),
    Head("trade_won", "binary", "a trade they start from here is won"),
    Head("trade_net", "regression", "net of a trade started from here (health taken minus lost)",
         "share of the bar"),
    Head("swing_10s", "regression", "change in the health difference over the next 10 seconds",
         "share of the bar"),
    Head("died_15s", "binary", "they die within 15 seconds"),
    Head("back_45s", "binary", "they are back in base within 45 seconds (a recall or a death)"),
)
BY_NAME = {h.name: h for h in HEADS}
PRIMARY = "trade"
