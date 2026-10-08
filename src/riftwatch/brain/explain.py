"""What the brain learned, in words, and why it sees one situation the way it does.

**Patterns** are controlled comparisons. Real situations are taken from the training data and
one thing is changed -- the ultimate ready or on cooldown, 25 health points ahead or behind, a
bigger or smaller wave -- with everything else left as it was; the brain is asked about both
versions. A pattern is kept only if

* every bagged model agrees on its direction, and the effect is big enough to matter;
* the feature group behind it measurably helps on held-out games (scrambling it makes the
  held-out predictions worse in most folds) -- bagged models resample the same games, so
  their agreement alone can't rule out a quirk of those games; and
* the raw data doesn't point the other way: situations that naturally differ like that
  (confounded, but a check on the model) show the same direction, where there are enough.

**Why** a situation scores as it does: each feature group in turn is set to a typical value
(the training median) and the change in the prediction is that group's push. The biggest
pushes are the reasons quoted with a key moment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from riftwatch.brain import features

MIN_RATIO = 1.2           # a probability pattern must change the odds this much
MIN_DIFF = 0.02           # a health pattern must move this much (2 health points)
SAMPLE = 2000
MIN_ROWS = 100
PATTERN_HEADS = ("trade", "traded_on", "trade_won", "trade_net")


def _set(**cols) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def apply(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for k, v in cols.items():
            d[k] = v(d) if callable(v) else v
        return d
    return apply


def _health(delta: float) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def apply(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["opponent"] = np.clip(d["me"] - delta, 0.02, 1.0)
        d["hp_diff"] = d["me"] - d["opponent"]
        return d
    return apply


def _wave(edge: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def apply(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        if edge > 0:
            d["my_minions"] = d["their_minions"] + edge
        else:
            d["their_minions"] = d["my_minions"] - edge
        d["minion_edge"] = d["my_minions"] - d["their_minions"]
        return d
    return apply


@dataclass(frozen=True)
class Contrast:
    key: str
    a: str                                   # "with the ultimate ready"
    b: str                                   # "with it on cooldown"
    needs: tuple[str, ...]
    keep: Callable[[pd.DataFrame], pd.Series]
    set_a: Callable[[pd.DataFrame], pd.DataFrame]
    set_b: Callable[[pd.DataFrame], pd.DataFrame]
    natural: Callable[[pd.DataFrame], tuple[pd.Series, pd.Series]]


CONTRASTS: tuple[Contrast, ...] = (
    Contrast("ultimate", "with the ultimate ready", "with it on cooldown", ("r_ready",),
             lambda d: (d["level"] >= 6) & d["r_ready"].notna(),
             _set(r_ready=1.0), _set(r_ready=0.0),
             lambda d: ((d["level"] >= 6) & (d["r_ready"] == 1), (d["level"] >= 6) & (d["r_ready"] == 0))),
    Contrast("health", "25 health points ahead", "25 points behind", ("hp_diff",),
             lambda d: d["me"].between(0.35, 0.75), _health(0.25), _health(-0.25),
             lambda d: (d["hp_diff"] >= 0.15, d["hp_diff"] <= -0.15)),
    Contrast("wave", "with 3 more minions close by", "with 3 fewer", ("minion_edge",),
             lambda d: d["my_minions"].notna() & d["their_minions"].notna(), _wave(3), _wave(-3),
             lambda d: (d["minion_edge"] >= 2, d["minion_edge"] <= -2)),
    Contrast("level_up", "within 3 seconds of a level-up", "a minute after one",
             ("since_level_up_s",), lambda d: d["since_level_up_s"].notna(),
             _set(since_level_up_s=3.0), _set(since_level_up_s=60.0),
             lambda d: (d["since_level_up_s"] <= 10, d["since_level_up_s"] >= 30)),
    Contrast("mana", "at 80% mana", "at 20%", ("mana",), lambda d: d["mana"].notna(),
             _set(mana=0.8), _set(mana=0.2),
             lambda d: (d["mana"] >= 0.6, d["mana"] <= 0.3)),
    Contrast("summoners", "with both summoner spells up", "with both down", ("d_ready", "f_ready"),
             lambda d: d["d_ready"].notna() & d["f_ready"].notna(),
             _set(d_ready=1.0, f_ready=1.0), _set(d_ready=0.0, f_ready=0.0),
             lambda d: ((d["d_ready"] == 1) & (d["f_ready"] == 1),
                        (d["d_ready"] == 0) & (d["f_ready"] == 0))),
    Contrast("opponent_hurt", "right after the opponent lost 10 health points",
             "when neither just lost any", ("opp_trend_3s",),
             lambda d: d["opp_trend_3s"].notna(), _set(opp_trend_3s=-0.1), _set(opp_trend_3s=0.0),
             lambda d: (d["opp_trend_3s"] <= -0.05, d["opp_trend_3s"].abs() < 0.01)),
    Contrast("another_enemy", "with another enemy close by", "without one", ("others",),
             lambda d: d["others"].notna(),
             _set(others=1.0, others_10s=lambda d: np.maximum(d["others_10s"], 1.0)),
             _set(others=0.0, others_10s=0.0),
             lambda d: (d["others"] >= 1, d["others"] == 0)),
    Contrast("closing", "while closing in", "while moving apart", ("closing_2s",),
             lambda d: d["closing_2s"].notna(), _set(closing_2s=-0.1), _set(closing_2s=0.1),
             lambda d: (d["closing_2s"] <= -0.05, d["closing_2s"] >= 0.05)),
)


def _text(head: str, a: str, b: str, va: float, vb: float) -> str:
    if head in ("trade", "traded_on"):
        if va < vb:
            a, b, va, vb = b, a, vb, va
        who = ("high-elo players start trades" if head == "trade"
               else "opponents start trades on high-elo players")
        return (f"In otherwise similar spots, {who} {va / vb:.1f}x as often {a} as {b} "
                f"(about {60 * va:.1f} vs {60 * vb:.1f} times per minute in range).")
    if head == "trade_won":
        return (f"Trades high-elo players start {a} are won {100 * va:.0f}% of the time, against "
                f"{100 * vb:.0f}% {b}, other things equal.")
    return (f"Trades high-elo players start {a} net {100 * va:+.0f} health points on average, "
            f"against {100 * vb:+.0f} {b}, other things equal.")


def _helps(unit, group: str) -> bool:
    """Scrambling the group made held-out predictions worse, in most folds."""
    importance = unit.metrics.get("importance")
    if importance is None:
        return True                              # not measured: no gate
    g = importance.get(group)
    folds = unit.metrics.get("folds", 0)
    return bool(g and g["loss_increase"] > 0 and g["folds_hurt"] >= np.ceil(0.6 * folds))


def patterns(brain, df: pd.DataFrame, seed: int = 0) -> list[dict]:
    """The controlled comparisons that every bagged model agrees on, strongest first."""
    rng = np.random.default_rng(seed)
    out = []
    for head in PATTERN_HEADS:
        unit = brain.unit(head, "full")
        if unit is None or not unit.usable or len(df) == 0:
            continue
        labelled = df[df[head].notna()]
        for c in CONTRASTS:
            if not all(n in unit.columns for n in c.needs):
                continue
            rows = labelled[c.keep(labelled).fillna(False).to_numpy(bool)]
            if len(rows) < MIN_ROWS:
                continue
            if len(rows) > SAMPLE:
                rows = rows.iloc[np.sort(rng.choice(len(rows), SAMPLE, replace=False))]
            each_a = unit.predict_each(c.set_a(rows)).mean(axis=1)
            each_b = unit.predict_each(c.set_b(rows)).mean(axis=1)
            va, vb = float(each_a.mean()), float(each_b.mean())
            agree = int(max((each_a > each_b).sum(), (each_a < each_b).sum()))
            binary = unit.kind == "binary"
            big = (max(va, vb) / max(min(va, vb), 1e-9) >= MIN_RATIO if binary
                   else abs(va - vb) >= MIN_DIFF)
            if agree < len(each_a) or not big or len(each_a) < 2:
                continue
            if not _helps(unit, features.group_of(c.needs[0])):
                continue
            na, nb = c.natural(labelled)
            na, nb = na.fillna(False).to_numpy(bool), nb.fillna(False).to_numpy(bool)
            natural = None
            if na.sum() >= MIN_ROWS and nb.sum() >= MIN_ROWS:
                ya = labelled[head].to_numpy(float)
                natural = {"a": round(float(ya[na].mean()), 5), "n_a": int(na.sum()),
                           "b": round(float(ya[nb].mean()), 5), "n_b": int(nb.sum())}
                if (natural["a"] - natural["b"]) * (va - vb) < 0:
                    continue                     # the raw data points the other way
            strength = (abs(np.log(max(va, 1e-9) / max(vb, 1e-9))) if binary
                        else abs(va - vb) * 10)
            out.append({"head": head, "key": c.key, "a": c.a, "b": c.b,
                        "value_a": round(va, 5), "value_b": round(vb, 5),
                        "models_agree": f"{agree}/{len(each_a)}", "situations": int(len(rows)),
                        "natural": natural, "strength": round(float(strength), 4),
                        "text": _text(head, c.a, c.b, va, vb)})
    return sorted(out, key=lambda p: -p["strength"])


# -- one situation ------------------------------------------------------------------------

def pushes(unit, rows: pd.DataFrame) -> list[list[tuple[str, float]]]:
    """For each row, every feature group's push on the prediction (its value minus the
    prediction with that group set to typical), biggest first. Measured before calibration,
    which keeps their order and direction (it is monotonic) and doesn't flatten small ones."""
    if not len(rows):
        return []
    base = unit.predict_raw(rows).mean(axis=0)
    groups: dict[str, list[str]] = {}
    for c in unit.columns + list(unit.categories):
        groups.setdefault(features.group_of(c), []).append(c)
    deltas = {}
    for g, cols in groups.items():
        d = rows.copy()
        for c in cols:
            d[c] = unit.medians.get(c, np.nan) if c in unit.medians else None
        deltas[g] = base - unit.predict_raw(d).mean(axis=0)
    out = []
    for i in range(len(rows)):
        out.append(sorted(((g, float(v[i])) for g, v in deltas.items()), key=lambda gv: -gv[1]))
    return out


def describe(group: str, r: pd.Series) -> str | None:
    """The situation's values for one feature group, in words, for the player's own game."""
    def known(*names):
        return all(n in r and pd.notna(r[n]) for n in names)

    if group == "health" and known("me", "opponent"):
        return f"your health {100 * r['me']:.0f}% against their {100 * r['opponent']:.0f}%"
    if group == "wave" and known("my_minions", "their_minions"):
        return f"{r['my_minions']:.0f} of your minions close by against {r['their_minions']:.0f} of theirs"
    if group == "abilities" and known("q_ready", "w_ready", "e_ready", "r_ready"):
        up = [k for k in "QWER" if r[f"{k.lower()}_ready"] == 1]
        if not up:
            return "no ability ready"
        return (up[0] if len(up) == 1 else ", ".join(up[:-1]) + " and " + up[-1]) + " ready"
    if group == "summoners" and known("d_ready", "f_ready"):
        n = int(r["d_ready"] + r["f_ready"])
        return ("both summoner spells up", "one summoner spell up", "both summoner spells down")[2 - n]
    if group == "level" and known("level", "since_level_up_s"):
        recent = r["since_level_up_s"]
        return (f"level {r['level']:.0f}, {recent:.0f} seconds after leveling up"
                if recent < 30 else f"level {r['level']:.0f}")
    if group == "resources" and known("mana"):
        return f"mana at {100 * r['mana']:.0f}%"
    if group == "momentum" and known("opp_trend_3s", "me_trend_3s"):
        if r["opp_trend_3s"] <= -0.03:
            return f"they had just lost {-100 * r['opp_trend_3s']:.0f} health points"
        if r["me_trend_3s"] <= -0.03:
            return f"you had just lost {-100 * r['me_trend_3s']:.0f} health points"
        return None
    if group == "spacing" and known("in_range_s"):
        return f"{r['in_range_s']:.0f} seconds in range of them"
    if group == "history" and known("since_trade_s"):
        if r["since_trade_s"] >= features.CAP_S:
            return "no trade for over two minutes"
        return f"{r['since_trade_s']:.0f} seconds since the last trade"
    if group == "threat" and known("others"):
        return "another enemy close by" if r["others"] >= 1 else "no other enemy close by"
    if group == "position" and known("depth"):
        return "pushed up the lane" if r["depth"] > 0.1 else (
            "back near your tower" if r["depth"] < -0.1 else "mid-lane")
    if group == "time" and known("t_min"):
        return None
    return None


def reasons(unit, rows: pd.DataFrame, top: int = 2, sign: int = 1) -> list[list[str]]:
    """For each row, the ``top`` feature groups pushing the prediction the way of ``sign``
    (+1 up, -1 down), in words."""
    out = []
    for row_pushes, (_, r) in zip(pushes(unit, rows), rows.iterrows(), strict=True):
        words = []
        for g, v in sorted(row_pushes, key=lambda gv: -sign * gv[1]):
            if sign * v <= 0 or len(words) >= top:
                break
            text = describe(g, r)
            if text:
                words.append(text)
        out.append(words)
    return out
