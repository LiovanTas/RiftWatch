"""What the brain sees at a moment of laning.

Each feature belongs to a group -- what it's about -- so importance can be measured and an
explanation given per group ("mostly the health difference and the wave"), and says whether
only replays show it: the HUD panel (mana, abilities, level) and the minimap (lane depth) are
read from replays, not from your own recordings. A brain is trained in two variants, ``full``
(everything) and ``basic`` (what every video shows), and a video is judged by the variant
whose features it has.

Besides the readings themselves, a moment carries their recent history, since a decision
depends on what just happened: who has been losing health over the last seconds, whether the
two champions are closing in, how long they've been in range, how long since the last trade
and how it went, how long since a level-up (a power spike), whether another enemy has been
around. Only the past is used: nothing about a moment's features comes from after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riftwatch.vision import lane

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
CAP_S = 120.0             # "seconds since" features stop here: long ago, or never
FILL_S = 1.0              # a reading missing for up to this long takes the last one


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    about: str
    needs: str | None = None      # "panel" or "minimap": only replays show it


CATALOGUE: tuple[Feature, ...] = (
    Feature("t_min", "time", "game time, minutes"),
    Feature("me", "health", "the player's health, share of the bar"),
    Feature("opponent", "health", "the lane opponent's health"),
    Feature("hp_diff", "health", "the player's health minus the opponent's"),
    Feature("distance", "spacing", "distance between the two health bars, share of the frame height"),
    Feature("closing_2s", "spacing", "change in that distance over the last 2 s (negative: closing in)"),
    Feature("in_range_s", "spacing", "seconds continuously within trading range"),
    Feature("others", "threat", "other enemy champions close by"),
    Feature("others_10s", "threat", "most other enemies close by at once over the last 10 s"),
    Feature("my_minions", "wave", "the player's minions close by"),
    Feature("their_minions", "wave", "the opponent's minions close by"),
    Feature("minion_edge", "wave", "the player's minions minus the opponent's"),
    Feature("me_trend_3s", "momentum", "the player's health change over the last 3 s"),
    Feature("opp_trend_3s", "momentum", "the opponent's health change over the last 3 s"),
    Feature("me_trend_10s", "momentum", "the player's health change over the last 10 s"),
    Feature("opp_trend_10s", "momentum", "the opponent's health change over the last 10 s"),
    Feature("since_trade_s", "history", "seconds since the last trade ended"),
    Feature("last_trade_net", "history", "the last trade's net (health taken minus lost)"),
    Feature("trades_so_far", "history", "trades so far this game"),
    Feature("net_so_far", "history", "net health over those trades"),
    Feature("mana", "resources", "mana (or energy), share of the bar", "panel"),
    Feature("mana_trend_10s", "resources", "mana change over the last 10 s", "panel"),
    *(Feature(f"{k.lower()}_ready", "abilities", f"{k} ready", "panel") for k in "QWER"),
    Feature("basics_ready", "abilities", "how many of Q, W and E are ready", "panel"),
    Feature("d_ready", "summoners", "first summoner spell ready", "panel"),
    Feature("f_ready", "summoners", "second summoner spell ready", "panel"),
    Feature("level", "level", "champion level", "panel"),
    Feature("since_level_up_s", "level", "seconds since the last level-up", "panel"),
    Feature("depth", "position", "how far up the lane: -1 at their own base, +1 at the enemy's",
            "minimap"),
    Feature("depth_trend_5s", "position", "change in depth over the last 5 s (positive: moving up)",
            "minimap"),
    *(Feature(f"role_{r}", "role", f"plays {r.lower()}") for r in ROLES),
)
# Named categories, coded per model (only champions seen in enough games get a code).
CATEGORICAL: tuple[Feature, ...] = (
    Feature("champion", "matchup", "the player's champion"),
    Feature("opponent_champion", "matchup", "the opponent's champion"),
)
BY_NAME = {f.name: f for f in CATALOGUE + CATEGORICAL}
GROUPS = tuple(dict.fromkeys(f.group for f in CATALOGUE + CATEGORICAL))
VARIANTS = {
    "full": [f.name for f in CATALOGUE],
    "basic": [f.name for f in CATALOGUE if f.needs is None],
}


def group_of(name: str) -> str:
    return BY_NAME[name].group


# -- derivation ---------------------------------------------------------------------------

def _ffill(t: np.ndarray, values: np.ndarray, max_gap: float = FILL_S) -> np.ndarray:
    """Missing readings take the last one, if it is at most ``max_gap`` seconds old."""
    have = ~np.isnan(values)
    last_t = pd.Series(np.where(have, t, np.nan)).ffill().to_numpy()
    filled = pd.Series(values).ffill().to_numpy()
    return np.where(t - last_t <= max_gap, filled, np.nan)


def lag(t: np.ndarray, values: np.ndarray, seconds: float, tol: float = 1.0) -> np.ndarray:
    """Each moment's value ``seconds`` earlier (the last reading at or before then, if within
    ``tol``), NaN where there is none."""
    target = t - seconds
    idx = np.searchsorted(t, target, side="right") - 1
    out = np.full(len(t), np.nan)
    ok = idx >= 0
    ok[ok] = target[ok] - t[idx[ok]] <= tol
    out[ok] = values[idx[ok]]
    return out


def _run_seconds(t: np.ndarray, flag: np.ndarray, gap: float = 1.0) -> np.ndarray:
    """Seconds each moment has been in an unbroken run of ``flag``."""
    out = np.zeros(len(t))
    start = None
    for i in range(len(t)):
        if not flag[i]:
            start = None
            continue
        if start is None or t[i] - t[i - 1] > gap:
            start = t[i]
        out[i] = t[i] - start
    return out


def _since_event(t: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """Seconds since the latest event at or before each moment, capped at CAP_S."""
    if len(event_times) == 0:
        return np.full(len(t), CAP_S)
    k = np.searchsorted(event_times, t, side="right")
    since = np.where(k > 0, t - event_times[np.maximum(k - 1, 0)], CAP_S)
    return np.minimum(since, CAP_S)


def derive(s: pd.DataFrame, trades: pd.DataFrame, role: str | None = None,
           champion: str | None = None, opponent: str | None = None) -> pd.DataFrame:
    """Features for every reading of one video. ``s``: its samples sorted by time (columns as
    in video_samples, missing values NaN); ``trades``: its trades."""
    t = s["t"].to_numpy(float)
    col = {c: s[c].to_numpy(float) if c in s else np.full(len(s), np.nan)
           for c in ("me", "opponent", "distance", "others", "my_minions", "their_minions",
                     "mana", "ready", "level", "depth")}
    me, opp, dist = col["me"], col["opponent"], col["distance"]
    out: dict[str, np.ndarray] = {"t_min": t / 60, "me": me, "opponent": opp, "hp_diff": me - opp,
                                  "distance": dist}
    dist_f = _ffill(t, dist)
    out["closing_2s"] = dist_f - lag(t, dist_f, 2)
    out["in_range_s"] = _run_seconds(t, np.nan_to_num(dist_f, nan=np.inf) <= lane.TRADE_RANGE)
    others = np.nan_to_num(col["others"])
    out["others"] = others
    out["others_10s"] = (pd.Series(others, index=pd.to_timedelta(t, unit="s"))
                         .rolling("10s").max().to_numpy())
    out["my_minions"], out["their_minions"] = col["my_minions"], col["their_minions"]
    out["minion_edge"] = col["my_minions"] - col["their_minions"]
    me_f, opp_f = _ffill(t, me), _ffill(t, opp)
    for secs in (3, 10):
        out[f"me_trend_{secs}s"] = me_f - lag(t, me_f, secs)
        out[f"opp_trend_{secs}s"] = opp_f - lag(t, opp_f, secs)

    # The trades that ended before each moment.
    if len(trades):
        order = np.argsort(trades["end_s"].to_numpy(float))
        ends = trades["end_s"].to_numpy(float)[order]
        nets = (trades["opponent_lost"].to_numpy(float) - trades["me_lost"].to_numpy(float))[order]
    else:
        ends = nets = np.array([])
    k = np.searchsorted(ends, t, side="left")              # trades ended strictly before t
    out["since_trade_s"] = np.minimum(np.where(k > 0, t - ends[np.maximum(k - 1, 0)] if len(ends)
                                               else CAP_S, CAP_S), CAP_S)
    out["last_trade_net"] = np.where(k > 0, nets[np.maximum(k - 1, 0)] if len(nets) else 0.0, 0.0)
    out["trades_so_far"] = k.astype(float)
    cum = np.concatenate(([0.0], np.cumsum(nets)))
    out["net_so_far"] = cum[k]

    # The HUD panel (replays): resources, abilities, level.
    mana = col["mana"]
    out["mana"] = mana
    mana_f = _ffill(t, mana)
    out["mana_trend_10s"] = mana_f - lag(t, mana_f, 10)
    ready = col["ready"]
    for i, k_ in enumerate(lane.SLOTS):
        bit = np.where(np.isnan(ready), np.nan, (np.nan_to_num(ready).astype(int) >> i) & 1)
        out[f"{k_.lower()}_ready"] = bit.astype(float)
    out["basics_ready"] = out["q_ready"] + out["w_ready"] + out["e_ready"]
    level = col["level"]
    out["level"] = level
    level_f = pd.Series(level).ffill().to_numpy()
    ups = t[1:][np.nan_to_num(np.diff(level_f)) > 0] if len(t) > 1 else np.array([])
    out["since_level_up_s"] = np.where(np.isnan(level), np.nan, _since_event(t, ups))

    # The minimap (replays): how far up the lane.
    depth = col["depth"]
    out["depth"] = depth
    depth_f = _ffill(t, depth, 3.0)
    out["depth_trend_5s"] = depth_f - lag(t, depth_f, 5)

    for r in ROLES:
        out[f"role_{r}"] = np.full(len(t), float(role == r))
    frame = pd.DataFrame(out, index=s.index)
    frame["champion"] = champion
    frame["opponent_champion"] = opponent
    return frame
