"""The brain applied to one video: how the player's laning compares with high-elo play, and the
moments that show it.

Every situation of the video is put to the brain. Situations next to each other form a
*spot* (an unbroken stretch in range of the opponent, outside a trade). Then:

* **Trading rate.** Each situation's trade probability is the chance a high-elo player starts
  a trade within the next second, so summed over a spot (times the reading interval) it is the
  number of trades they would start there on average. Summed over the video, it is compared
  with the trades the player actually started.
* **Missed trades.** Spots where high-elo players would on average have started at least
  MISSED_EXPECTED trades, which went well for them, and the player started none.
* **Rare trades that went badly.** Trades the player started from a spot rarer than
  RARE of the spots high-elo players start trades from, that the brain expected to go
  badly, and that the player lost.
* **Good trades.** Trades started from a spot where high-elo players commonly trade, won.
* **Spacing.** Trades the opponent started and won, after seconds the brain rated among the
  most dangerous tenth of high-elo situations, with the player staying in range.
* **Risky trades.** Trades the player started and died right after, from a spot where the
  brain expected deaths at least twice as often as usual.

Each moment carries the reasons behind the brain's view (``explain.reasons``). Moments only
come from heads the brain marked usable, and outcome numbers only from outcome heads that are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psycopg

from riftwatch.brain import data, explain

MISSED_EXPECTED = 0.6     # high-elo players would have started at least this many trades ...
MISSED_NET = 0.04         # ... netting at least this (share of the bar) ...
MISSED_WON = 0.5          # ... or winning at least this share
RARE = "p10"              # a spot below this quantile of high-elo trade starts is rare
COMMON = "p50"
THREAT = "p90"            # seconds rated above this quantile of high-elo situations: danger
RISK_RATIO = 2.0
GAP_S = 0.75              # readings further apart than this split spots
PRE_S = 1.5               # the situation read at most this long before a trade is its spot
SEPARATE_S = 10.0         # two moments closer than this are one
MAX_WEAK, MAX_STRONG = 5, 2
ROLE_NAMES = {"TOP": "top laners", "JUNGLE": "junglers", "MIDDLE": "mid laners",
              "BOTTOM": "ADCs", "UTILITY": "supports"}


@dataclass
class Moment:
    kind: str                # missed_trade | rare_trade | good_trade | spacing | risky_trade
    t: float                 # game time, seconds
    polarity: str            # weakness | strength
    impact: float            # health points at stake, for ranking
    text: str
    data: dict = field(default_factory=dict)


@dataclass
class Review:
    video_id: int
    role: str | None
    variant: str
    brain_version: str
    brain_games: int
    situations: int
    minutes_in_range: float
    expected_trades: float | None        # what high-elo players would have started
    trades: int                          # what the player started
    expected_net: float | None           # the brain's expectation for the player's trades
    actual_net: float | None
    moments: list[Moment] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)

    @property
    def who(self) -> str:
        return f"high-elo {ROLE_NAMES.get(self.role or '', 'laners')}"

    def summary_text(self) -> str | None:
        if self.expected_trades is None:
            return None
        text = (f"Laning brain (trained on {self.brain_games} high-elo games, scored on games it "
                f"never saw): over the {self.minutes_in_range:.1f} minutes you spent within trading "
                f"range of your opponent before 14:00, {self.who} in the same spots would have "
                f"started about {self.expected_trades:.1f} trades; you started {self.trades}.")
        if self.expected_net is not None and self.actual_net is not None:
            text += (f" From the spots you traded from, it expected a net of "
                     f"{100 * self.expected_net:+.0f} health points per trade; yours netted "
                     f"{100 * self.actual_net:+.0f}.")
        return text

    @property
    def style(self) -> str | None:
        """passive / aggressive when the player's trading rate is far from high-elo's."""
        e = self.expected_trades
        if e is None:
            return None
        if e >= 3 and self.trades < 0.5 * e:
            return "passive"
        if self.trades >= 3 and self.trades > 2 * max(e, 0.5):
            return "aggressive"
        return None


def clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _hp(x: float) -> str:
    return f"{100 * x:.0f}"


def _and(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _trades(conn, video_id: int) -> pd.DataFrame:
    rows = conn.execute(
        """SELECT start_s, end_s, me_lost, opponent_lost, started_by, skirmish, result, died
             FROM video_trades WHERE video_id = %s ORDER BY start_s""", (video_id,)).fetchall()
    return pd.DataFrame(rows, columns=["start_s", "end_s", "me_lost", "opponent_lost",
                                       "started_by", "skirmish", "result", "died"])


def _spots(t: np.ndarray) -> list[np.ndarray]:
    if not len(t):
        return []
    breaks = np.flatnonzero(np.diff(t) > GAP_S) + 1
    return np.split(np.arange(len(t)), breaks)


def review(conn: psycopg.Connection, brain, video_id: int) -> Review | None:
    df = data.load(conn, view=None, video_ids=[video_id], checked=False)
    if df.empty:
        return None
    df = df.sort_values("t").reset_index(drop=True)
    role = next((r for r in ROLE_NAMES if df[f"role_{r}"].iloc[0] == 1), None)
    variant = brain.variant_for(df)
    pred = brain.predict(df, variant)
    trades = _trades(conn, video_id)
    t = df["t"].to_numpy(float)
    step = float(np.clip(np.median(np.diff(t)), 0.1, 1.0)) if len(t) > 1 else 0.25
    mine = trades[trades["started_by"].isin(["you", "both"])]
    rv = Review(video_id, role, variant, brain.version, brain.data.get("games", 0), len(df),
                round(len(df) * step / 60, 1), None, int(len(mine)), None, None)
    rv.patterns = [p for p in brain.patterns if p["head"] in ("trade", "trade_won", "trade_net")]

    def unit(head):
        u = brain.unit(head, variant, role)
        return u if u is not None and u.usable and head in pred else None

    trade_u, net_u, won_u = unit("trade"), unit("trade_net"), unit("trade_won")
    p_trade = pred["trade"].to_numpy(float) if trade_u else None
    if p_trade is not None:
        rv.expected_trades = round(float(np.nansum(p_trade) * step / data.DECISION_S), 1)

    # The situation each trade started from.
    def pre_row(start: float) -> int | None:
        i = int(np.searchsorted(t, start, side="left")) - 1
        return i if i >= 0 and start - t[i] <= PRE_S else None

    clean_mine = mine[~mine["skirmish"].astype(bool)]
    pre = [(tr, pre_row(tr.start_s)) for tr in clean_mine.itertuples()]
    pre = [(tr, i) for tr, i in pre if i is not None]
    if net_u is not None and pre:
        rv.expected_net = round(float(np.mean([pred["trade_net"].iloc[i] for _, i in pre])), 4)
        rv.actual_net = round(float(np.mean([tr.opponent_lost - tr.me_lost for tr, _ in pre])), 4)

    moments: list[Moment] = []
    if trade_u is not None:
        moments += _missed(df, pred, t, step, trades, trade_u, net_u, won_u, rv.who)
        moments += _rare_and_good(df, pred, pre, trade_u, net_u, won_u, rv.who)
    traded_on = unit("traded_on")
    if traded_on is not None:
        moments += _spacing(df, pred, t, trades, traded_on, rv.who)
    died = unit("died_15s")
    if died is not None:
        moments += _risky(df, pred, pre, died, rv.who)
    rv.moments = _pick(moments)
    return rv


def _missed(df, pred, t, step, trades, trade_u, net_u, won_u, who) -> list[Moment]:
    p = pred["trade"].to_numpy(float)
    starts = trades["start_s"].to_numpy(float)
    mine = trades["started_by"].isin(["you", "both"]).to_numpy()
    out = []
    for idx in _spots(t):
        lo, hi = t[idx[0]], t[idx[-1]] + data.DECISION_S
        expected = float(np.nansum(p[idx]) * step / data.DECISION_S)
        if expected < MISSED_EXPECTED or (mine & (starts >= lo) & (starts <= hi)).any():
            continue
        weight = np.nan_to_num(p[idx])
        net = float(np.average(pred["trade_net"].to_numpy(float)[idx], weights=weight)) \
            if net_u is not None else None
        won = float(np.average(pred["trade_won"].to_numpy(float)[idx], weights=weight)) \
            if won_u is not None else None
        if net is None and won is None and expected < 1.0:
            continue
        if (net is not None and net < MISSED_NET) or (won is not None and won < MISSED_WON):
            continue
        best = idx[int(np.nanargmax(p[idx]))]
        why = explain.reasons(trade_u, df.iloc[[best]], sign=1)[0]
        text = (f"At {clock(t[idx[0]])} you were within trading range of your opponent for "
                f"{t[idx[-1]] - t[idx[0]] + step:.0f} seconds"
                + (f" ({_and(why)})" if why else "")
                + f" and didn't trade. {who.capitalize()} in spots like this would have started "
                f"about {expected:.1f} trades over that time")
        payoff = []
        if net is not None:
            payoff.append(f"netted {100 * net:+.0f} health points on average")
        if won is not None:
            payoff.append(f"were won {100 * won:.0f}% of the time")
        text += (f"; trades they started from such spots {_and(payoff)}." if payoff else ".")
        out.append(Moment("missed_trade", float(t[idx[0]]), "weakness",
                          expected * max(net or 0.05, 0.05) * 100, text,
                          {"expected": round(expected, 2), "net": net, "won": won}))
    return out


def _rare_and_good(df, pred, pre, trade_u, net_u, won_u, who) -> list[Moment]:
    q = trade_u.quantiles.get("positive", {})
    if RARE not in q:
        return []
    out = []
    for tr, i in pre:
        p = float(pred["trade"].iloc[i])
        net_exp = float(pred["trade_net"].iloc[i]) if net_u is not None else None
        won_exp = float(pred["trade_won"].iloc[i]) if won_u is not None else None
        lost, taken = tr.me_lost, tr.opponent_lost
        result = (f"{'won' if tr.result == 'won' else 'lost' if tr.result == 'lost' else 'evened'} "
                  f"it ({_hp(taken)} health points taken, {_hp(lost)} lost)")
        if p < q[RARE] and tr.result == "lost" and (
                (net_exp is not None and net_exp < 0) or (won_exp is not None and won_exp < 0.45)):
            why = explain.reasons(trade_u, df.iloc[[i]], sign=-1)[0]
            text = (f"At {clock(tr.start_s)} you started a trade and {result}. {who.capitalize()} "
                    f"rarely start trades from spots like this"
                    + (f" ({_and(why)})" if why else "")
                    + f": rarer than {100 - int(RARE[1:])}% of the spots they do trade from")
            if net_exp is not None:
                text += f", and trades they started from such spots netted {100 * net_exp:+.0f} on average"
            out.append(Moment("rare_trade", float(tr.start_s), "weakness",
                              100 * (lost - taken), text + ".",
                              {"p": p, "net_expected": net_exp, "won_expected": won_exp}))
        elif p >= q.get(COMMON, 1.0) and tr.result == "won":
            why = explain.reasons(trade_u, df.iloc[[i]], sign=1)[0]
            out.append(Moment("good_trade", float(tr.start_s), "strength", 100 * (taken - lost),
                              f"At {clock(tr.start_s)} you started a trade and {result}, from a "
                              f"spot where {who} commonly trade"
                              + (f" ({_and(why)})" if why else "") + ".", {"p": p}))
    return out


def _spacing(df, pred, t, trades, unit, who) -> list[Moment]:
    q = unit.quantiles.get("all", {})
    rate = unit.metrics.get("rate")
    if THREAT not in q or not rate:
        return []
    p = pred["traded_on"].to_numpy(float)
    out = []
    for tr in trades[(trades["started_by"] == "opponent") & ~trades["skirmish"].astype(bool)
                     & (trades["result"] == "lost")].itertuples():
        window = np.flatnonzero((t >= tr.start_s - 3) & (t < tr.start_s))
        if not len(window):
            continue
        i = window[int(np.nanargmax(p[window]))]
        if p[i] < q[THREAT] or df["in_range_s"].iloc[window[-1]] < 2:
            continue
        why = explain.reasons(unit, df.iloc[[i]], sign=1)[0]
        text = (f"At {clock(tr.start_s)} your opponent started a trade and won it "
                f"({_hp(tr.me_lost)} health points to {_hp(tr.opponent_lost)}). In the seconds "
                f"before, the brain rated the spot as one where opponents hit {who} "
                f"{p[i] / rate:.1f}x as often as usual" + (f" ({_and(why)})" if why else "")
                + f"; you had stayed in range for {df['in_range_s'].iloc[window[-1]]:.0f} seconds.")
        out.append(Moment("spacing", float(tr.start_s), "weakness",
                          100 * (tr.me_lost - tr.opponent_lost), text, {"ratio": p[i] / rate}))
    return out


def _risky(df, pred, pre, unit, who) -> list[Moment]:
    rate = unit.metrics.get("rate")
    if not rate:
        return []
    out = []
    for tr, i in pre:
        p = float(pred["died_15s"].iloc[i])
        if not tr.died or p < RISK_RATIO * rate:
            continue
        why = explain.reasons(unit, df.iloc[[i]], sign=1)[0]
        out.append(Moment(
            "risky_trade", float(tr.start_s), "weakness", 100.0,
            f"At {clock(tr.start_s)} you started a trade and died within 10 seconds of it. In "
            f"spots like this {who} die within 15 seconds {p / rate:.1f}x as often as usual"
            + (f" ({_and(why)})" if why else "") + ".", {"ratio": p / rate}))
    return out


def _pick(moments: list[Moment]) -> list[Moment]:
    """The biggest weaknesses and strengths, no two within SEPARATE_S of each other."""
    out: list[Moment] = []
    for polarity, limit in (("weakness", MAX_WEAK), ("strength", MAX_STRONG)):
        chosen: list[Moment] = []
        for m in sorted((m for m in moments if m.polarity == polarity), key=lambda m: -m.impact):
            if len(chosen) >= limit:
                break
            if all(abs(m.t - c.t) >= SEPARATE_S for c in out + chosen):
                chosen.append(m)
        out += chosen
    return sorted(out, key=lambda m: m.t)
