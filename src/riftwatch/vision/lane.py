"""Laning from video: trades with the lane opponent, and spacing.

The camera follows one champion (yours, or the one a replay follows), so the bars on screen
place everyone relative to it. For each moment this module takes the followed champion's
health, the nearest enemy champion bar (tracked from moment to moment, so a passing jungler
doesn't replace the opponent), and the distance between them in screen pixels.

A trade is a stretch where the two are within TRADE_RANGE and at least one of them loses
health; it ends when nobody has lost health for QUIET_S. Bars flicker out of view under spell
effects mid-fight, so the opponent counts as in range for QUIET_S after last being seen there
-- otherwise one fight would be cut into pieces and the losses between them dropped. Its result compares the health each
side lost, in health points (share of each champion's own maximum): more than EVEN_MARGIN
apart is won or lost. Who started it is whose health dropped first. Another enemy within range
makes it a skirmish, not a clean trade.

Health is read from bars, about two health points off at the 90th percentile on real footage
(see vision.spectator); shields, heals and regeneration mid-trade show up as they look on
screen. Distances are screen pixels, so they shrink when a replay zooms out.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import fmean, median

from riftwatch.vision.healthbars import Bar

TRADE_RANGE = 0.35       # max distance between bars, as a share of frame height
MINION_RANGE = 0.5       # minions this close (share of frame height) count as "your wave"
WAVE_EDGE = 2            # this many more minions than the opponent is a minion advantage
DROP = 0.03              # a health drop of 3 points between samples starts or extends a trade
QUIET_S = 2.0            # a trade ends after this long without either side losing health
EVEN_MARGIN = 0.05       # health-point difference below which a trade is even
LANE_END_S = 14 * 60     # game time when laning is over


@dataclass
class Sample:
    t: float                         # game time, seconds
    me: float | None                 # followed champion's health share
    opponent: float | None
    distance: float | None           # px between the two bars, as a share of frame height
    others_in_range: int = 0         # other enemy champions within TRADE_RANGE
    my_minions: int | None = None    # your side's minions within MINION_RANGE (None: not read)
    their_minions: int | None = None


@dataclass
class Trade:
    start: float
    end: float
    me_lost: float                   # health share
    opponent_lost: float
    started_by: str                  # "you" | "opponent" | "both"
    skirmish: bool
    minion_edge: int | None = None   # your minions minus theirs near you when it started

    @property
    def result(self) -> str:
        net = self.opponent_lost - self.me_lost
        return "won" if net > EVEN_MARGIN else "lost" if net < -EVEN_MARGIN else "even"


@dataclass
class LaneReport:
    samples: int
    trades: list[Trade] = field(default_factory=list)
    in_range_share: float = 0.0      # of samples with the opponent on screen
    median_distance: float | None = None

    def summary(self) -> dict:
        clean = [t for t in self.trades if not t.skirmish]
        count = lambda r: sum(t.result == r for t in clean)   # noqa: E731
        started = [t for t in clean if t.started_by == "you"]
        return {
            "trades": len(clean), "skirmishes": len(self.trades) - len(clean),
            "won": count("won"), "lost": count("lost"), "even": count("even"),
            "net_per_trade": round(100 * fmean(t.opponent_lost - t.me_lost for t in clean), 1)
            if clean else None,
            "you_started": len(started),
            "won_when_you_started": sum(t.result == "won" for t in started),
            "with_minion_advantage": _tally([t for t in clean if t.minion_edge is not None
                                             and t.minion_edge >= WAVE_EDGE]),
            "with_minion_disadvantage": _tally([t for t in clean if t.minion_edge is not None
                                                and t.minion_edge <= -WAVE_EDGE]),
            "in_range_share": round(self.in_range_share, 3),
            "median_distance": None if self.median_distance is None else round(self.median_distance, 3),
        }


def _tally(trades_: list[Trade]) -> dict:
    return {"trades": len(trades_), "won": sum(t.result == "won" for t in trades_),
            "lost": sum(t.result == "lost" for t in trades_)}


def frame_extra(frame_bgr) -> list[Bar]:
    """What laning needs from a frame besides champion bars: the minions. Module-level so
    video.scan can send it to its worker processes."""
    from riftwatch.vision.healthbars import find_minion_bars

    return find_minion_bars(frame_bgr)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def samples(rows: Iterable[tuple], followed_team: str | None = None,
            game_offset: float = 0.0) -> list[Sample]:
    """``rows`` are (t, width, height, bars, minions) from video.scan with
    ``extra=frame_extra`` (minions may be None: then wave counts are left out). The followed
    champion's bar
    is the "self" bar if there is one (your own recordings); otherwise the bar of
    ``followed_team`` nearest the top-centre (replays) -- learned by vote if not given."""
    out: list[Sample] = []
    votes: dict[str, int] = {}
    last_opp: tuple[float, float] | None = None
    for row in rows:
        t, w, h, bars = row[0], row[1], row[2], row[3]
        target = (w / 2, h / 2 - 0.12 * h)
        mine = [b for b in bars if b.team == "self"]
        if mine:
            me, my_team = min(mine, key=lambda b: _dist(b.center, target)), "ally"
        else:
            near = min(bars, key=lambda b: _dist(b.center, target), default=None)
            if near is not None:
                votes[near.team] = votes.get(near.team, 0) + 1
            team = followed_team or (max(votes, key=votes.get) if votes else None)
            pool = [b for b in bars if b.team == team]
            me = min(pool, key=lambda b: _dist(b.center, target), default=None)
            if me is not None and _dist(me.center, target) > 0.2 * h:
                me = None
            my_team = team
        enemy_team = {"ally": "enemy", "enemy": "ally"}.get(my_team or "", "enemy")
        if me is None:
            out.append(Sample(t + game_offset, None, None, None))
            continue
        enemies: list[Bar] = [b for b in bars if b.team == enemy_team]
        anchor = last_opp or me.center
        opp = min(enemies, key=lambda b: _dist(b.center, anchor), default=None)
        last_opp = opp.center if opp else None
        dist = _dist(opp.center, me.center) / h if opp else None
        others = sum(1 for b in enemies if b is not opp and _dist(b.center, me.center) / h <= TRADE_RANGE)
        minions = row[4] if len(row) > 4 else None
        mine_n = theirs_n = None
        if minions is not None:
            near = [m for m in minions if _dist(m.center, me.center) / h <= MINION_RANGE]
            mine_n = sum(m.team == my_team for m in near)
            theirs_n = sum(m.team == enemy_team for m in near)
        out.append(Sample(t + game_offset, me.fraction, opp.fraction if opp else None, dist,
                          others, mine_n, theirs_n))
    return out


def _smooth(values: list[float | None]) -> list[float | None]:
    """Median of each reading and its neighbours: one misread frame isn't a hit."""
    out = []
    for i, v in enumerate(values):
        window = [x for x in values[max(0, i - 1):i + 2] if x is not None]
        out.append(median(window) if v is not None and window else v)
    return out


def trades(seq: list[Sample], until: float = LANE_END_S) -> LaneReport:
    seq = [s for s in seq if s.t <= until]
    me = _smooth([s.me for s in seq])
    opp = _smooth([s.opponent for s in seq])
    report = LaneReport(len(seq))
    seen = [s for s in seq if s.distance is not None]
    if seen:
        report.in_range_share = sum(s.distance <= TRADE_RANGE for s in seen) / len(seen)
        report.median_distance = median(s.distance for s in seen)

    current: dict | None = None
    last_close: float | None = None
    for i in range(1, len(seq)):
        s = seq[i]
        if s.distance is not None and s.distance <= TRADE_RANGE:
            last_close = s.t
        close = last_close is not None and s.t - last_close <= QUIET_S
        me_drop = me[i] is not None and me[i - 1] is not None and me[i - 1] - me[i] >= DROP
        opp_drop = opp[i] is not None and opp[i - 1] is not None and opp[i - 1] - opp[i] >= DROP
        if current is None:
            if close and (me_drop or opp_drop):
                edge = (None if s.my_minions is None or s.their_minions is None
                        else s.my_minions - s.their_minions)
                current = {"i0": i - 1, "last": i, "first": "both" if me_drop and opp_drop
                           else "you" if opp_drop else "opponent",
                           "skirmish": s.others_in_range > 0, "edge": edge}
            continue
        if close and (me_drop or opp_drop):
            current["last"] = i
            current["skirmish"] |= s.others_in_range > 0
        if s.t - seq[current["last"]].t >= QUIET_S or i == len(seq) - 1:
            a, b = current["i0"], min(len(seq) - 1, current["last"] + 1)
            me_lost = _lost(me, a, b)
            opp_lost = _lost(opp, a, b)
            if me_lost is not None and opp_lost is not None:
                report.trades.append(Trade(seq[a].t, seq[current["last"]].t, me_lost, opp_lost,
                                           current["first"], current["skirmish"],
                                           current["edge"]))
            current = None
    return report


def _lost(values: list[float | None], a: int, b: int) -> float | None:
    """Health lost over a trade: before it, minus the lowest point during it."""
    before = next((values[i] for i in range(a, -1, -1) if values[i] is not None), None)
    during = [v for v in values[a:b + 1] if v is not None]
    if before is None or not during:
        return None
    return max(0.0, before - min(during))
