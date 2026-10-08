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

Replays add the followed champion's HUD panel (mana, which abilities and summoner spells are
ready, level; see vision.panel) and its place on the map from the minimap (vision.minimap). So
each moment also records those, and each trade records the state it started in and what came
after it: a death within DIED_WITHIN_S, a trip back to base within BACK_WITHIN_S (a recall or a
death), and whether the opponent left the lane low.

Health is read from bars, about two health points off at the 90th percentile on real footage
(see vision.spectator); shields, heals and regeneration mid-trade show up as they look on
screen. Distances are screen pixels, so they shrink when a replay zooms out.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import fmean, median

from riftwatch.vision.healthbars import Bar
from riftwatch.vision.panel import PanelReading

TRADE_RANGE = 0.35       # max distance between bars, as a share of frame height
MINION_RANGE = 0.5       # minions this close (share of frame height) count as "your wave"
WAVE_EDGE = 2            # this many more minions than the opponent is a minion advantage
DROP = 0.03              # a health drop of 3 points between samples starts or extends a trade
QUIET_S = 2.0            # a trade ends after this long without either side losing health
EVEN_MARGIN = 0.05       # health-point difference below which a trade is even
LANE_END_S = 14 * 60     # game time when laning is over
DIED_WITHIN_S = 10.0     # a death this soon after a trade counts as its consequence
BACK_WITHIN_S = 45.0     # back in base this soon after a trade's start: it cost a recall
LEFT_LOW = 0.2           # opponent this low at a trade's end and then gone: left the lane low
GONE_S = 8.0             # gone: out of range or out of view from half this after it to this
POSITION_HOLD_S = 3.0    # a missed camera reading takes the last one for this long
LEVEL_KNOWN_BEFORE = 100.0   # levels count from 1 only if the video starts before this
SLOTS = "QWERDF"         # ready bits, in this order (D and F are the summoner spells)


@dataclass
class Sample:
    t: float                         # game time, seconds
    me: float | None                 # followed champion's health share
    opponent: float | None
    distance: float | None           # px between the two bars, as a share of frame height
    others_in_range: int = 0         # other enemy champions within TRADE_RANGE
    my_minions: int | None = None    # your side's minions within MINION_RANGE (None: not read)
    their_minions: int | None = None
    mana: float | None = None        # from the HUD panel (replays)
    ready: int | None = None         # bit i set when SLOTS[i] is ready
    level: int | None = None
    map_x: float | None = None       # map position, 0..1, blue base at (0, 0)
    map_y: float | None = None
    depth: float | None = None       # -1 own base .. +1 enemy base
    in_base: bool | None = None
    dead: bool = False

    def is_ready(self, slot: str) -> bool | None:
        return None if self.ready is None else bool(self.ready >> SLOTS.index(slot) & 1)


@dataclass
class Trade:
    start: float
    end: float
    me_lost: float                   # health share
    opponent_lost: float
    started_by: str                  # "you" | "opponent" | "both"
    skirmish: bool
    minion_edge: int | None = None   # your minions minus theirs near you when it started
    # The state it started in (just before the first hit) ...
    mana: float | None = None
    ready: int | None = None
    level: int | None = None
    depth: float | None = None
    # ... and what came after it.
    died: bool | None = None         # the followed champion died within DIED_WITHIN_S
    back_after_s: float | None = None    # seconds from its start to being back in base
    opponent_left_low: bool | None = None

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
            "with_ultimate_ready": _tally([t for t in clean if t.ready is not None
                                           and t.ready >> SLOTS.index("R") & 1]),
            "with_ultimate_down": _tally([t for t in clean if t.ready is not None and t.level
                                          and t.level >= 6 and not t.ready >> SLOTS.index("R") & 1]),
            "died_after": sum(bool(t.died) for t in clean),
            "back_after": sum(t.back_after_s is not None for t in clean),
            "opponent_left_low": sum(bool(t.opponent_left_low) for t in clean),
            "in_range_share": round(self.in_range_share, 3),
            "median_distance": None if self.median_distance is None else round(self.median_distance, 3),
        }


def _tally(trades_: list[Trade]) -> dict:
    return {"trades": len(trades_), "won": sum(t.result == "won" for t in trades_),
            "lost": sum(t.result == "lost" for t in trades_)}


@dataclass
class FrameExtra:
    """What laning needs from a frame besides champion bars."""
    minions: list[Bar]
    panel: PanelReading | None = None
    camera: tuple[float, float] | None = None


def frame_extra(frame_bgr) -> FrameExtra:
    """Minions, the HUD panel and the minimap camera. Module-level so video.scan can send it
    to its worker processes."""
    from riftwatch.vision import minimap, panel
    from riftwatch.vision.healthbars import find_minion_bars

    return FrameExtra(find_minion_bars(frame_bgr), panel.read(frame_bgr), minimap.camera(frame_bgr))


def frame_extra_player(frame_bgr) -> FrameExtra:
    """For your own recordings: just the minions. Their HUD is the in-game one, not the
    replay's panel, and the camera there needn't be on your champion, so the panel and
    minimap readers don't apply."""
    from riftwatch.vision.healthbars import find_minion_bars

    return FrameExtra(find_minion_bars(frame_bgr))


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
    rows_list = list(rows)
    for row in rows_list:
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
        if isinstance(minions, FrameExtra):
            minions = minions.minions
        mine_n = theirs_n = None
        if minions is not None:
            near = [m for m in minions if _dist(m.center, me.center) / h <= MINION_RANGE]
            mine_n = sum(m.team == my_team for m in near)
            theirs_n = sum(m.team == enemy_team for m in near)
        out.append(Sample(t + game_offset, me.fraction, opp.fraction if opp else None, dist,
                          others, mine_n, theirs_n))
    _add_hud(out, [row[4] if len(row) > 4 and isinstance(row[4], FrameExtra) else None
                   for row in rows_list], my_team_votes=votes)
    return out


def _add_hud(out: list[Sample], extras: list, my_team_votes: dict[str, int]) -> None:
    """Fill each sample's panel and map fields. Readiness and levels need the whole video
    (each slot's bright level; counting level-ups from the start), so this runs after."""
    from riftwatch.vision import minimap, panel

    if not any(extras):
        return
    readings = [e.panel if e is not None else None for e in extras]
    have = [r for r in readings if r is not None]
    ready = panel.readiness(have) if have else []
    it = iter(ready)
    ready_all = [next(it) if r is not None else None for r in readings]
    xp = [r.xp if r is not None else None for r in readings]
    levels = panel.levels([s.t for s in out], xp)
    first_xp = next((s.t for s, x in zip(out, xp, strict=True) if x is not None), None)
    level_known = first_xp is not None and first_xp <= LEVEL_KNOWN_BEFORE

    # Which side the followed champion is on: replays colour the followed team's bars red or
    # blue; your own videos colour yours green, so then it's the base the camera starts in.
    side = None
    if my_team_votes:
        top = max(my_team_votes, key=my_team_votes.get)
        side = {"enemy": "red", "ally": "blue"}.get(top)
    if side is None:
        for e in extras:
            if e is not None and e.camera is not None:
                for team in ("blue", "red"):
                    if minimap.in_base(e.camera, team):
                        side = team
                break

    last_pos, last_t = None, None
    for s, e, r, rd, lv in zip(out, extras, readings, ready_all, levels, strict=True):
        if r is not None:
            s.dead = r.hp is None and r.mana is None and r.xp is None
            s.mana = r.mana
            if rd is not None and not s.dead:
                s.ready = sum(1 << i for i, k in enumerate(SLOTS) if rd.get(k))
            s.level = lv if level_known else None
        pos = None if e is None or s.dead else e.camera
        if pos is None and last_pos is not None and not s.dead and s.t - last_t <= POSITION_HOLD_S:
            pos = last_pos
        elif pos is not None:
            last_pos, last_t = pos, s.t
        if s.dead:
            last_pos = None
        if pos is not None:
            s.map_x, s.map_y = round(pos[0], 3), round(pos[1], 3)
            if side is not None:
                s.depth = round(minimap.depth(pos, side), 3)
                s.in_base = minimap.in_base(pos, side)


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
                before = seq[i - 1]
                current = {"i0": i - 1, "last": i, "first": "both" if me_drop and opp_drop
                           else "you" if opp_drop else "opponent",
                           "skirmish": s.others_in_range > 0, "edge": edge,
                           "state": (before.mana, before.ready, before.level, before.depth)}
            continue
        if close and (me_drop or opp_drop):
            current["last"] = i
            current["skirmish"] |= s.others_in_range > 0
        if s.t - seq[current["last"]].t >= QUIET_S or i == len(seq) - 1:
            a, b = current["i0"], min(len(seq) - 1, current["last"] + 1)
            me_lost = _lost(me, a, b)
            opp_lost = _lost(opp, a, b)
            if me_lost is not None and opp_lost is not None:
                trade = Trade(seq[a].t, seq[current["last"]].t, me_lost, opp_lost,
                              current["first"], current["skirmish"], current["edge"],
                              *current["state"])
                _consequences(trade, seq, opp, current["last"])
                report.trades.append(trade)
            current = None
    return report


def _consequences(trade: Trade, seq: list[Sample], opp: list[float | None], last: int) -> None:
    """What followed a trade, from the moments after it (None where the video doesn't say)."""
    after = [s for s in seq if trade.end < s.t <= trade.end + DIED_WITHIN_S]
    if any(s.mana is not None or s.dead for s in seq):   # the panel is read: deaths show
        trade.died = any(s.dead for s in after)
    if any(s.in_base is not None for s in seq):
        back = next((s for s in seq if trade.start < s.t <= trade.start + BACK_WITHIN_S
                     and s.in_base), None)
        trade.back_after_s = None if back is None else round(back.t - trade.start, 1)
    last_opp = next((opp[i] for i in range(last, -1, -1) if opp[i] is not None), None)
    if last_opp is not None:
        window = [s for s in seq if trade.end + GONE_S / 2 < s.t <= trade.end + GONE_S]
        watched = [s for s in window if s.me is not None]
        if watched:
            gone = all(s.distance is None or s.distance > TRADE_RANGE for s in watched)
            trade.opponent_left_low = last_opp <= LEFT_LOW and gone


def _lost(values: list[float | None], a: int, b: int) -> float | None:
    """Health lost over a trade: before it, minus the lowest point during it."""
    before = next((values[i] for i in range(a, -1, -1) if values[i] is not None), None)
    during = [v for v in values[a:b + 1] if v is not None]
    if before is None or not during:
        return None
    return max(0.0, before - min(during))
