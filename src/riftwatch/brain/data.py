"""Situations from the video library, with the label for every head, and the checks a video's
readings must pass before the brain learns from it.

A situation is a moment when both health bars are read, the two laners are within trading
range, neither is in a trade and the player isn't dead, before laning ends (14:00). Its labels
look ahead from that moment: whether a trade starts within DECISION_S and who started it; for
a trade the player started, how it went; how the health difference moved over SWING_S; and
whether the player died within DIED_S or was back in base within BACK_S. A label whose window
the video doesn't fully show (the end of the recording, a reading the video can't make) is
left unknown, and that situation simply doesn't count for that head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psycopg

from riftwatch.brain import features
from riftwatch.vision import lane

DECISION_S = 1.0          # a trade starting within this long is the decision taken here
SWING_S = 10.0
DIED_S = 15.0
BACK_S = 45.0
TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER",
         "GRANDMASTER", "CHALLENGER")
LABELS = ("trade", "traded_on", "trade_won", "trade_net", "swing_10s", "died_15s", "back_45s")

# A video the brain trains on must have ...
MIN_LANE_MINUTES = 5.0    # ... this much laning read,
MIN_COVERAGE = 0.4        # ... its player's own bar read in this share of its moments (the
                          # opponent is often off screen, so theirs says little about reading),
MIN_CLOCK = 0.6           # ... and its game clock agreed on by this share of readings.

_SAMPLE_COLS = ("t", "me", "opponent", "distance", "others", "my_minions", "their_minions",
                "mana", "ready", "level", "depth", "in_base", "dead")
_TRADE_COLS = ("start_s", "end_s", "me_lost", "opponent_lost", "started_by", "skirmish",
               "result", "died", "back_after_s")


@dataclass
class VideoCheck:
    video_id: int
    title: str
    role: str | None
    tier: str | None
    lane_minutes: float
    coverage: float                  # share of moments with the player's own bar read
    in_range: int                    # moments within trading range
    trades: int
    clock_agreement: float | None
    problems: list[str] = field(default_factory=list)    # reasons not to train on it
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check(conn: psycopg.Connection, view: str = "spectator") -> list[VideoCheck]:
    """Every processed video of ``view``, with anything that should keep the brain from
    learning from it."""
    from riftwatch.vision.library import ANALYZER_VERSION

    rows = conn.execute(
        """
        SELECT v.id, v.title, v.role, v.tier, v.clock_agreement, v.analyzer_version,
               coalesce((max(s.t) - min(s.t)) / 60, 0),
               coalesce(avg((s.me IS NOT NULL)::int), 0),
               count(*) FILTER (WHERE s.distance <= %s),
               (SELECT count(*) FROM video_trades t WHERE t.video_id = v.id)
          FROM videos v LEFT JOIN video_samples s ON s.video_id = v.id
         WHERE v.status = 'done' AND v.view = %s
         GROUP BY v.id ORDER BY v.id
        """, (lane.TRADE_RANGE, view)).fetchall()
    out = []
    for vid, title, role, tier, clock, version, minutes, coverage, in_range, trades in rows:
        c = VideoCheck(vid, title, role, tier, float(minutes), float(coverage), int(in_range),
                       int(trades), clock)
        if version != ANALYZER_VERSION:
            c.problems.append("processed by an older analyser (run: riftwatch videos process)")
        if c.lane_minutes < MIN_LANE_MINUTES:
            c.problems.append(f"only {c.lane_minutes:.1f} minutes of laning read")
        if c.coverage < MIN_COVERAGE:
            c.problems.append(f"the player's own health bar read in only {100 * c.coverage:.0f}% "
                              "of moments")
        if clock is not None and clock < MIN_CLOCK:
            c.problems.append(f"game clock uncertain ({100 * clock:.0f}% of readings agree)")
        if not in_range:
            c.problems.append("never within trading range of the opponent")
        if role is None:
            c.warnings.append("role unknown (put it in the file name, e.g. '(TOP)')")
        if not trades and in_range:
            c.warnings.append("no trades found")
        out.append(c)
    # A trade rate far outside the rest suggests misread bars (with enough videos to judge).
    rates = [c.trades / c.lane_minutes * 10 for c in out if c.ok and c.lane_minutes]
    if len(rates) >= 8:
        q1, q3 = np.percentile(rates, [25, 75])
        hi = q3 + 3 * (q3 - q1)
        for c in out:
            if c.ok and c.lane_minutes and c.trades / c.lane_minutes * 10 > hi:
                c.warnings.append(f"{c.trades / c.lane_minutes * 10:.0f} trades per 10 minutes, "
                                  "far above the rest (check the bar readings)")
    return out


def _videos(conn, view: str | None, video_ids: list[int] | None,
            min_tier: str | None) -> pd.DataFrame:
    where, params = ["status = 'done'"], []
    if view is not None:
        where.append("view = %s")
        params.append(view)
    if video_ids is not None:
        where.append("id = ANY(%s)")
        params.append(list(video_ids))
    rows = conn.execute(
        f"SELECT id, role, champion, opponent, tier, view FROM videos WHERE {' AND '.join(where)} "
        "ORDER BY id", params).fetchall()
    df = pd.DataFrame(rows, columns=["video_id", "role", "champion", "opponent", "tier", "view"])
    if min_tier:
        floor = TIERS.index(min_tier.upper())
        df = df[df["tier"].map(lambda t: t in TIERS and TIERS.index(t) >= floor)]
    return df


def _numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c].astype(object).where(df[c].notna(), np.nan), errors="coerce")
    return df


def readings(conn: psycopg.Connection, video_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All samples and trades of the given videos, missing values as NaN."""
    samples = pd.DataFrame(conn.execute(
        f"SELECT video_id, {', '.join(_SAMPLE_COLS)} FROM video_samples "
        "WHERE video_id = ANY(%s) ORDER BY video_id, t", (video_ids,)).fetchall(),
        columns=["video_id", *_SAMPLE_COLS])
    samples = _numeric(samples, _SAMPLE_COLS)
    trades = pd.DataFrame(conn.execute(
        f"SELECT video_id, {', '.join(_TRADE_COLS)} FROM video_trades "
        "WHERE video_id = ANY(%s) ORDER BY video_id, start_s", (video_ids,)).fetchall(),
        columns=["video_id", *_TRADE_COLS])
    trades = _numeric(trades, ("start_s", "end_s", "me_lost", "opponent_lost", "back_after_s"))
    return samples, trades


def labels(s: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Every head's label at every reading of one video (NaN where unknown)."""
    t = s["t"].to_numpy(float)
    n = len(t)
    last_t = t[-1] if n else 0.0
    out = {k: np.full(n, np.nan) for k in LABELS}

    starts = trades["start_s"].to_numpy(float) if len(trades) else np.array([])
    j = np.searchsorted(starts, t, side="right")             # first trade starting after t
    has = (j < len(starts))
    has[has] = starts[j[has]] <= t[has] + DECISION_S
    by = trades["started_by"].to_numpy(object) if len(trades) else np.array([], object)
    mine = np.zeros(n, bool)
    theirs = np.zeros(n, bool)
    mine[has] = np.isin(by[j[has]], ("you", "both"))
    theirs[has] = by[j[has]] == "opponent"
    out["trade"] = mine.astype(float)
    out["traded_on"] = theirs.astype(float)
    if len(trades):
        clean = mine.copy()
        clean[mine] = ~trades["skirmish"].to_numpy(bool)[j[mine]]
        won = (trades["result"].to_numpy(object) == "won").astype(float)
        net = trades["opponent_lost"].to_numpy(float) - trades["me_lost"].to_numpy(float)
        out["trade_won"][clean] = won[j[clean]]
        out["trade_net"][clean] = net[j[clean]]

    dead = np.nan_to_num(s["dead"].to_numpy(float)) > 0
    in_base = np.nan_to_num(s["in_base"].to_numpy(float)) > 0
    has_panel = bool(s["mana"].notna().any() or dead.any())
    has_map = bool(s["in_base"].notna().any())
    dead_cum = np.concatenate(([0], np.cumsum(dead)))
    base_cum = np.concatenate(([0], np.cumsum(in_base)))
    idx = np.arange(n)

    def window(seconds: float) -> np.ndarray:
        """Index just past the last reading within ``seconds`` after each moment."""
        return np.searchsorted(t, t + seconds, side="right")

    # Health swing: the difference SWING_S later (a death counts as zero health), unless
    # the player went back to base (health refills) or nothing was read then.
    me_f = features._ffill(t, s["me"].to_numpy(float), 2.0)
    opp_f = features._ffill(t, s["opponent"].to_numpy(float), 2.0)
    end = window(SWING_S)
    later = end - 1
    full = t + SWING_S <= last_t
    died_by = dead_cum[end] - dead_cum[idx + 1] > 0
    based_by = base_cum[end] - base_cum[idx + 1] > 0
    me_later = np.where(died_by, 0.0, me_f[later])
    opp_later = np.where(np.isnan(opp_f[later]), opp_f[idx], opp_f[later])
    swing = (me_later - opp_later) - (s["me"].to_numpy(float) - s["opponent"].to_numpy(float))
    out["swing_10s"] = np.where(full & ~(based_by & ~died_by), swing, np.nan)

    if has_panel:
        end = window(DIED_S)
        out["died_15s"] = np.where(t + DIED_S <= last_t,
                                   (dead_cum[end] - dead_cum[idx + 1] > 0).astype(float), np.nan)
    if has_map:
        end = window(BACK_S)
        out["back_45s"] = np.where(t + BACK_S <= last_t,
                                   (base_cum[end] - base_cum[idx + 1] > 0).astype(float), np.nan)
    return pd.DataFrame(out, index=s.index)


def situations_mask(s: pd.DataFrame, trades: pd.DataFrame) -> np.ndarray:
    t = s["t"].to_numpy(float)
    mask = np.array(s["me"].notna() & s["opponent"].notna()
                    & (s["distance"].to_numpy(float) <= lane.TRADE_RANGE), dtype=bool)
    mask &= ~(np.nan_to_num(s["dead"].to_numpy(float)) > 0)
    mask &= t <= lane.LANE_END_S
    for start, end in zip(trades["start_s"].to_numpy(float), trades["end_s"].to_numpy(float),
                          strict=True):
        mask &= ~((t >= start) & (t <= end))
    return mask


def load(conn: psycopg.Connection, *, view: str | None = "spectator",
         video_ids: list[int] | None = None, min_tier: str | None = None,
         checked: bool = True) -> pd.DataFrame:
    """One row per situation: video_id, t, view, tier, every feature and every label.
    ``checked`` leaves out videos that fail ``check``."""
    videos = _videos(conn, view, video_ids, min_tier)
    if checked and len(videos):
        bad = {c.video_id for v in videos["view"].unique() for c in check(conn, v) if not c.ok}
        videos = videos[~videos["video_id"].isin(bad)]
    if videos.empty:
        return pd.DataFrame(columns=["video_id", "t", "view", "tier",
                                     *features.VARIANTS["full"], "champion",
                                     "opponent_champion", *LABELS])
    samples, trades = readings(conn, videos["video_id"].tolist())
    trades_by = {vid: g.reset_index(drop=True) for vid, g in trades.groupby("video_id")}
    meta = videos.set_index("video_id")
    empty = pd.DataFrame(columns=list(_TRADE_COLS))
    parts = []
    for vid, s in samples.groupby("video_id", sort=True):
        s = s.reset_index(drop=True)
        tr = trades_by.get(vid, empty)
        m = meta.loc[vid]
        mask = situations_mask(s, tr)
        if not mask.any():
            continue
        f = features.derive(s, tr, m["role"], m["champion"], m["opponent"])
        lab = labels(s, tr)
        part = pd.concat([f, lab], axis=1)[mask]
        part.insert(0, "video_id", int(vid))
        part.insert(1, "t", s["t"].to_numpy(float)[mask])
        part.insert(2, "view", m["view"])
        part.insert(3, "tier", m["tier"])
        parts.append(part)
    if not parts:
        return load(conn, video_ids=[])
    return pd.concat(parts, ignore_index=True)
