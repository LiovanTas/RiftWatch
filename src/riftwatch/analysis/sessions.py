"""Session patterns: does the player's play change the longer they keep playing, or after
losses?

Games are grouped into sessions: a break of SESSION_GAP_S or more between one game ending
and the next starting begins a new one. Each game is summarised the usual way (mean "better
than N%" over its coachable stats, against the player's rank, adjusted for champion and
matchup), then grouped by its place in the session and by the run of losses before it in
the same session.

Patterns are judged *within* sessions: for every session that has both kinds of game (a
first game and a fourth-or-later game; a game right after a win and one right after two or
more losses), the difference between them, averaged over those sessions. Comparing all first
games with all late games instead would mix in which days had long sessions -- on real data
that pooled gap was 7 points while the within-session one was 2, inside the noise, because
long sessions had become more common as the player's form dipped. A pattern is reported only
with MIN_SESSIONS such sessions and a mean difference beyond CLEAR_GAP_SE standard errors.
It is still an association, not proof of cause.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta

SESSION_GAP_S = 3600
MIN_SESSIONS = 10
CLEAR_GAP_SE = 2.0
LATE_GAME = 4            # "late in a session" = this game or later


@dataclass
class Group:
    label: str
    games: int
    wins: int
    score: float          # mean per-game "better than N%"
    se: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


@dataclass
class Paired:
    """Mean within-session difference (later group minus earlier) across sessions."""
    sessions: int
    gap: float
    se: float

    @property
    def clear(self) -> bool:
        return self.sessions >= MIN_SESSIONS and abs(self.gap) > CLEAR_GAP_SE * self.se


@dataclass
class Finding:
    kind: str             # "session_length" | "after_losses"
    text: str             # evidence wording, numbers as written in the groups
    worse: bool           # the later / after-losses group is the worse one
    gap: float


@dataclass
class SessionReport:
    games: int
    sessions: int
    by_position: list[Group]
    by_streak: list[Group]
    late_vs_first: Paired | None = None          # 4th+ game minus the session's first
    losses_vs_win: Paired | None = None          # after 2+ losses minus after a win
    findings: list[Finding] = field(default_factory=list)

    @property
    def games_per_session(self) -> float:
        return self.games / self.sessions if self.sessions else 0.0


def _group(label: str, rows: list[tuple[bool, float]]) -> Group:
    scores = [s for _, s in rows]
    n = len(rows)
    return Group(label, n, sum(w for w, _ in rows),
                 round(statistics.fmean(scores), 1) if n else 0.0,
                 round(statistics.stdev(scores) / n ** 0.5, 1) if n > 1 else 0.0)


def _paired(pairs: list[tuple[list[float], list[float]]]) -> Paired | None:
    diffs = [statistics.fmean(later) - statistics.fmean(earlier)
             for earlier, later in pairs if earlier and later]
    if len(diffs) < 2:
        return None
    return Paired(len(diffs), round(statistics.fmean(diffs), 1),
                  round(statistics.stdev(diffs) / len(diffs) ** 0.5, 1))


def summarize(games: list[tuple[datetime, int, bool, float]]) -> SessionReport:
    """``games`` is (start, duration in seconds, won, per-game score), in any order."""
    games = sorted(games, key=lambda g: g[0])
    position: dict[str, list[tuple[bool, float]]] = {k: [] for k in ("1st", "2nd", "3rd", "4th+")}
    streak: dict[str, list[tuple[bool, float]]] = {k: [] for k in ("after a win", "after 1 loss",
                                                                   "after 2+ losses")}
    sessions, index, losses, prev_end = 0, 0, 0, None
    # Per session: first game, 4th+ games, games after a win, games after 2+ losses.
    by_session: list[dict[str, list[float]]] = []
    for start, duration, won, score in games:
        if prev_end is None or (start - prev_end).total_seconds() >= SESSION_GAP_S:
            sessions += 1
            index, losses = 0, 0
            by_session.append({"first": [], "late": [], "win": [], "losses": []})
        index += 1
        here = by_session[-1]
        if index == 1:
            here["first"].append(score)
        elif index >= LATE_GAME:
            here["late"].append(score)
        if index > 1 and losses == 0:
            here["win"].append(score)
        elif losses >= 2:
            here["losses"].append(score)
        position[("1st", "2nd", "3rd")[index - 1] if index < LATE_GAME else "4th+"].append((won, score))
        if index > 1:       # the first game of a session follows a break, not a result
            key = "after a win" if losses == 0 else "after 1 loss" if losses == 1 else "after 2+ losses"
            streak[key].append((won, score))
        losses = 0 if won else losses + 1
        prev_end = start + timedelta(seconds=duration)

    by_position = [_group(k, v) for k, v in position.items()]
    by_streak = [_group(k, v) for k, v in streak.items()]
    report = SessionReport(
        len(games), sessions, by_position, by_streak,
        late_vs_first=_paired([(s["first"], s["late"]) for s in by_session]),
        losses_vs_win=_paired([(s["win"], s["losses"]) for s in by_session]))

    lf = report.late_vs_first
    if lf is not None and lf.clear:
        report.findings.append(Finding(
            "session_length",
            f"Across the last {len(games)} games, within the same session (a session ends at a "
            f"break of an hour or more): from the fourth game on, the player's games averaged "
            f"{abs(lf.gap):.0f} points {'lower' if lf.gap < 0 else 'higher'} on the "
            f"better-than-comparable-players scale than that session's first game, over "
            f"{lf.sessions} sessions of four or more games.", worse=lf.gap < 0, gap=lf.gap))
    lw = report.losses_vs_win
    if lw is not None and lw.clear:
        report.findings.append(Finding(
            "after_losses",
            f"Across the last {len(games)} games, within the same session: games right after "
            f"two or more losses in a row averaged {abs(lw.gap):.0f} points "
            f"{'lower' if lw.gap < 0 else 'higher'} on the better-than-comparable-players scale "
            f"than games right after a win, over {lw.sessions} sessions that had both.",
            worse=lw.gap < 0, gap=lw.gap))
    return report
