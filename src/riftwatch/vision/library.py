"""The video library: gameplay videos registered once, processed into laning readings and
trades, and turned into baselines and training data.

    add       register video files (metadata from the file name where it follows the
              "CHAMPION vs OPPONENT (ROLE) REGION Tier patch" pattern replay channels use)
    process   read each video's clock, scan its laning phase, store samples and trades
    stats     how the library's players trade, by role and champion
    dataset   one row per moment in trading range: situation, decision, outcome

Processing is versioned with ANALYZER_VERSION: when the analysis improves, bump it and
``process`` re-runs every video, so the training data never mixes old and new readings.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import psycopg

from riftwatch.vision import lane

ANALYZER_VERSION = 2       # 2: HUD panel and minimap fields
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER",
         "GRANDMASTER", "CHALLENGER")
ROLES = {"TOP": "TOP", "JUNGLE": "JUNGLE", "JG": "JUNGLE", "MID": "MIDDLE", "MIDDLE": "MIDDLE",
         "ADC": "BOTTOM", "BOT": "BOTTOM", "BOTTOM": "BOTTOM", "SUPPORT": "UTILITY",
         "SUP": "UTILITY", "SUPP": "UTILITY", "UTILITY": "UTILITY"}
_TITLE = re.compile(
    r"^(?P<champion>.+?)\s+vs\.?\s+(?P<opponent>.+?)\s*\((?P<role>[A-Za-z]+)\)"
    r"(?:\s+(?P<region>[A-Za-z]{2,4}))?"
    r"(?:\s+(?P<tier>" + "|".join(TIERS) + r"))?"
    r"(?:\s+(?P<patch>\d+\.\d+))?", re.IGNORECASE)


@dataclass
class VideoMeta:
    title: str
    champion: str | None = None
    opponent: str | None = None
    role: str | None = None
    region: str | None = None
    tier: str | None = None
    patch: str | None = None


def parse_title(name: str) -> VideoMeta:
    """Metadata from a file name such as "IRELIA vs RENEKTON (TOP) NA Grandmaster 26.1_720p".
    Anything that doesn't follow the pattern just has no metadata."""
    title = re.sub(r"[_ ]\d{3,4}p$", "", Path(name).stem).strip()
    m = _TITLE.match(title)
    if not m:
        return VideoMeta(title)

    def name_case(s: str) -> str:
        return " ".join(w.capitalize() for w in s.strip().split())

    role = ROLES.get(m["role"].upper())
    return VideoMeta(title, name_case(m["champion"]), name_case(m["opponent"]), role,
                     m["region"].upper() if m["region"] else None,
                     m["tier"].upper() if m["tier"] else None, m["patch"])


def fingerprint(path: Path, chunk: int = 1 << 20) -> str:
    """Size plus hashes of the first and last MB: cheap on multi-GB files, and the same
    video renamed or moved is still recognised."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(chunk, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def add(conn: psycopg.Connection, paths: list[Path], *, view: str = "spectator",
        match_id: str | None = None, overrides: dict | None = None) -> tuple[int, int]:
    """Register video files (folders are searched). Returns (added, already known); a known
    video that moved gets its new path."""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_SUFFIXES)
        elif p.suffix.lower() in VIDEO_SUFFIXES:
            files.append(p)
    added = known = 0
    for f in files:
        meta = parse_title(f.name)
        values = {**meta.__dict__, **{k: v for k, v in (overrides or {}).items() if v}}
        row = conn.execute(
            """
            INSERT INTO videos (path, fingerprint, title, champion, opponent, role, region,
                                tier, patch, view, match_id)
            VALUES (%(path)s, %(fp)s, %(title)s, %(champion)s, %(opponent)s, %(role)s,
                    %(region)s, %(tier)s, %(patch)s, %(view)s, %(match_id)s)
            ON CONFLICT (fingerprint) DO UPDATE SET path = EXCLUDED.path
            RETURNING (xmax = 0)
            """,
            {**values, "path": str(f.resolve()), "fp": fingerprint(f), "view": view,
             "match_id": match_id},
        ).fetchone()
        if row[0]:
            added += 1
        else:
            known += 1
    return added, known


def pending(conn: psycopg.Connection, limit: int | None = None) -> list[tuple[int, str]]:
    return conn.execute(
        """
        SELECT id, path FROM videos
         WHERE status = 'pending' OR analyzer_version IS DISTINCT FROM %s
         ORDER BY id LIMIT %s
        """,
        (ANALYZER_VERSION, limit),
    ).fetchall()


def process_one(conn: psycopg.Connection, video_id: int, path: str,
                progress: Callable[[str], None] | None = None) -> lane.LaneReport:
    """Clock, laning scan, samples and trades for one video, stored in one transaction."""
    import cv2

    from riftwatch.vision import clock, video

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"can't open {path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    cap.release()
    found = clock.video_offset(path)
    if found is None or found[1] < 0.5:
        raise ValueError("game clock not readable (agreement "
                         f"{0 if found is None else round(100 * found[1])}%)")
    offset, agreement = found
    (view,) = conn.execute("SELECT view FROM videos WHERE id = %s", (video_id,)).fetchone()
    extra = lane.frame_extra if view == "spectator" else lane.frame_extra_player
    rows = video.scan(path, fps=4, extra=extra, progress=progress,
                      end_s=max(0.0, lane.LANE_END_S - offset))
    seq = [s for s in lane.samples(rows, game_offset=offset) if s.t <= lane.LANE_END_S]
    report = lane.trades(seq)
    with conn.transaction():
        conn.execute("DELETE FROM video_samples WHERE video_id = %s", (video_id,))
        conn.execute("DELETE FROM video_trades WHERE video_id = %s", (video_id,))
        with conn.cursor() as cur:
            with cur.copy("COPY video_samples (video_id, t, me, opponent, distance, others, "
                          "my_minions, their_minions, mana, ready, level, map_x, map_y, depth, "
                          "in_base, dead) FROM STDIN") as copy:
                seen = set()
                for s in seq:
                    key = round(s.t, 3)
                    if key in seen:
                        continue
                    seen.add(key)
                    copy.write_row((video_id, key, s.me, s.opponent, s.distance,
                                    s.others_in_range, s.my_minions, s.their_minions, s.mana,
                                    s.ready, s.level, s.map_x, s.map_y, s.depth, s.in_base,
                                    s.dead))
            for t in report.trades:
                cur.execute(
                    """
                    INSERT INTO video_trades (video_id, start_s, end_s, me_lost, opponent_lost,
                                              started_by, skirmish, minion_edge, result, mana,
                                              ready, level, depth, died, back_after_s,
                                              opponent_left_low)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (video_id, round(t.start, 3), round(t.end, 3), t.me_lost, t.opponent_lost,
                     t.started_by, t.skirmish, t.minion_edge, t.result, t.mana, t.ready,
                     t.level, t.depth, t.died, t.back_after_s, t.opponent_left_low))
        conn.execute(
            """
            UPDATE videos SET width = %s, height = %s, fps = %s, duration_s = %s,
                   game_offset_s = %s, clock_agreement = %s, status = 'done', error = NULL,
                   analyzer_version = %s, processed_at = now()
             WHERE id = %s
            """,
            (width, height, fps, duration, offset, agreement, ANALYZER_VERSION, video_id))
    return report


def process(conn: psycopg.Connection, limit: int | None = None,
            progress: Callable[[str], None] | None = None) -> tuple[int, list[tuple[str, str]]]:
    """Process every pending or outdated video. One bad file is recorded, not raised."""
    done, failed = 0, []
    for video_id, path in pending(conn, limit):
        if progress:
            progress(f"processing {Path(path).name}")
        try:
            report = process_one(conn, video_id, path)
            done += 1
            if progress:
                s = report.summary()
                progress(f"  {s['trades']} trades, {s['skirmishes']} skirmishes")
        except Exception as exc:     # a corrupt or unreadable file mustn't stop the batch
            failed.append((path, f"{type(exc).__name__}: {exc}"))
            conn.execute("UPDATE videos SET status = 'failed', error = %s, analyzer_version = %s "
                         "WHERE id = %s", (failed[-1][1], ANALYZER_VERSION, video_id))
    return done, failed


@dataclass
class TradeStats:
    group: str
    videos: int
    lane_minutes: float
    trades: int
    won: int
    lost: int
    net_per_trade: float | None
    started_share: float | None              # share of trades the player started
    won_with_edge: tuple[int, int]           # (won, trades) with a bigger wave
    won_without_edge: tuple[int, int]        # (won, trades) with a smaller wave
    won_ultimate_ready: tuple[int, int] = (0, 0)   # (won, trades) with R up
    won_ultimate_down: tuple[int, int] = (0, 0)    # (won, trades) at 6+ with R on cooldown
    died_after: int = 0                      # trades followed by the player's death
    back_after: int = 0                      # ... by a trip back to base within 45 s

    @property
    def trades_per_10_min(self) -> float:
        return 10 * self.trades / self.lane_minutes if self.lane_minutes else 0.0

    @property
    def win_share(self) -> float | None:
        decided = self.won + self.lost
        return self.won / decided if decided else None


def stats(conn: psycopg.Connection, *, role: str | None = None, champion: str | None = None,
          tier: str | None = None, view: str = "spectator") -> TradeStats:
    """How the library's players trade in lane (clean trades, not skirmishes)."""
    where = ["v.status = 'done'", "v.view = %(view)s"]
    params: dict = {"view": view, "role": role, "champion": champion, "tier": tier}
    if role:
        where.append("v.role = %(role)s")
    if champion:
        where.append("lower(v.champion) = lower(%(champion)s)")
    if tier:
        where.append("v.tier = %(tier)s")
    clause = " AND ".join(where)
    videos = conn.execute(
        f"""
        SELECT v.id, (SELECT max(t) - min(t) FROM video_samples s WHERE s.video_id = v.id)
          FROM videos v WHERE {clause}
        """, params).fetchall()
    rows = conn.execute(
        f"""
        SELECT t.me_lost, t.opponent_lost, t.started_by, t.minion_edge, t.result,
               t.ready, t.level, t.died, t.back_after_s
          FROM video_trades t JOIN videos v ON v.id = t.video_id
         WHERE {clause} AND NOT t.skirmish
        """, params).fetchall()
    r_bit = 1 << lane.SLOTS.index("R")
    r_up = [r for r in rows if r[5] is not None and r[5] & r_bit]
    r_down = [r for r in rows if r[5] is not None and (r[6] or 0) >= 6 and not r[5] & r_bit]
    group = " ".join(x for x in (tier, role, champion) if x) or "all"
    edge = [r for r in rows if r[3] is not None and r[3] >= lane.WAVE_EDGE]
    behind = [r for r in rows if r[3] is not None and r[3] <= -lane.WAVE_EDGE]
    return TradeStats(
        group, len(videos), sum(m or 0 for _, m in videos) / 60, len(rows),
        sum(r[4] == "won" for r in rows), sum(r[4] == "lost" for r in rows),
        round(100 * fmean(r[1] - r[0] for r in rows), 1) if rows else None,
        sum(r[2] == "you" for r in rows) / len(rows) if rows else None,
        (sum(r[4] == "won" for r in edge), len(edge)),
        (sum(r[4] == "won" for r in behind), len(behind)),
        (sum(r[4] == "won" for r in r_up), len(r_up)),
        (sum(r[4] == "won" for r in r_down), len(r_down)),
        sum(bool(r[7]) for r in rows), sum(r[8] is not None for r in rows))


MIN_LIBRARY_VIDEOS = 5     # high-elo games in a role before the coach compares against them


@dataclass
class PlayerLane:
    """A player's laning from their own processed video of one match."""
    video_id: int
    role: str | None
    trades: list[tuple]                      # (start, me_lost, opp_lost, started_by, edge, result)
    situations: list[dict]                   # learn features at each trade the player started

    def summary(self) -> dict:
        n = len(self.trades)
        return {"trades": n, "won": sum(t[5] == "won" for t in self.trades),
                "died_after": sum(bool(t[6]) for t in self.trades if len(t) > 6),
                "back_after": sum(t[7] is not None for t in self.trades if len(t) > 7),
                "lost": sum(t[5] == "lost" for t in self.trades),
                "even": sum(t[5] == "even" for t in self.trades),
                "net": round(100 * fmean(t[2] - t[1] for t in self.trades), 1) if n else None,
                "started": sum(t[3] in ("you", "both") for t in self.trades)}


def player_lane(conn: psycopg.Connection, match_id: str) -> PlayerLane | None:
    """The player's processed video of ``match_id`` (view "player"), if there is one."""
    from riftwatch.vision import learn

    row = conn.execute(
        "SELECT id, role FROM videos WHERE match_id = %s AND view = 'player' AND status = 'done' "
        "ORDER BY processed_at DESC LIMIT 1", (match_id,)).fetchone()
    if row is None:
        return None
    video_id, role = row
    trades = conn.execute(
        """SELECT start_s, me_lost, opponent_lost, started_by, minion_edge, result, died,
                  back_after_s
             FROM video_trades WHERE video_id = %s AND NOT skirmish ORDER BY start_s""",
        (video_id,)).fetchall()
    situations = []
    for start, *_rest in [t for t in trades if t[3] in ("you", "both")]:
        s = conn.execute(
            """SELECT t, me, opponent, distance, others, my_minions, their_minions, mana,
                      ready, level, depth
                 FROM video_samples WHERE video_id = %s AND t < %s AND me IS NOT NULL
                  AND opponent IS NOT NULL AND distance IS NOT NULL
                ORDER BY t DESC LIMIT 1""", (video_id, start)).fetchone()
        if s is not None:
            t, me, opp, dist, others, mine, theirs, mana, ready, level, dep = s
            situations.append(learn._features(t, me, opp, dist, mine, theirs, others, role,
                                              mana=mana, ready=ready, level=level, depth=dep))
    return PlayerLane(video_id, role, [tuple(t) for t in trades], situations)
