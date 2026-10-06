"""Guards for a public site: per-visitor limits on the actions that cost something (Riot
requests, LLM calls), and a daily cap on what the LLM coach may spend.

Counters live in Postgres so every server process enforces the same limit. Each is a fixed
window: the first request in a window creates the row, later ones increment it.
"""

from __future__ import annotations

import psycopg

WINDOW_S = 3600
# Requests per visitor per hour. Update and scouting cost Riot requests; coaching costs money.
DEFAULT_LIMITS = {"sync": 30, "scout": 60, "coach": 10}
DEFAULT_COACH_BUDGET_USD = 5.0


def parse_limits(text: str | None) -> dict[str, int]:
    """``"sync=30,coach=10"`` -> {"sync": 30, "coach": 10}, on top of the defaults."""
    limits = dict(DEFAULT_LIMITS)
    for part in (text or "").split(","):
        if not part.strip():
            continue
        name, sep, value = part.partition("=")
        name = name.strip()
        if not sep or name not in DEFAULT_LIMITS or not value.strip().isdigit():
            raise ValueError(f"bad rate limit {part!r}; expected e.g. sync=30,scout=60,coach=10")
        limits[name] = int(value)
    return limits


def hit(conn: psycopg.Connection, key: str, limit: int, window_s: int = WINDOW_S) -> float | None:
    """Count one request for ``key``. Returns None if it's allowed, otherwise the seconds
    until the window resets."""
    count, remaining = conn.execute(
        """
        INSERT INTO rate_limits (key, window_start, count)
        VALUES (%(key)s, to_timestamp(floor(extract(epoch FROM now()) / %(w)s) * %(w)s), 1)
        ON CONFLICT (key, window_start) DO UPDATE SET count = rate_limits.count + 1
        RETURNING count,
                  extract(epoch FROM window_start + make_interval(secs => %(w)s) - now())
        """,
        {"key": key, "w": window_s},
    ).fetchone()
    if count == 1:      # first request of a window: drop windows that are long over
        conn.execute("DELETE FROM rate_limits WHERE window_start < now() - interval '1 day'")
    return None if count <= limit else max(1.0, float(remaining))


def coach_spend_today(conn: psycopg.Connection) -> float:
    """US dollars the LLM coach has spent since midnight UTC (from stored coaching)."""
    row = conn.execute(
        """
        SELECT coalesce(sum((usage->>'cost_usd')::float8), 0) FROM coach_reports
         WHERE created_at >= date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
        """
    ).fetchone()
    return float(row[0])
