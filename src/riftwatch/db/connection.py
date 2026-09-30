from __future__ import annotations

import psycopg


def connect(database_url: str, timeout_s: int = 5) -> psycopg.Connection:
    # Without a timeout a stopped database makes every command hang instead of failing.
    return psycopg.connect(database_url, autocommit=True, connect_timeout=timeout_s)
