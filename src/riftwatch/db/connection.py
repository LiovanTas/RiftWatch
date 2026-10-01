from __future__ import annotations

import orjson
import psycopg
from psycopg.types.json import set_json_dumps, set_json_loads

# jsonb in and out through orjson instead of the stdlib json module: decoding a ~200 KB
# timeline is several times faster, and raw Riot JSON is the bulk of what we store.
set_json_loads(orjson.loads)
set_json_dumps(orjson.dumps)


def connect(database_url: str, timeout_s: int = 5) -> psycopg.Connection:
    # Without a timeout a stopped database makes every command hang instead of failing.
    return psycopg.connect(database_url, autocommit=True, connect_timeout=timeout_s)
