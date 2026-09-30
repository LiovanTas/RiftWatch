"""Minimal forward-only migration runner.

Migrations are ``NNN_name.sql`` files in ``db/migrations``, applied in filename order. One
``migrate`` run is a single transaction (Postgres DDL is transactional), so if any migration
fails the database is left exactly as it was before the run. Applied files must never be edited; add a new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")

# Arbitrary constant: serialises concurrent `migrate` runs on the same database.
_LOCK_ID = 0x52_49_46_54  # "RIFT"


@dataclass(frozen=True)
class Migration:
    version: str  # "001_init"
    path: Path

    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    migrations = []
    seen_numbers: set[str] = set()
    for path in sorted(directory.glob("*.sql")):
        m = _NAME.match(path.name)
        if not m:
            raise ValueError(f"bad migration filename {path.name!r}; expected NNN_snake_case.sql")
        if m.group(1) in seen_numbers:
            raise ValueError(f"duplicate migration number {m.group(1)}")
        seen_numbers.add(m.group(1))
        migrations.append(Migration(path.stem, path))
    return migrations


def _ensure_table(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    text PRIMARY KEY,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def applied_versions(conn: psycopg.Connection) -> set[str]:
    _ensure_table(conn)
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def pending(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    done = applied_versions(conn)
    return [m for m in discover(directory) if m.version not in done]


def migrate(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every pending migration. Returns the versions applied, in order."""
    applied = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
        for migration in pending(conn, directory):
            conn.execute(migration.sql())
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (migration.version,)
            )
            applied.append(migration.version)
    return applied
