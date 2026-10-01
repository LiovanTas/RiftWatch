"""Migration tests. The database ones need a disposable Postgres:

    RIFTWATCH_TEST_DATABASE_URL=postgresql://riftwatch:riftwatch@127.0.0.1:5432/riftwatch_test

CI provides one; locally they are skipped unless the variable is set.
WARNING: the test database is wiped.
"""

import os

import pytest

from riftwatch.db import migrate

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


def test_shipped_migrations_are_well_formed():
    found = migrate.discover()
    assert found, "no migrations shipped"
    assert found[0].version == "001_init"
    numbers = [m.version[:3] for m in found]
    assert numbers == sorted(numbers)


def test_discover_rejects_bad_names(tmp_path):
    (tmp_path / "1_init.sql").write_text("")
    with pytest.raises(ValueError, match="bad migration filename"):
        migrate.discover(tmp_path)


def test_discover_rejects_duplicate_numbers(tmp_path):
    (tmp_path / "001_a.sql").write_text("")
    (tmp_path / "001_b.sql").write_text("")
    with pytest.raises(ValueError, match="duplicate"):
        migrate.discover(tmp_path)


@pytest.fixture
def conn():
    if not TEST_DB:
        pytest.skip("RIFTWATCH_TEST_DATABASE_URL not set")
    from riftwatch.db.connection import connect

    with connect(TEST_DB) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        yield c


def test_migrate_applies_then_is_idempotent(conn):
    applied = migrate.migrate(conn)
    assert applied == [m.version for m in migrate.discover()]
    assert migrate.migrate(conn) == []
    assert migrate.pending(conn) == []
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
    }
    assert {"accounts", "matches", "match_timelines", "match_participants",
            "rank_snapshots"} <= tables


def test_failed_migration_rolls_back_whole_run(conn, tmp_path):
    (tmp_path / "001_ok.sql").write_text("CREATE TABLE a (x int);")
    (tmp_path / "002_bad.sql").write_text("CREATE TABLE b (x nosuchtype);")
    with pytest.raises(Exception):
        migrate.migrate(conn, tmp_path)
    assert migrate.applied_versions(conn) == set()
    exists = conn.execute("SELECT to_regclass('public.a')").fetchone()[0]
    assert exists is None
