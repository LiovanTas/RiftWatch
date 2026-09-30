"""``riftwatch`` command line."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from riftwatch import __version__
from riftwatch.config import Settings
from riftwatch.db import migrate as migrations
from riftwatch.db.connection import connect
from riftwatch.riot.routing import account_region, match_region


def cmd_doctor(settings: Settings, _args: argparse.Namespace) -> int:
    """Report what is configured and reachable, without calling the Riot API."""
    ok = True
    platform = settings.default_platform
    print(f"riftwatch {__version__}")
    print(
        f"default region   {platform} "
        f"(match-v5: {match_region(platform)}, account-v1: {account_region(platform)})"
    )
    print(f"riot api key     {'set' if settings.riot_api_key else 'MISSING'}")
    ok &= settings.riot_api_key is not None
    print(f"anthropic key    {'set' if settings.anthropic_api_key else 'missing (coach only)'}")
    print(f"coach model      {settings.coach_model}")
    try:
        with connect(settings.database_url) as conn:
            todo = migrations.pending(conn)
        state = f"{len(todo)} pending migration(s)" if todo else "schema up to date"
        print(f"database         reachable, {state}")
        ok &= not todo
    except Exception as exc:  # any connection failure is a diagnosis, not a crash
        print(f"database         UNREACHABLE: {exc}".splitlines()[0])
        print("                 start it with: docker compose up -d db")
        ok = False
    return 0 if ok else 1


def cmd_db_migrate(settings: Settings, _args: argparse.Namespace) -> int:
    with connect(settings.database_url) as conn:
        applied = migrations.migrate(conn)
    if applied:
        for version in applied:
            print(f"applied {version}")
    else:
        print("nothing to apply")
    return 0


def cmd_db_status(settings: Settings, _args: argparse.Namespace) -> int:
    with connect(settings.database_url) as conn:
        done = migrations.applied_versions(conn)
    for m in migrations.discover():
        print(f"[{'x' if m.version in done else ' '}] {m.version}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="riftwatch", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check configuration and database").set_defaults(
        func=cmd_doctor
    )

    db = sub.add_parser("db", help="database management").add_subparsers(
        dest="db_command", required=True
    )
    db.add_parser("migrate", help="apply pending migrations").set_defaults(func=cmd_db_migrate)
    db.add_parser("status", help="list migrations and whether each is applied").set_defaults(
        func=cmd_db_status
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(Settings.from_env(), args)


if __name__ == "__main__":
    sys.exit(main())
