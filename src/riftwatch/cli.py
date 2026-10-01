"""``riftwatch`` command line."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

from riftwatch import __version__
from riftwatch.config import ConfigError, Settings
from riftwatch.db import migrate as migrations
from riftwatch.db import repo
from riftwatch.db.connection import connect
from riftwatch.features import store as feature_store
from riftwatch.ingest import Ingestor, NotFound
from riftwatch.riot.api import RiotApi
from riftwatch.riot.client import RiotApiError, RiotClient
from riftwatch.riot.routing import account_region, match_region, parse_riot_id


def _api(settings: Settings) -> RiotApi:
    return RiotApi(RiotClient(settings.require_riot_key()))


def _platform(settings: Settings, args: argparse.Namespace) -> str:
    return args.region or settings.default_platform


def _rank_text(rank: dict | None) -> str:
    if not rank:
        return "unranked (solo/duo)"
    division = f" {rank['rank']}" if rank.get("rank") and rank["tier"] not in repo.APEX else ""
    wins, losses = rank.get("wins", 0), rank.get("losses", 0)
    wr = f", {100 * wins / (wins + losses):.0f}% WR" if wins + losses else ""
    return f"{rank['tier'].title()}{division} {rank.get('leaguePoints', 0)} LP ({wins}W {losses}L{wr})"


def _client_stats(api: RiotApi) -> str:
    s = api.client.stats
    parts = [f"{s['requests']} API requests"]
    rl = sum(v for k, v in s.items() if k.startswith("429"))
    parts.append(f"{rl} rate-limited (429)")
    if s["retries"]:
        parts.append(f"{s['retries']} retries")
    return ", ".join(parts)


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


def cmd_lookup(settings: Settings, args: argparse.Namespace) -> int:
    riot_id = parse_riot_id(args.riot_id)
    platform = _platform(settings, args)
    api = _api(settings)
    with connect(settings.database_url) as conn:
        ing = Ingestor(conn, api)
        account = ing.resolve(riot_id, platform)
        rank = ing.refresh_rank(account["puuid"], platform)
        cached = len(repo.player_match_ids(conn, account["puuid"], limit=100_000))
    print(f"{account['gameName']}#{account['tagLine']}  ({platform})")
    print(f"  rank     {_rank_text(rank)}")
    print(f"  puuid    {account['puuid']}")
    print(f"  cached   {cached} match(es)")
    return 0


def cmd_sync(settings: Settings, args: argparse.Namespace) -> int:
    riot_id = parse_riot_id(args.riot_id)
    api = _api(settings)
    with connect(settings.database_url) as conn:
        result = Ingestor(conn, api).sync(
            riot_id, _platform(settings, args), count=args.count,
            queue=None if args.all_queues else 420,
            with_timeline=not args.no_timelines, progress=print,
        )
    print(f"{result.riot_id}: {_rank_text(result.rank)}")
    print(f"  {len(result.new_matches)} new match(es), {result.already_cached} already cached")
    print(f"  {_client_stats(api)}")
    return 0


def cmd_backfill(settings: Settings, args: argparse.Namespace) -> int:
    riot_id = parse_riot_id(args.riot_id)
    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC) if args.since else None
    api = _api(settings)
    with connect(settings.database_url) as conn:
        result = Ingestor(conn, api).backfill(
            riot_id, _platform(settings, args),
            queue=None if args.all_queues else 420, start_time=since,
            with_timeline=not args.no_timelines, progress=print,
        )
    state = "complete" if result.finished else "paused (rerun the same command to resume)"
    print(f"backfill job {result.job_id} {state}")
    print(f"  {result.ids_seen} match ids, {result.fetched} downloaded, {result.cached} already cached")
    print(f"  this run: {_client_stats(api)}")
    return 0


def cmd_cache(settings: Settings, _args: argparse.Namespace) -> int:
    with connect(settings.database_url) as conn:
        stats = repo.cache_stats(conn)
    print(f"matches    {stats['matches']}  (timelines {stats['timelines']})")
    print(f"players    {stats['players_seen']} seen across cached games")
    print(f"disk       {stats['raw_bytes'] / 1e6:.1f} MB of raw Riot JSON")
    for kind, c in sorted(stats["counters"].items()):
        total = c["hits"] + c["misses"]
        rate = f"{100 * c['hits'] / total:.0f}%" if total else "-"
        print(f"{kind:<10} {c['hits']} cache hits / {c['misses']} downloads  (hit rate {rate}, "
              f"{c['hits']} API calls saved)")
    return 0


def cmd_features(settings: Settings, args: argparse.Namespace) -> int:
    with connect(settings.database_url) as conn:
        if args.rebuild:
            conn.execute("UPDATE participant_game_summary SET extractor_version = 0")
        done, failures = feature_store.extract_pending(conn, args.limit, progress=print)
    print(f"extracted features for {done} match(es)")
    for match_id, error in failures:
        print(f"  skipped {match_id}: {error}")
    return 0 if not failures else 1


def _add_region(p: argparse.ArgumentParser) -> None:
    p.add_argument("--region", "-r", help="na, euw, eune, kr, ... (default: RIFTWATCH_DEFAULT_REGION)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="riftwatch", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check configuration and database").set_defaults(
        func=cmd_doctor
    )

    p = sub.add_parser("lookup", help="resolve a Riot ID and show its current rank")
    p.add_argument("riot_id", help="Name#TAG")
    _add_region(p)
    p.set_defaults(func=cmd_lookup)

    p = sub.add_parser("sync", help="download a player's newest games into the cache")
    p.add_argument("riot_id", help="Name#TAG")
    _add_region(p)
    p.add_argument("--count", "-n", type=int, default=20, help="newest N games (max 100)")
    p.add_argument("--all-queues", action="store_true", help="not just ranked solo/duo")
    p.add_argument("--no-timelines", action="store_true", help="skip per-minute timelines")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("backfill", help="download a player's whole match history (resumable)")
    p.add_argument("riot_id", help="Name#TAG")
    _add_region(p)
    p.add_argument("--since", help="oldest game date, YYYY-MM-DD")
    p.add_argument("--all-queues", action="store_true")
    p.add_argument("--no-timelines", action="store_true")
    p.set_defaults(func=cmd_backfill)

    sub.add_parser("cache", help="cache size and hit rate").set_defaults(func=cmd_cache)

    p = sub.add_parser("features", help="extract per-minute features from cached timelines")
    p.add_argument("--limit", type=int, help="at most N matches this run")
    p.add_argument("--rebuild", action="store_true", help="re-extract every match")
    p.set_defaults(func=cmd_features)

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
    try:
        return args.func(Settings.from_env(), args)
    except (ConfigError, NotFound, RiotApiError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
