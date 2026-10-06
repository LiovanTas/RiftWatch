"""``riftwatch`` command line."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import psycopg
from dotenv import load_dotenv

from riftwatch import __version__
from riftwatch.baselines import build as baseline_build
from riftwatch.baselines import crawl as baseline_crawl
from riftwatch.coach.llm import Coach, CoachError
from riftwatch.coach.pipeline import ReportError, game_report, recent_report
from riftwatch.config import ConfigError, Settings
from riftwatch.db import migrate as migrations
from riftwatch.db import repo
from riftwatch.db.connection import connect
from riftwatch.features import store as feature_store
from riftwatch.ingest import Ingestor, NotFound
from riftwatch.report.terminal import render
from riftwatch.riot.api import QUEUE_NAMES, RiotApi, parse_queues
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
    lines = [", ".join(parts)]
    for hit in api.client.rate_limited:
        lines.append(f"    429 from {hit['type']} limit on {hit['routing']} {hit['method']}: "
                     f"retry-after {hit['retry_after']}, app count {hit['app_count']}, "
                     f"method count {hit['method_count']}")
    return "\n".join(lines)


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
            queues=None if args.all_queues else parse_queues(args.queues),
            with_timeline=not args.no_timelines, progress=print,
        )
    print(f"{result.riot_id}: {_rank_text(result.rank)}")
    print(f"  {len(result.new_matches)} new match(es), {result.already_cached} already cached")
    print(f"  {_client_stats(api)}")
    _import_recordings(settings, quiet=True)
    return 0


def _import_recordings(settings: Settings, quiet: bool = False) -> None:
    """Import live recordings and link them to synced matches (cheap; runs after sync)."""
    from riftwatch.live.recorder import default_dir
    from riftwatch.live.store import import_dir

    directory = default_dir()
    if not directory.exists():
        if not quiet:
            print(f"no recordings yet in {directory}")
        return
    with connect(settings.database_url) as conn:
        report = import_dir(conn, directory)
    if report.imported or not quiet:
        print(f"live recordings: {report.imported} imported, {report.linked} linked to matches")
    for name, why in report.skipped if not quiet else []:
        print(f"  skipped {name}: {why}")


def cmd_ml_dataset(settings: Settings, args: argparse.Namespace) -> int:
    from pathlib import Path

    from riftwatch.ml.dataset import build

    tiers = [t.strip().upper() for t in args.tiers.split(",")]
    with connect(settings.database_url) as conn:
        report = build(conn, Path(args.out), tiers=tiers, progress=print)
    print(f"{report.games} games ({report.skipped} skipped) -> {args.out}")
    for role, n in report.rows.items():
        print(f"  {role:<8} {n} rows")
    return 0


def cmd_ml_train(settings: Settings, args: argparse.Namespace) -> int:
    from pathlib import Path

    from riftwatch.ml.situations import ROLES
    from riftwatch.ml.train import save, train

    data = Path(args.data)
    roles = [r.strip().upper() for r in args.roles.split(",")] if args.roles else list(ROLES)
    print(f"{'role':<8} {'decision acc':>12} {'best baseline':>13} {'top-2':>6} "
          f"{'objective AUC':>13} {'death AUC':>9}")
    for role in roles:
        path = data / f"{role}.parquet"
        if not path.exists():
            print(f"{role:<8} no dataset at {path} -- run `riftwatch ml dataset` first")
            continue
        trained = train(path)
        save(trained, Path(settings.models_dir))
        d, o = trained.metrics["decision"], trained.metrics["outcomes"]
        best = max(d["baseline_majority_accuracy"], d["baseline_by_minute_accuracy"])
        def auc(name: str) -> str:
            value = o.get(name, {}).get("auc")
            return "n/a" if value is None else f"{value:.3f}"

        print(f"{role:<8} {d['accuracy']:>12.1%} {best:>13.1%} {d['top2_accuracy']:>6.1%} "
              f"{auc('team_objective'):>13} {auc('player_died'):>9}")
    print(f"models saved to {settings.models_dir} "
          f"({trained.metrics['games']} games, tested on {trained.metrics['test_games']} unseen)")
    return 0


def cmd_refresh(settings: Settings, args: argparse.Namespace) -> int:
    from riftwatch.baselines import refresh as rf
    from riftwatch.riot.ddragon import DataDragon

    patch = args.patch or rf.live_patch(DataDragon())
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    api = None if args.dry_run else _api(settings)
    with connect(settings.database_url) as conn:
        report = rf.refresh(
            conn, current_patch=patch, target=args.target, dry_run=args.dry_run,
            crawl_fn=(rf.crawler(api, regions, connect, settings.database_url)
                      if api else lambda tiers, players: 0),
            extract_fn=lambda: feature_store.extract_pending(conn, progress=print),
        )
    print(f"patch {report.patch}: crawled games per bucket")
    for bucket in rf.BUCKET_TIERS:
        after = report.after.get(bucket, 0)
        added = after - report.before.get(bucket, 0)
        print(f"  {bucket:<12} {after:>6}" + (f"  (+{added})" if added else ""))
    if api is not None:
        print(f"  {_client_stats(api)}")
    return 0


def cmd_record(settings: Settings, args: argparse.Namespace) -> int:
    if args.import_only:
        _import_recordings(settings)
        return 0
    from riftwatch.live.recorder import Recorder

    Recorder().run(interval=args.interval)
    return 0


def cmd_backfill(settings: Settings, args: argparse.Namespace) -> int:
    riot_id = parse_riot_id(args.riot_id)
    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC) if args.since else None
    api = _api(settings)
    queues = [None] if args.all_queues else list(parse_queues(args.queues))
    with connect(settings.database_url) as conn:
        for queue in queues:
            if queue is not None:
                print(f"{QUEUE_NAMES[queue]}:")
            # One resumable job per queue: Riot filters match ids by one queue at a time.
            result = Ingestor(conn, api).backfill(
                riot_id, _platform(settings, args), queue=queue, start_time=since,
                with_timeline=not args.no_timelines, progress=print,
            )
            state = "complete" if result.finished else "paused (rerun the same command to resume)"
            print(f"  backfill job {result.job_id} {state}: {result.ids_seen} match ids, "
                  f"{result.fetched} downloaded, {result.cached} already cached")
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


def cmd_crawl(settings: Settings, args: argparse.Namespace) -> int:
    api = _api(settings)
    if args.high_elo:
        tiers = baseline_crawl.HIGH_ELO_TIERS
    elif args.tiers:
        tiers = [t.strip().upper() for t in args.tiers.split(",")]
    else:
        tiers = baseline_crawl.DEFAULT_TIERS
    regions = ([r.strip() for r in args.regions.split(",") if r.strip()] if args.regions
               else [_platform(settings, args)])
    reports = baseline_crawl.crawl_regions(
        settings.database_url, api, regions, connect=connect, progress=print, tiers=tiers,
        players_per_division=args.players, matches_per_player=args.matches,
        max_age_days=args.days,
    )
    with connect(settings.database_url) as conn:
        feature_store.extract_pending(conn)
        counts = baseline_crawl.sample_counts(conn)
        apex = baseline_crawl.high_elo_counts(conn)
    for r in reports:
        state = f"stopped: {r.error}" if r.error else "done"
        print(f"{r.platform}: sampled {r.players} players, {r.matches_new} games downloaded, "
              f"{r.matches_cached} already cached ({state})")
    print(f"  {_client_stats(api)}")
    print("crawled games per tier bucket (all crawls so far):")
    for bucket in baseline_build.TIER_ORDER:
        if bucket in counts:
            print(f"  {bucket:<12} {counts[bucket]}")
    if apex:
        print("games found through each apex tier:")
        for tier in baseline_crawl.HIGH_ELO_TIERS:
            if tier in apex:
                print(f"  {tier:<12} {apex[tier]}")
    return 0 if not any(r.error for r in reports) else 1


def cmd_baselines(settings: Settings, args: argparse.Namespace) -> int:
    with connect(settings.database_url) as conn:
        feature_store.extract_pending(conn)
        report = baseline_build.build(conn, patch_count=args.patches, min_n=args.min_n)
        coverage = conn.execute(
            """
            SELECT tier_bucket, role, max(n) FROM baselines
             WHERE champion_id = 0 AND minute IS NULL GROUP BY tier_bucket, role
            """
        ).fetchall()
    if not report.games:
        print("no crawled games yet -- run `riftwatch crawl` first "
              "(your own synced games are left out so you aren't compared against yourself)")
        return 1
    print(f"built {report.rows} baseline rows from {report.games} games "
          f"(patches {report.patch_window})")
    table: dict[str, dict[str, int]] = {}
    for bucket, role, n in coverage:
        table.setdefault(bucket, {})[role] = n
    roles = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
    print(f"  {'players per group':<14}" + "".join(f"{r[:6]:>8}" for r in roles))
    for bucket in baseline_build.TIER_ORDER:
        if bucket in table:
            print(f"  {bucket:<14}" + "".join(f"{table[bucket].get(r, 0):>8}" for r in roles))
    return 0


def _advisor(settings: Settings):
    """The high-elo advisor, if models have been trained (riftwatch ml train)."""
    from pathlib import Path

    directory = Path(settings.models_dir)
    if not any(directory.glob("*.joblib")):
        return None
    from riftwatch.ml.advisor import Advisor

    return Advisor(directory)


def cmd_coach(settings: Settings, args: argparse.Namespace) -> int:
    riot_id = parse_riot_id(args.riot_id)
    coach = None
    if not args.offline and settings.anthropic_api_key:
        coach = Coach(settings.coach_model, api_key=settings.anthropic_api_key,
                      effort=settings.coach_effort, thinking=settings.coach_thinking)
    with connect(settings.database_url) as conn:
        account = repo.find_account(conn, riot_id.game_name, riot_id.tag_line)
        if account is None or args.sync:
            api = _api(settings)
            Ingestor(conn, api).sync(riot_id, _platform(settings, args), count=args.games,
                                     queues=parse_queues(args.queues))
            feature_store.extract_pending(conn)
            account = repo.find_account(conn, riot_id.game_name, riot_id.tag_line)
        puuid = account["puuid"]
        if args.match or args.last:
            match_id = args.match or (repo.player_match_ids(
                conn, puuid, queue_id=parse_queues(args.queues), limit=1) or [None])[0]
            if match_id is None:
                raise ReportError("no cached games for this player -- run sync first")
            result = game_report(conn, puuid, match_id, coach=coach, tier=args.tier,
                                 refresh=args.refresh, advisor=_advisor(settings))
        else:
            result = recent_report(conn, puuid, games=args.games, coach=coach, tier=args.tier,
                                   refresh=args.refresh, queue_id=parse_queues(args.queues))
    print(render(result, show_evidence=args.evidence))
    if args.html:
        from pathlib import Path

        from riftwatch.report.html import game_html, recent_html

        page = game_html(result) if result.scope == "game" else recent_html(result)
        Path(args.html).write_text(page, encoding="utf-8")
        print(f"\nHTML report written to {args.html}")
    return 0


def cmd_watchdog(settings: Settings, args: argparse.Namespace) -> int:
    from riftwatch.watchdog import win32
    from riftwatch.watchdog.core import parse_hotkey

    if not win32.IS_WINDOWS:
        print("error: the watchdog only runs on Windows", file=sys.stderr)
        return 2
    if args.status:
        from riftwatch.watchdog.run import observe

        obs, windows = observe(0.0)
        if not obs.running:
            print("League game is not running")
        else:
            print(f"League game running (pid {sorted({w.pid for w in windows})}): "
                  f"foreground={obs.foreground} fullscreen={obs.fullscreen} "
                  f"responding={obs.responding}")
        return 0
    from pathlib import Path

    from riftwatch.watchdog.run import run

    spec = parse_hotkey(args.hotkey or settings.kill_hotkey)
    log_file = Path(args.log) if args.log else None
    if args.record:
        import threading

        from riftwatch.live.recorder import Recorder

        stop = threading.Event()
        threading.Thread(target=Recorder().run, kwargs={"should_stop": stop.is_set},
                         daemon=True, name="recorder").start()
    run(spec, auto_kill_after=args.auto_kill_after, hang_threshold=args.hang_seconds,
        log_file=log_file)
    return 0


def cmd_scout(settings: Settings, args: argparse.Namespace) -> int:
    from riftwatch.report.terminal import scout_text
    from riftwatch.riot.ddragon import ChampionNames, DataDragon
    from riftwatch.scout import scout

    api = _api(settings)
    with connect(settings.database_url) as conn:
        report = scout(conn, api, parse_riot_id(args.riot_id), _platform(settings, args),
                       games=args.games, names=ChampionNames(DataDragon()), progress=print)
    print()
    print(scout_text(report))
    print()
    print(f"  {_client_stats(api)}")
    return 0


def cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: the web server needs the web extras: pip install -e \".[web]\"",
              file=sys.stderr)
        return 2
    # A factory string, so uvicorn can start several worker processes; each builds the app
    # from the same environment (.env is already loaded). Jobs, rate limits and the baseline
    # cache's staleness check all go through Postgres, so the workers stay consistent.
    uvicorn.run("riftwatch.web.app:create_app", factory=True, host=args.host, port=args.port,
                workers=args.workers, log_level="info")
    return 0


def _add_region(p: argparse.ArgumentParser) -> None:
    p.add_argument("--region", "-r", help="na, euw, eune, kr, ... (default: RIFTWATCH_DEFAULT_REGION)")


def _add_queues(p: argparse.ArgumentParser) -> None:
    p.add_argument("--queues", default="solo,flex,draft",
                   help="modes: solo, flex, draft, comma-separated (default all three)")


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
    _add_queues(p)
    p.add_argument("--all-queues", action="store_true", help="every mode, not just draft and ranked")
    p.add_argument("--no-timelines", action="store_true", help="skip per-minute timelines")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("backfill", help="download a player's whole match history (resumable)")
    p.add_argument("riot_id", help="Name#TAG")
    _add_region(p)
    p.add_argument("--since", help="oldest game date, YYYY-MM-DD")
    _add_queues(p)
    p.add_argument("--all-queues", action="store_true", help="every mode, not just draft and ranked")
    p.add_argument("--no-timelines", action="store_true")
    p.set_defaults(func=cmd_backfill)

    sub.add_parser("cache", help="cache size and hit rate").set_defaults(func=cmd_cache)

    p = sub.add_parser("crawl", help="sample ladder games from each tier for baselines")
    _add_region(p)
    p.add_argument("--regions", help="crawl several regions at once, e.g. na,euw,kr")
    p.add_argument("--tiers", help="comma list, e.g. GOLD,PLATINUM (default: Iron..Master)")
    p.add_argument("--high-elo", action="store_true", help="Challenger, Grandmaster and Master")
    p.add_argument("--players", type=int, default=2, help="players per division (default 2)")
    p.add_argument("--matches", type=int, default=5, help="ranked games per player (default 5)")
    p.add_argument("--days", type=float, default=baseline_crawl.DEFAULT_MAX_AGE_DAYS,
                   help="only games from the last N days (default 14)")
    p.set_defaults(func=cmd_crawl)

    p = sub.add_parser("baselines", help="rebuild rank-matched baselines from crawled games")
    p.add_argument("--patches", type=int, default=3, help="newest N patches (default 3)")
    p.add_argument("--min-n", type=int, default=20, help="min players per group (default 20)")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("coach", help="coaching for a player's recent games or one game")
    p.add_argument("riot_id", help="Name#TAG")
    _add_region(p)
    which = p.add_mutually_exclusive_group()
    which.add_argument("--match", help="coach one game by match id")
    which.add_argument("--last", action="store_true", help="coach the most recent game")
    p.add_argument("--games", type=int, default=20, help="recent games to analyse (default 20)")
    p.add_argument("--tier", help="compare against this tier instead of the player's rank")
    _add_queues(p)
    p.add_argument("--sync", action="store_true", help="download new games first")
    p.add_argument("--offline", action="store_true", help="template coach, no LLM call")
    p.add_argument("--refresh", action="store_true", help="ignore the cached coaching")
    p.add_argument("--evidence", action="store_true", help="also print every evidence item")
    p.add_argument("--html", metavar="FILE", help="also write a self-contained HTML report")
    p.set_defaults(func=cmd_coach)

    p = sub.add_parser("watchdog", help="kill switch for League's black-screen hang (Windows)")
    p.add_argument("--hotkey", help="e.g. ctrl+alt+k (default: RIFTWATCH_KILL_HOTKEY or ctrl+alt+k)")
    p.add_argument("--hang-seconds", type=float, default=5.0,
                   help="unresponsive this long while fullscreen = hang (default 5)")
    p.add_argument("--auto-kill-after", type=float, metavar="SECONDS",
                   help="kill a confirmed hang automatically after this long (off by default)")
    p.add_argument("--log", help="append triggers to this file")
    p.add_argument("--status", action="store_true", help="show what the watchdog sees, then exit")
    p.add_argument("--record", action="store_true",
                   help="also record each game second by second for post-game analysis")
    p.set_defaults(func=cmd_watchdog)

    p = sub.add_parser("refresh", help="top up crawled games on the current patch and rebuild baselines")
    p.add_argument("--regions", default="na,euw,kr", help="regions to crawl (default na,euw,kr)")
    p.add_argument("--target", type=int, default=300, help="crawled games per rank bucket (default 300)")
    p.add_argument("--patch", help="patch to fill (default: the live patch from Data Dragon)")
    p.add_argument("--dry-run", action="store_true", help="only report what would be crawled")
    p.set_defaults(func=cmd_refresh)

    ml = sub.add_parser("ml", help="high-elo models: build the dataset, train").add_subparsers(
        dest="ml_command", required=True)
    p = ml.add_parser("dataset", help="build per-role training tables from crawled games")
    p.add_argument("--tiers", default="CHALLENGER,GRANDMASTER",
                   help="ladder tiers whose games to use (default CHALLENGER,GRANDMASTER)")
    p.add_argument("--out", default="out/ml/data", help="where to write the tables")
    p.set_defaults(func=cmd_ml_dataset)
    p = ml.add_parser("train", help="train and evaluate the models for each role")
    p.add_argument("--data", default="out/ml/data", help="tables from `ml dataset`")
    p.add_argument("--roles", help="comma list, e.g. JUNGLE,MIDDLE (default: all five)")
    p.set_defaults(func=cmd_ml_train)

    p = sub.add_parser("record", help="record games second by second from the game client")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    p.add_argument("--import", dest="import_only", action="store_true",
                   help="import and link recordings instead of recording")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("scout", help="rank, champion experience and form of everyone in a live game")
    p.add_argument("riot_id", help="Name#TAG of a player in the game")
    _add_region(p)
    p.add_argument("--games", type=int, default=10, help="recent ranked games per player (max 100)")
    p.set_defaults(func=cmd_scout)

    p = sub.add_parser("serve", help="run the web API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1, help="server processes (default 1)")
    p.set_defaults(func=cmd_serve)

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
    # Riot IDs can be in any script; a redirected Windows console is cp1252, so print what
    # it can rather than crash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    load_dotenv()
    # Scheduled tasks start in another folder (Task Scheduler uses System32), so also read the
    # project's own .env; values already set (including from the line above) win.
    from pathlib import Path

    project_env = Path(__file__).resolve().parents[2] / ".env"
    if project_env.exists():
        load_dotenv(project_env)
    args = build_parser().parse_args(argv)
    try:
        return args.func(Settings.from_env(), args)
    except (ConfigError, NotFound, RiotApiError, ReportError, CoachError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except psycopg.OperationalError as exc:
        print(f"error: database unreachable ({str(exc).splitlines()[0]}); "
              "start it with: docker compose up -d db", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
