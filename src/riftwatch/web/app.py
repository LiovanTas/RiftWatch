"""RiftWatch web API.

    uvicorn riftwatch.web.app:app

Request handlers only ever read Postgres (pooled connections, indexed feature tables, cached
coaching). Anything slow -- downloading from Riot, generating coaching -- is either a
background job or an explicit POST, so opening a page never waits on Riot or the LLM.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from psycopg_pool import ConnectionPool
from starlette.datastructures import MutableHeaders
from starlette.middleware.gzip import GZipMiddleware

import riftwatch.db.connection  # noqa: F401  (installs the orjson jsonb codecs)
from riftwatch import __version__
from riftwatch.coach.llm import Coach, CoachError
from riftwatch.coach.pipeline import (
    ReportError,
    champion_pool,
    game_report,
    progress_report,
    sessions_report,
    recent_report,
    stream_coaching,
)
from riftwatch.config import ConfigError, Settings
from riftwatch.db import repo
from riftwatch.features import store as feature_store
from riftwatch.ingest import Ingestor, NotFound
from riftwatch.report.html import game_html
from riftwatch.riot.api import QUEUE_NAMES, SUPPORTED_QUEUES, RiotApi
from riftwatch.riot.client import RiotApiError, RiotClient
from riftwatch.riot.routing import RiotId, UnknownRegion, parse_riot_id, platform_for
from riftwatch.web import limits, pages, serialize
from riftwatch.web.jobs import JobQueue

SECURITY_HEADERS = (
    ("x-content-type-options", "nosniff"),
    ("x-frame-options", "DENY"),
    ("referrer-policy", "same-origin"),
)


class SecurityHeaders:
    """Plain ASGI middleware (not BaseHTTPMiddleware), so streamed coaching passes through
    untouched."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS:
                    headers.setdefault(name, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


def parse_path_riot_id(text: str) -> RiotId:
    """URL form ``Name-TAG`` (as op.gg uses); the tag is after the last hyphen."""
    name, sep, tag = text.rpartition("-")
    if not sep or not name.strip() or not tag.strip():
        raise HTTPException(400, f"expected Name-TAG in the URL, got {text!r}")
    return RiotId(name.strip(), tag.strip())


class Services:
    """Everything a request handler needs, shared across requests."""

    def __init__(self, settings: Settings, pool: ConnectionPool, api: RiotApi | None,
                 coach: Coach | None, jobs: JobQueue) -> None:
        self.settings, self.pool, self.api, self.coach, self.jobs = settings, pool, api, coach, jobs

    def require_api(self) -> RiotApi:
        if self.api is None:
            raise HTTPException(503, "Riot API key not configured on the server")
        return self.api


def create_app(
    settings: Settings | None = None,
    *,
    api: RiotApi | None = None,
    coach: Coach | None = None,
    pool: ConnectionPool | None = None,
    jobs: JobQueue | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if api is None and settings.riot_api_key:
        # One client for the whole server: every user's requests share one rate limiter,
        # which is what keeps the key under Riot's limits.
        api = RiotApi(RiotClient(settings.riot_api_key))
    if coach is None and settings.anthropic_api_key:
        coach = Coach(settings.coach_model, api_key=settings.anthropic_api_key,
                      effort=settings.coach_effort, thinking=settings.coach_thinking)
    from pathlib import Path

    advisor = None
    if any(Path(settings.models_dir).glob("*.joblib")):
        from riftwatch.ml.advisor import Advisor

        advisor = Advisor(Path(settings.models_dir))
    owns_pool = pool is None
    pool = pool or ConnectionPool(
        settings.database_url, min_size=1, max_size=8, open=False,
        kwargs={"autocommit": True, "connect_timeout": 5},
    )
    services = Services(settings, pool, api, coach, jobs or JobQueue(pool))
    rate_limits = limits.parse_limits(settings.rate_limits)

    def sync_job(params: dict[str, Any], progress) -> dict[str, Any]:
        """Runs in whichever server process claims the job, so it rebuilds what it needs
        from its parameters."""
        riot = services.require_api()
        rid = RiotId(params["game_name"], params["tag_line"])
        with pool.connection() as conn:
            result = Ingestor(conn, riot).sync(rid, params["platform"], count=params["count"],
                                               progress=progress)
            for match_id in result.new_matches:
                game = feature_store.load(conn, match_id)
                if game is not None:
                    feature_store.save(conn, game)
            progress(f"analysed {len(result.new_matches)} new game(s)")
            return {"riot_id": result.riot_id, "new_matches": len(result.new_matches),
                    "already_cached": result.already_cached, "failed": len(result.failed)}

    services.jobs.register("sync", sync_job)

    names = None

    def scout_job(params: dict[str, Any], progress) -> dict[str, Any]:
        nonlocal names
        from riftwatch.riot.ddragon import ChampionNames, DataDragon
        from riftwatch.scout import scout

        names = names or ChampionNames(DataDragon())
        with pool.connection() as conn:
            report = scout(conn, services.require_api(),
                           RiotId(params["game_name"], params["tag_line"]), params["platform"],
                           names=names, progress=progress)
        return report.to_json()

    services.jobs.register("scout", scout_job)

    def warm_baselines() -> None:
        try:
            from riftwatch.baselines import build as baseline_build

            with pool.connection() as conn:
                baseline_build.warm(conn)
        except Exception:   # a cold cache is only slower, never wrong
            pass

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if owns_pool:
            pool.open()
        # In the background: the server answers at once, and the first player page finds the
        # shared baselines already in memory.
        import threading

        threading.Thread(target=warm_baselines, name="warm-baselines", daemon=True).start()
        services.jobs.start()
        yield
        services.jobs.shutdown()
        if owns_pool:
            pool.close()

    # Every handler declares its return type, so FastAPI serializes straight to JSON bytes;
    # no custom response class needed.
    app = FastAPI(title="RiftWatch", version=__version__, lifespan=lifespan)
    app.state.services = services
    # Pages carry their CSS and JavaScript inline; compressed they are a fraction of the size.
    # Starlette never compresses text/event-stream, so streamed coaching still streams.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(SecurityHeaders)

    # -- errors -> HTTP ---------------------------------------------------------------------

    def error(status: int):
        async def handler(_request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"error": str(exc)}, status_code=status)
        return handler

    app.add_exception_handler(UnknownRegion, error(400))
    app.add_exception_handler(NotFound, error(404))
    app.add_exception_handler(ReportError, error(404))
    app.add_exception_handler(ConfigError, error(503))
    app.add_exception_handler(RiotApiError, error(502))
    app.add_exception_handler(CoachError, error(502))

    # -- helpers ----------------------------------------------------------------------------

    def visitor(request: Request) -> str:
        # Behind a proxy, uvicorn's --proxy-headers (with FORWARDED_ALLOW_IPS set to the
        # proxy) puts the real client address here.
        return request.client.host if request.client else "anonymous"

    def limit(request: Request, action: str) -> None:
        """429 with Retry-After once a visitor has used up this action for the hour."""
        with pool.connection() as conn:
            retry = limits.hit(conn, f"{action}:{visitor(request)}", rate_limits[action])
        if retry is not None:
            raise HTTPException(429, f"too many {action} requests; try again in "
                                     f"{int(retry // 60) + 1} min",
                                headers={"Retry-After": str(int(retry))})

    def require_coach_budget() -> None:
        with pool.connection() as conn:
            spent = limits.coach_spend_today(conn)
        if spent >= settings.coach_daily_budget_usd:
            raise HTTPException(503, "AI coaching has reached today's budget; "
                                     "it comes back at midnight UTC")

    def account_or_404(conn, region: str, riot_id: str) -> dict[str, Any]:
        platform = platform_for(region)
        rid = parse_path_riot_id(riot_id)
        account = repo.find_account(conn, rid.game_name, rid.tag_line)
        if account is None or account["platform"] != platform:
            raise HTTPException(404, f"{rid} ({platform}) isn't synced yet -- "
                                     f"POST /api/players/{region}/{riot_id}/sync")
        return account

    def match_rows(conn, puuid: str, limit: int, offset: int) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT m.match_id, m.game_start, m.duration_s, m.patch, m.queue_id,
                   s.champion_name, s.role, p.win, p.kills, p.deaths, p.assists,
                   (s.metrics->>'cs_per_min')::float, (s.metrics->>'kill_participation')::float
              FROM match_participants p
              JOIN matches m USING (match_id)
              LEFT JOIN participant_game_summary s USING (match_id, participant_id)
             WHERE p.puuid = %s AND m.queue_id = ANY(%s)
             ORDER BY m.game_start DESC
             LIMIT %s OFFSET %s
            """,
            (puuid, list(SUPPORTED_QUEUES), limit, offset),
        ).fetchall()
        return [{
            "match_id": r[0], "game_start": r[1].isoformat(), "duration_s": r[2], "patch": r[3],
            "queue_id": r[4], "mode": QUEUE_NAMES.get(r[4], "Other"), "champion": r[5], "role": r[6], "win": r[7],
            "kills": r[8], "deaths": r[9], "assists": r[10],
            "cs_per_min": round(r[11], 2) if r[11] is not None else None,
            "kill_participation": round(r[12], 3) if r[12] is not None else None,
            "analysed": r[5] is not None,
        } for r in rows]

    # -- routes -----------------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        return {"ok": True, "version": __version__, "riot_api": api is not None,
                "llm_coach": coach is not None}

    def player_data(conn, account: dict[str, Any]) -> dict[str, Any]:
        puuid = account["puuid"]
        total = conn.execute(
            """SELECT count(*) FROM match_participants p JOIN matches m USING (match_id)
                WHERE p.puuid = %s AND m.queue_id = ANY(%s)""", (puuid, list(SUPPORTED_QUEUES)),
        ).fetchone()[0]
        return {
            "riot_id": f"{account['game_name']}#{account['tag_line']}",
            "platform": account["platform"],
            "rank": serialize.rank_json(repo.latest_rank(conn, puuid)),
            "games_cached": total,
            "recent": match_rows(conn, puuid, 20, 0),
        }

    @app.get("/api/players/{region}/{riot_id}")
    def player(region: str, riot_id: str) -> dict[str, Any]:
        with pool.connection() as conn:
            return player_data(conn, account_or_404(conn, region, riot_id))

    @app.get("/api/players/{region}/{riot_id}/matches")
    def matches(region: str, riot_id: str, limit: int = Query(20, ge=1, le=100),
                offset: int = Query(0, ge=0)) -> dict[str, Any]:
        with pool.connection() as conn:
            account = account_or_404(conn, region, riot_id)
            return {"matches": match_rows(conn, account["puuid"], limit, offset),
                    "limit": limit, "offset": offset}

    @app.post("/api/players/{region}/{riot_id}/sync", status_code=202)
    def sync(request: Request, region: str, riot_id: str,
             count: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
        platform = platform_for(region)
        rid = parse_path_riot_id(riot_id)
        services.require_api()
        limit(request, "sync")
        key = f"sync:{platform}:{rid.game_name.lower()}#{rid.tag_line.lower()}"
        owner = visitor(request)
        job, created = services.jobs.submit(
            key, "sync", {"platform": platform, "game_name": rid.game_name,
                          "tag_line": rid.tag_line, "count": count}, owner=owner)
        return {"job": job.to_json(), "created": created}

    @app.post("/api/scout/{region}/{riot_id}", status_code=202)
    def scout(request: Request, region: str, riot_id: str) -> dict[str, Any]:
        """Scout the player's live game as a background job; the job's result is the report."""
        platform = platform_for(region)
        rid = parse_path_riot_id(riot_id)
        services.require_api()
        limit(request, "scout")
        key = f"scout:{platform}:{rid.game_name.lower()}#{rid.tag_line.lower()}"
        owner = visitor(request)
        job, created = services.jobs.submit(
            key, "scout", {"platform": platform, "game_name": rid.game_name,
                           "tag_line": rid.tag_line}, owner=owner)
        return {"job": job.to_json(), "created": created}

    @app.get("/scout/{region}/{riot_id}", response_class=HTMLResponse)
    def scout_page(region: str, riot_id: str, job: int | None = None) -> HTMLResponse:
        platform_for(region)
        parse_path_riot_id(riot_id)
        found = services.jobs.get(job) if job is not None else None
        if found is not None and found.kind == "scout" and found.status == "done":
            return HTMLResponse(pages.scout_page(region, riot_id, found.result))
        error = found.error if found is not None and found.status == "failed" else None
        return HTMLResponse(pages.scout_page(region, riot_id, None, error))

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: int) -> dict[str, Any]:
        job = services.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.to_json()

    def review(region: str, riot_id: str, match_id: str, generate: bool) -> dict[str, Any]:
        with pool.connection() as conn:
            account = account_or_404(conn, region, riot_id)
            result = game_report(conn, account["puuid"], match_id, coach=coach,
                                 generate=generate, advisor=advisor)
            return serialize.game_review(result)

    @app.get("/api/players/{region}/{riot_id}/matches/{match_id}")
    def match_review(region: str, riot_id: str, match_id: str) -> dict[str, Any]:
        """Scores, curves, deaths and evidence. Coaching only if it's already cached;
        ``coach.pending`` says when the LLM coach could be generated with a POST."""
        return review(region, riot_id, match_id, generate=False)

    @app.post("/api/players/{region}/{riot_id}/matches/{match_id}/coach")
    def match_coach(request: Request, region: str, riot_id: str, match_id: str) -> dict[str, Any]:
        if coach is None:
            raise HTTPException(503, "LLM coach not configured on the server")
        limit(request, "coach")
        require_coach_budget()
        return review(region, riot_id, match_id, generate=True)

    @app.post("/api/players/{region}/{riot_id}/matches/{match_id}/coach/stream")
    def match_coach_stream(request: Request, region: str, riot_id: str,
                           match_id: str) -> StreamingResponse:
        """Coaching as server-sent events: each point once it is complete and grounded, then
        the validated final answer. A POST, because generating coaching costs money."""
        if coach is None:
            raise HTTPException(503, "LLM coach not configured on the server")
        limit(request, "coach")
        require_coach_budget()
        with pool.connection() as conn:     # fail fast (404s) before the stream starts
            account = account_or_404(conn, region, riot_id)

        def events():
            try:
                with pool.connection() as conn:
                    result = game_report(conn, account["puuid"], match_id, coach=coach,
                                         generate=False, advisor=advisor)
                    for event in stream_coaching(conn, result, coach, "this single game"):
                        yield f"data: {json.dumps(event)}\n\n"
            except (CoachError, ReportError) as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store"})

    def recent(region: str, riot_id: str, games: int, generate: bool) -> dict[str, Any]:
        with pool.connection() as conn:
            account = account_or_404(conn, region, riot_id)
            result = recent_report(conn, account["puuid"], games=games, coach=coach,
                                   generate=generate)
            return serialize.recent_review(result)

    @app.get("/api/players/{region}/{riot_id}/champions")
    def champions(region: str, riot_id: str,
                  games: int = Query(100, ge=5, le=300)) -> dict[str, Any]:
        with pool.connection() as conn:
            account = account_or_404(conn, region, riot_id)
            return serialize.pool_json(champion_pool(conn, account["puuid"], games=games))

    @app.get("/api/players/{region}/{riot_id}/progress")
    def progress(region: str, riot_id: str,
                 weeks: int = Query(12, ge=1, le=52)) -> dict[str, Any]:
        with pool.connection() as conn:
            account = account_or_404(conn, region, riot_id)
            return serialize.progress_json(progress_report(conn, account["puuid"], weeks=weeks))

    @app.get("/api/players/{region}/{riot_id}/sessions")
    def sessions(region: str, riot_id: str,
                 games: int = Query(200, ge=20, le=500)) -> dict[str, Any]:
        with pool.connection() as conn:
            account = account_or_404(conn, region, riot_id)
            return serialize.sessions_json(sessions_report(conn, account["puuid"], games=games))

    @app.get("/api/players/{region}/{riot_id}/recent")
    def recent_review(region: str, riot_id: str,
                      games: int = Query(20, ge=5, le=50)) -> dict[str, Any]:
        return recent(region, riot_id, games, generate=False)

    @app.post("/api/players/{region}/{riot_id}/recent/coach")
    def recent_coach(request: Request, region: str, riot_id: str,
                     games: int = Query(20, ge=5, le=50)) -> dict[str, Any]:
        if coach is None:
            raise HTTPException(503, "LLM coach not configured on the server")
        limit(request, "coach")
        require_coach_budget()
        return recent(region, riot_id, games, generate=True)

    @app.get("/players/{region}/{riot_id}/matches/{match_id}", response_class=HTMLResponse)
    def match_page(region: str, riot_id: str, match_id: str) -> HTMLResponse:
        with pool.connection() as conn:
            account = account_or_404(conn, region, riot_id)
            result = game_report(conn, account["puuid"], match_id, coach=coach, generate=False,
                                 advisor=advisor)
        coach_url = (f"/api/players/{region}/{riot_id}/matches/{match_id}/coach/stream"
                     if coach is not None else None)
        links = [("RiftWatch", "/"),
                 (f"{account['game_name']}#{account['tag_line']}", f"/players/{region}/{riot_id}")]
        return HTMLResponse(game_html(result, coach_url=coach_url, links=links))

    # -- pages ------------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse(pages.search_page())

    @app.get("/search")
    def search(region: str, riot_id: str):
        try:
            platform_for(region)
            rid = parse_riot_id(riot_id)
        except ValueError as exc:
            return HTMLResponse(pages.search_page(str(exc)), status_code=400)
        return RedirectResponse(
            f"/players/{region.lower()}/{pages.path_id(rid.game_name, rid.tag_line)}", 303)

    @app.get("/players/{region}/{riot_id}", response_class=HTMLResponse)
    def player_page(region: str, riot_id: str) -> HTMLResponse:
        platform_for(region)
        parse_path_riot_id(riot_id)
        with pool.connection() as conn:
            try:
                account = account_or_404(conn, region, riot_id)
            except HTTPException:
                return HTMLResponse(pages.player_page(region, riot_id, None, None, None))
            data = player_data(conn, account)
            try:
                recent = recent_report(conn, account["puuid"], coach=coach, generate=False)
            except ReportError:
                recent = None
            try:
                champs = champion_pool(conn, account["puuid"])
            except ReportError:
                champs = None
            try:
                progress_data = progress_report(conn, account["puuid"])
            except ReportError:
                progress_data = None
            try:
                session_data = sessions_report(conn, account["puuid"])
            except ReportError:
                session_data = None
        coach_url = f"/api/players/{region}/{riot_id}/recent/coach" if coach is not None else None
        return HTMLResponse(pages.player_page(region, riot_id, data, recent, coach_url, champs,
                                              progress_data, session_data))

    return app


def __getattr__(name: str):
    # `uvicorn riftwatch.web.app:app` builds the app on first access, so importing this
    # module (tests, tools) never connects to anything.
    if name == "app":
        from dotenv import load_dotenv

        load_dotenv()
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
