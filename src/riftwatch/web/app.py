"""RiftWatch web API.

    uvicorn riftwatch.web.app:app

Request handlers only ever read Postgres (pooled connections, indexed feature tables, cached
coaching). Anything slow -- downloading from Riot, generating coaching -- is either a
background job or an explicit POST, so opening a page never waits on Riot or the LLM.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from psycopg_pool import ConnectionPool

import riftwatch.db.connection  # noqa: F401  (installs the orjson jsonb codecs)
from riftwatch import __version__
from riftwatch.coach.llm import Coach, CoachError
from riftwatch.coach.pipeline import ReportError, game_report, recent_report
from riftwatch.config import ConfigError, Settings
from riftwatch.db import repo
from riftwatch.features import store as feature_store
from riftwatch.ingest import Ingestor, NotFound
from riftwatch.report.html import game_html
from riftwatch.riot.api import RiotApi
from riftwatch.riot.client import RiotApiError, RiotClient
from riftwatch.riot.routing import RiotId, UnknownRegion, parse_riot_id, platform_for
from riftwatch.web import pages, serialize
from riftwatch.web.jobs import JobManager


def parse_path_riot_id(text: str) -> RiotId:
    """URL form ``Name-TAG`` (as op.gg uses); the tag is after the last hyphen."""
    name, sep, tag = text.rpartition("-")
    if not sep or not name.strip() or not tag.strip():
        raise HTTPException(400, f"expected Name-TAG in the URL, got {text!r}")
    return RiotId(name.strip(), tag.strip())


class Services:
    """Everything a request handler needs, shared across requests."""

    def __init__(self, settings: Settings, pool: ConnectionPool, api: RiotApi | None,
                 coach: Coach | None, jobs: JobManager) -> None:
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
    jobs: JobManager | None = None,
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
    services = Services(settings, pool, api, coach, jobs or JobManager())

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if owns_pool:
            pool.open()
        yield
        services.jobs.shutdown()
        if owns_pool:
            pool.close()

    # Every handler declares its return type, so FastAPI serializes straight to JSON bytes;
    # no custom response class needed.
    app = FastAPI(title="RiftWatch", version=__version__, lifespan=lifespan)
    app.state.services = services

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
             WHERE p.puuid = %s
             ORDER BY m.game_start DESC
             LIMIT %s OFFSET %s
            """,
            (puuid, limit, offset),
        ).fetchall()
        return [{
            "match_id": r[0], "game_start": r[1].isoformat(), "duration_s": r[2], "patch": r[3],
            "queue_id": r[4], "champion": r[5], "role": r[6], "win": r[7],
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
            "SELECT count(*) FROM match_participants WHERE puuid = %s", (puuid,)
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
    def sync(region: str, riot_id: str, count: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
        platform = platform_for(region)
        rid = parse_path_riot_id(riot_id)
        riot = services.require_api()

        def work(progress) -> dict[str, Any]:
            with pool.connection() as conn:
                result = Ingestor(conn, riot).sync(rid, platform, count=count, progress=progress)
                for match_id in result.new_matches:
                    game = feature_store.load(conn, match_id)
                    if game is not None:
                        feature_store.save(conn, game)
                progress(f"analysed {len(result.new_matches)} new game(s)")
                return {"riot_id": result.riot_id, "new_matches": len(result.new_matches),
                        "already_cached": result.already_cached,
                        "failed": len(result.failed)}

        key = f"sync:{platform}:{rid.game_name.lower()}#{rid.tag_line.lower()}"
        job, created = services.jobs.submit(key, "sync", work)
        return {"job": job.to_json(), "created": created}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
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
    def match_coach(region: str, riot_id: str, match_id: str) -> dict[str, Any]:
        if coach is None:
            raise HTTPException(503, "LLM coach not configured on the server")
        return review(region, riot_id, match_id, generate=True)

    def recent(region: str, riot_id: str, games: int, generate: bool) -> dict[str, Any]:
        with pool.connection() as conn:
            account = account_or_404(conn, region, riot_id)
            result = recent_report(conn, account["puuid"], games=games, coach=coach,
                                   generate=generate)
            return serialize.recent_review(result)

    @app.get("/api/players/{region}/{riot_id}/recent")
    def recent_review(region: str, riot_id: str,
                      games: int = Query(20, ge=5, le=50)) -> dict[str, Any]:
        return recent(region, riot_id, games, generate=False)

    @app.post("/api/players/{region}/{riot_id}/recent/coach")
    def recent_coach(region: str, riot_id: str,
                     games: int = Query(20, ge=5, le=50)) -> dict[str, Any]:
        if coach is None:
            raise HTTPException(503, "LLM coach not configured on the server")
        return recent(region, riot_id, games, generate=True)

    @app.get("/players/{region}/{riot_id}/matches/{match_id}", response_class=HTMLResponse)
    def match_page(region: str, riot_id: str, match_id: str) -> HTMLResponse:
        with pool.connection() as conn:
            account = account_or_404(conn, region, riot_id)
            result = game_report(conn, account["puuid"], match_id, coach=coach, generate=False,
                                 advisor=advisor)
        coach_url = (f"/api/players/{region}/{riot_id}/matches/{match_id}/coach"
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
        coach_url = f"/api/players/{region}/{riot_id}/recent/coach" if coach is not None else None
        return HTMLResponse(pages.player_page(region, riot_id, data, recent, coach_url))

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
