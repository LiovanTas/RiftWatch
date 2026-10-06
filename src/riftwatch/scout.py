"""Live-game scouting: who is in the game a player is in right now.

Spectator-v5 gives the ten players and their champions. For each one RiftWatch looks up
solo/duo rank, mastery on the champion they locked in, and their recent ranked games --
downloaded once into the match cache, so scouting the same people again (duo partners,
the next game in a session) costs almost nothing.

Only public, pre-game facts: rank, champion experience, recent form. Nothing here reads the
game client or tells anyone what to do during the game.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import psycopg

from riftwatch.db import repo
from riftwatch.ingest import Ingestor, NotFound
from riftwatch.riot.api import RANKED_SOLO, RANKED_SOLO_QUEUE_ID, RiotApi
from riftwatch.riot.routing import RiotId, platform_for

Progress = Callable[[str], None]

DEFAULT_GAMES = 10          # recent ranked games per player
CHAMPION_HISTORY = 100      # cached ranked games searched for games on the current champion
MIN_GAME_S = 300            # remakes say nothing about anyone
NEW_TO_CHAMPION_POINTS = 10_000

QUEUES = {420: "Ranked Solo/Duo", 440: "Ranked Flex", 400: "Normal Draft", 430: "Normal Blind",
          490: "Quickplay", 450: "ARAM", 1700: "Arena", 1710: "Arena"}


class NotInGame(NotFound):
    pass


@dataclass
class PlayerScout:
    puuid: str | None                 # None for bots
    riot_id: str
    team_id: int
    champion_id: int
    champion: str
    rank: dict[str, Any] | None = None
    games: int = 0                    # recent ranked games looked at
    wins: int = 0
    kills: float = 0.0                # averages over those games
    deaths: float = 0.0
    assists: float = 0.0
    main_role: str | None = None
    main_role_share: float = 0.0
    streak: int = 0                   # +3 = won the last three, -2 = lost the last two
    champion_share: float = 0.0       # share of the recent games on this champion
    champion_games: int = 0           # cached ranked games on this champion
    champion_wins: int = 0
    mastery_points: int | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def kda(self) -> float:
        return (self.kills + self.assists) / max(self.deaths, 1.0)


@dataclass
class ScoutReport:
    platform: str
    game_id: int
    queue_id: int
    queue: str
    game_start_ms: int
    game_length_s: int
    me: str                           # puuid of the player who was looked up
    players: list[PlayerScout]
    bans: list[dict[str, Any]] = field(default_factory=list)

    def team(self, team_id: int) -> list[PlayerScout]:
        return [p for p in self.players if p.team_id == team_id]

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        for p, raw in zip(self.players, out["players"], strict=True):
            raw["kda"] = round(p.kda, 2)
        return out


def summarize(player: PlayerScout, recent: list[tuple], champion_rows: list[tuple],
              entry: dict[str, Any] | None, mastery: dict[str, Any] | None) -> PlayerScout:
    """Fill in a player's numbers. ``recent`` and ``champion_rows`` are
    (champion_id, team_position, win, kills, deaths, assists) tuples, newest first."""
    player.rank = None if entry is None else {
        "tier": entry["tier"], "division": entry.get("rank"), "lp": entry["leaguePoints"],
        "wins": entry["wins"], "losses": entry["losses"]}
    player.mastery_points = None if mastery is None else int(mastery.get("championPoints", 0))
    if player.puuid is not None and mastery is None:
        player.mastery_points = 0               # 404: never played it

    n = len(recent)
    player.games = n
    player.wins = sum(1 for r in recent if r[2])
    if n:
        player.kills = round(sum(r[3] for r in recent) / n, 1)
        player.deaths = round(sum(r[4] for r in recent) / n, 1)
        player.assists = round(sum(r[5] for r in recent) / n, 1)
        roles = Counter(r[1] for r in recent if r[1])
        if roles:
            player.main_role, count = roles.most_common(1)[0]
            player.main_role_share = round(count / n, 2)
        first = recent[0][2]
        run = next((i for i, r in enumerate(recent) if r[2] != first), n)
        player.streak = run if first else -run
        player.champion_share = round(sum(1 for r in recent if r[0] == player.champion_id) / n, 2)
    player.champion_games = len(champion_rows)
    player.champion_wins = sum(1 for r in champion_rows if r[2])
    player.flags = flags(player)
    return player


def flags(p: PlayerScout) -> list[str]:
    """Short factual notes, the kind a player reads in the loading screen."""
    out = []
    if p.puuid is None:
        return ["bot"]
    if p.mastery_points is not None and p.mastery_points < NEW_TO_CHAMPION_POINTS:
        out.append(f"new to {p.champion}")
    if p.games >= 5 and p.champion_share >= 0.6:
        out.append(f"plays mostly {p.champion}")
    if p.streak >= 3:
        out.append(f"won last {p.streak}")
    elif p.streak <= -3:
        out.append(f"lost last {-p.streak}")
    if p.rank is not None and p.rank["wins"] + p.rank["losses"] < 20:
        out.append("few ranked games this season")
    if p.rank is None:
        out.append("unranked in solo/duo")
    return out


def _rows(conn: psycopg.Connection, puuids: list[str]) -> dict[str, list[tuple]]:
    """Every cached ranked game for these players, newest first."""
    rows = conn.execute(
        """
        SELECT p.puuid, m.match_id, p.champion_id, p.team_position, p.win,
               p.kills, p.deaths, p.assists
          FROM match_participants p JOIN matches m USING (match_id)
         WHERE p.puuid = ANY(%s) AND m.queue_id = %s AND m.duration_s >= %s
         ORDER BY m.game_start DESC
        """,
        (puuids, RANKED_SOLO_QUEUE_ID, MIN_GAME_S),
    ).fetchall()
    out: dict[str, list[tuple]] = {p: [] for p in puuids}
    for r in rows:
        out[r[0]].append(r[1:])
    return out


def scout(
    conn: psycopg.Connection,
    api: RiotApi,
    riot_id: RiotId,
    platform: str,
    *,
    games: int = DEFAULT_GAMES,
    names: Callable[[int], str] = lambda cid: f"Champion {cid}",
    progress: Progress | None = None,
) -> ScoutReport:
    platform = platform_for(platform)
    say = progress or (lambda _s: None)
    ingestor = Ingestor(conn, api)
    account = ingestor.resolve(riot_id, platform)
    game = api.active_game(platform, account["puuid"])
    if game is None:
        raise NotInGame(f"{account['gameName']}#{account['tagLine']} isn't in a game right now")

    players = [
        PlayerScout(
            puuid=p.get("puuid") or None,
            riot_id=p.get("riotId") or ("Bot" if p.get("bot") else "Unknown"),
            team_id=int(p["teamId"]),
            champion_id=int(p["championId"]),
            champion=names(int(p["championId"])),
        )
        for p in game["participants"]
    ]
    humans = [p for p in players if p.puuid]

    # Network only, in parallel: the shared rate limiter paces the threads.
    def lookup(p: PlayerScout):
        entries = api.league_entries_by_puuid(platform, p.puuid)
        ids = api.match_ids(platform, p.puuid, count=games, queue=RANKED_SOLO_QUEUE_ID)
        mastery = api.champion_mastery(platform, p.puuid, p.champion_id)
        return entries, ids, mastery

    say(f"looking up {len(humans)} players")
    with ThreadPoolExecutor(max_workers=max(1, len(humans))) as pool:
        looked_up = list(pool.map(lookup, humans))

    all_ids: list[str] = []
    for p, (entries, ids, _mastery) in zip(humans, looked_up, strict=True):
        repo.insert_rank_snapshots(conn, p.puuid, entries)
        all_ids.extend(ids)
    fetched = ingestor.fetch_many(all_ids, with_timeline=False, progress=say)
    say(f"{len(fetched.downloaded)} game(s) downloaded, {len(fetched.cached)} already cached")

    history = _rows(conn, [p.puuid for p in humans])
    for p, (entries, ids, mastery) in zip(humans, looked_up, strict=True):
        wanted = set(ids)
        mine = history[p.puuid]
        recent = [r[1:] for r in mine if r[0] in wanted]
        on_champ = [r[1:] for r in mine[:CHAMPION_HISTORY] if r[1] == p.champion_id]
        entry = next((e for e in entries if e["queueType"] == RANKED_SOLO), None)
        summarize(p, recent, on_champ, entry, mastery)
    for p in players:
        if p.puuid is None:
            p.flags = ["bot"]

    queue_id = int(game.get("gameQueueConfigId") or 0)
    return ScoutReport(
        platform=platform,
        game_id=int(game["gameId"]),
        queue_id=queue_id,
        queue=QUEUES.get(queue_id, game.get("gameMode", "Custom").title()),
        game_start_ms=int(game.get("gameStartTime") or 0),
        game_length_s=int(game.get("gameLength") or 0),
        me=account["puuid"],
        players=players,
        bans=[{"team_id": b["teamId"], "champion": names(b["championId"])}
              for b in game.get("bannedChampions", []) if b.get("championId", -1) > 0],
    )
