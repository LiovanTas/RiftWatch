"""Record a game second by second from Riot's Live Client Data API.

While a match is running, the League game client serves its own state at
``https://127.0.0.1:2999/liveclientdata/allgamedata`` -- an interface Riot provides for
apps to read. It covers the player's own health, mana and gold, every player's score,
level, items and respawn state, and the event feed (kills, objectives, game end). It has
no positions and no other player's health.

The recorder polls it once a second and appends to a local JSON-lines file per game, so a
crash or a closed laptop keeps everything recorded up to that point. Nothing is sent
anywhere; the file is imported into Postgres afterwards (:mod:`riftwatch.live.store`).

Riot's policy allows recording for post-game analysis; it does not allow telling players
what to do during a match, so nothing here produces live advice.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

LIVE_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"


def default_dir() -> Path:
    import os

    base = os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share"
    return Path(base) / "riftwatch" / "recordings"


class LiveClient:
    """Reads the local game API. The game serves a certificate signed by Riot's own root,
    not a public one; the request never leaves this machine, so it isn't verified."""

    def __init__(self, url: str = LIVE_URL, transport: httpx.BaseTransport | None = None) -> None:
        self.url = url
        self._http = httpx.Client(verify=False, timeout=2.0, transport=transport)

    def snapshot(self) -> dict[str, Any] | None:
        """The current game state, or None when no game is running or it's still loading."""
        try:
            r = self._http.get(self.url)
        except httpx.TransportError:
            return None
        if r.status_code != 200:
            return None
        data = r.json()
        if "activePlayer" not in data or "gameData" not in data:
            return None
        return data

    def close(self) -> None:
        self._http.close()


def sample(data: dict[str, Any]) -> dict[str, Any]:
    """The per-second record: compact, numbers only."""
    me = data["activePlayer"]
    stats = me.get("championStats") or {}
    abilities = me.get("abilities") or {}
    return {
        "t": round(float(data["gameData"].get("gameTime", 0.0)), 2),
        "hp": round(float(stats.get("currentHealth", 0.0)), 1),
        "hp_max": round(float(stats.get("maxHealth", 0.0)), 1),
        "res": round(float(stats.get("resourceValue", 0.0)), 1),
        "res_max": round(float(stats.get("resourceMax", 0.0)), 1),
        "gold": round(float(me.get("currentGold", 0.0)), 1),
        "level": int(me.get("level", 0)),
        "abilities": {k: (abilities.get(k) or {}).get("abilityLevel", 0) for k in ("Q", "W", "E", "R")},
        "players": [
            {
                "riot_id": p.get("riotId") or p.get("summonerName", ""),
                "dead": bool(p.get("isDead")),
                "respawn": round(float(p.get("respawnTimer", 0.0)), 1),
                "level": int(p.get("level", 0)),
                **{k: int((p.get("scores") or {}).get(k, 0))
                   for k in ("kills", "deaths", "assists", "creepScore")},
                "ward_score": round(float((p.get("scores") or {}).get("wardScore", 0.0)), 1),
                "items": sorted(int(i.get("itemID", 0)) for i in p.get("items") or []),
            }
            for p in data.get("allPlayers") or []
        ],
    }


def header(data: dict[str, Any]) -> dict[str, Any]:
    """Who and what this game is, written once at the top of the file."""
    me = data["activePlayer"]
    riot_id = me.get("riotId") or me.get("summonerName", "")
    players = data.get("allPlayers") or []
    mine = next((p for p in players if (p.get("riotId") or p.get("summonerName")) == riot_id), {})
    return {
        "kind": "header",
        "recorded_at": time.time(),
        "riot_id": riot_id,
        "champion": mine.get("championName", ""),
        "position": mine.get("position", ""),
        "team": mine.get("team", ""),
        "game_mode": data["gameData"].get("gameMode", ""),
        "players": [{"riot_id": p.get("riotId") or p.get("summonerName", ""),
                     "champion": p.get("championName", ""), "team": p.get("team", ""),
                     "position": p.get("position", "")} for p in players],
    }


@dataclass
class Session:
    path: Path
    started: float
    samples: int = 0
    events_seen: set[int] = field(default_factory=set)
    last_t: float = -1.0


class Recorder:
    """State machine: idle -> recording (game found) -> idle (game gone), one file per game."""

    def __init__(self, directory: Path | None = None, client: LiveClient | None = None,
                 say: Callable[[str], None] = print) -> None:
        self.directory = directory or default_dir()
        self.client = client or LiveClient()
        self.say = say
        self.session: Session | None = None

    def tick(self) -> None:
        """Poll once. Call about once a second."""
        data = self.client.snapshot()
        if data is None:
            if self.session is not None:
                self._finish("game closed")
            return
        if self.session is None:
            self._start(data)
        s = self.session
        assert s is not None
        rec = sample(data)
        if rec["t"] < s.last_t - 5:          # clock went backwards: a new game began
            self._finish("new game")
            self._start(data)
            s = self.session
        new_events = [e for e in (data.get("events") or {}).get("Events", [])
                      if e.get("EventID") not in s.events_seen]
        with gzip.open(s.path, "at", encoding="utf-8") as f:
            if rec["t"] != s.last_t:
                f.write(json.dumps({"kind": "sample", **rec}) + "\n")
                s.samples += 1
                s.last_t = rec["t"]
            for e in new_events:
                s.events_seen.add(e.get("EventID"))
                f.write(json.dumps({"kind": "event", **e}) + "\n")
                if e.get("EventName") == "GameEnd":
                    self.say(f"game over ({e.get('Result', '?')}), recording kept")

    def _start(self, data: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        head = header(data)
        safe = "".join(c if c.isalnum() else "_" for c in head["riot_id"])[:32]
        path = self.directory / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(head) + "\n")
        self.session = Session(path, time.time())
        self.say(f"recording {head['champion']} {head['position'].lower()} -> {path.name}")

    def _finish(self, why: str) -> None:
        s = self.session
        self.session = None
        if s is not None:
            self.say(f"recording stopped ({why}): {s.samples} samples in {s.path.name}")

    def run(self, interval: float = 1.0, should_stop: Callable[[], bool] = lambda: False) -> None:
        self.say(f"waiting for a game; recordings go to {self.directory}")
        try:
            while not should_stop():
                started = time.monotonic()
                self.tick()
                time.sleep(max(0.0, interval - (time.monotonic() - started)))
        except KeyboardInterrupt:
            self.say("recorder stopped")
        finally:
            if self.session is not None:
                self._finish("recorder stopped")


def read(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """(header, samples, events) from a recording file. A truncated last line -- the
    recorder was killed mid-write -- is skipped."""
    head: dict[str, Any] = {}
    samples, events = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        try:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = rec.pop("kind", None)
                if kind == "header":
                    head = rec
                elif kind == "sample":
                    samples.append(rec)
                elif kind == "event":
                    events.append(rec)
        except (EOFError, OSError):
            pass                            # gzip stream cut off mid-block
    return head, samples, events
