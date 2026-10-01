"""Data Dragon: Riot's static game data (champion and item names), no API key needed.

Files are cached on disk per game version, since a version's data never changes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

DDRAGON = "https://ddragon.leagueoflegends.com"


def default_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "riftwatch" / "ddragon"


def patch_of(game_version: str) -> str:
    """``"16.19.712.3456"`` -> ``"16.19"``."""
    parts = game_version.split(".")
    if len(parts) < 2 or not all(p.isdigit() for p in parts[:2]):
        raise ValueError(f"not a game version: {game_version!r}")
    return f"{parts[0]}.{parts[1]}"


class DataDragon:
    def __init__(
        self,
        cache_dir: Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_dir = cache_dir or default_cache_dir()
        self._http = httpx.Client(base_url=DDRAGON, timeout=15.0, transport=transport)
        self._versions: list[str] | None = None
        self._champions: dict[str, dict[int, str]] = {}
        self._items: dict[str, dict[int, str]] = {}

    def close(self) -> None:
        self._http.close()

    def _cached_json(self, version: str, name: str, url: str) -> Any:
        path = self.cache_dir / version / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        response = self._http.get(url)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        return response.json()

    def versions(self) -> list[str]:
        """All versions, newest first. Not cached on disk: it changes every patch."""
        if self._versions is None:
            response = self._http.get("/api/versions.json")
            response.raise_for_status()
            self._versions = response.json()
        return self._versions

    def version_for(self, patch_or_version: str) -> str:
        """Best Data Dragon version for a patch (``"16.19"``) or a full game version.
        Falls back to the newest version for patches Data Dragon doesn't have yet."""
        patch = patch_of(patch_or_version)
        for v in self.versions():
            if patch_of(v) == patch:
                return v
        return self.versions()[0]

    def champions(self, version: str) -> dict[int, str]:
        """Champion id -> display name (``202 -> "Jhin"``)."""
        if version not in self._champions:
            data = self._cached_json(
                version, "champion.json", f"/cdn/{version}/data/en_US/champion.json"
            )
            self._champions[version] = {
                int(c["key"]): c["name"] for c in data["data"].values()
            }
        return self._champions[version]

    def items(self, version: str) -> dict[int, str]:
        if version not in self._items:
            data = self._cached_json(version, "item.json", f"/cdn/{version}/data/en_US/item.json")
            self._items[version] = {int(k): v["name"] for k, v in data["data"].items()}
        return self._items[version]


class ChampionNames:
    """Lookup that never fails: falls back to ``Champion 202`` when offline or unknown.

    Match JSON already carries ``championName``, so this is only needed for ids that
    arrive without one (baseline tables, CLI filters).
    """

    def __init__(self, ddragon: DataDragon | None, version: str | None = None) -> None:
        self._names: dict[int, str] = {}
        if ddragon is not None:
            try:
                self._names = ddragon.champions(version or ddragon.versions()[0])
            except (httpx.HTTPError, OSError, KeyError, ValueError):
                self._names = {}

    def __call__(self, champion_id: int) -> str:
        return self._names.get(champion_id, f"Champion {champion_id}")

    def id_for(self, name: str) -> int | None:
        wanted = name.replace("'", "").replace(" ", "").replace(".", "").lower()
        for cid, cname in self._names.items():
            if cname.replace("'", "").replace(" ", "").replace(".", "").lower() == wanted:
                return cid
        return None
