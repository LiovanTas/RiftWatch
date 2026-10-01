"""Riot API routing: which host serves which request.

Riot splits its API across two kinds of host:

* **Platform** hosts (``na1.api.riotgames.com``) serve per-shard data -- league-v4,
  summoner-v4, spectator-v5.
* **Regional** hosts (``americas.api.riotgames.com``) serve cross-shard data -- account-v1
  and match-v5.

Every platform belongs to exactly one region for match-v5. account-v1 has no ``sea``
cluster, so SEA platforms resolve accounts through ``asia``.
"""

from __future__ import annotations

from dataclasses import dataclass

# platform -> (match-v5 region, account-v1 region)
_PLATFORMS: dict[str, tuple[str, str]] = {
    "na1": ("americas", "americas"),
    "br1": ("americas", "americas"),
    "la1": ("americas", "americas"),
    "la2": ("americas", "americas"),
    "euw1": ("europe", "europe"),
    "eun1": ("europe", "europe"),
    "tr1": ("europe", "europe"),
    "ru": ("europe", "europe"),
    "me1": ("europe", "europe"),
    "kr": ("asia", "asia"),
    "jp1": ("asia", "asia"),
    "oc1": ("sea", "asia"),
    "sg2": ("sea", "asia"),
    "tw2": ("sea", "asia"),
    "vn2": ("sea", "asia"),
}

# What people type (op.gg-style short names) -> platform.
_ALIASES: dict[str, str] = {
    "na": "na1",
    "br": "br1",
    "lan": "la1",
    "las": "la2",
    "euw": "euw1",
    "eune": "eun1",
    "tr": "tr1",
    "me": "me1",
    "jp": "jp1",
    "oce": "oc1",
    "sg": "sg2",
    "sea": "sg2",
    "tw": "tw2",
    "vn": "vn2",
}


class UnknownRegion(ValueError):
    pass


def platform_for(region: str) -> str:
    """Normalise a user-typed region (``"NA"``, ``"na1"``, ``"euw"``) to a platform id."""
    key = region.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _PLATFORMS:
        known = ", ".join(sorted(set(_ALIASES) | set(_PLATFORMS)))
        raise UnknownRegion(f"unknown region {region!r}; expected one of: {known}")
    return key


def match_region(platform: str) -> str:
    """Regional cluster that serves match-v5 for this platform."""
    return _PLATFORMS[platform_for(platform)][0]


def account_region(platform: str) -> str:
    """Regional cluster that serves account-v1 for this platform."""
    return _PLATFORMS[platform_for(platform)][1]


# Every host label the API answers on: platforms plus the regional clusters.
ROUTING_VALUES = frozenset(_PLATFORMS) | {r for pair in _PLATFORMS.values() for r in pair}


def platform_host(platform: str) -> str:
    return f"https://{platform_for(platform)}.api.riotgames.com"


def regional_host(region: str) -> str:
    return f"https://{region}.api.riotgames.com"


def platform_of_match(match_id: str) -> str:
    """``"NA1_5123456789"`` -> ``"na1"``. Match ids carry the platform they were played on."""
    prefix, sep, _ = match_id.partition("_")
    if not sep:
        raise ValueError(f"not a match id: {match_id!r}")
    return platform_for(prefix)


@dataclass(frozen=True)
class RiotId:
    game_name: str
    tag_line: str

    def __str__(self) -> str:
        return f"{self.game_name}#{self.tag_line}"


def parse_riot_id(text: str) -> RiotId:
    """Parse ``"Name#TAG"``. Game names may contain spaces; the tag is after the last ``#``."""
    name, sep, tag = text.strip().rpartition("#")
    name, tag = name.strip(), tag.strip()
    if not sep or not name or not tag:
        raise ValueError(f"expected a Riot ID like 'Name#TAG', got {text!r}")
    # Length rules are left to the API: legacy accounts predate the current limits.
    return RiotId(name, tag)
