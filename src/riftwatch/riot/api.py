"""Typed wrappers over the Riot endpoints RiftWatch uses.

Each wrapper picks the right routing value (platform or regional cluster), names the
method for rate limiting, and returns Riot's JSON unchanged -- the raw responses are what
gets cached, and parsing happens later in :mod:`riftwatch.features`.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from riftwatch.riot.client import RiotClient
from riftwatch.riot.routing import (
    RiotId,
    account_region,
    match_region,
    platform_for,
    platform_of_match,
)

RANKED_SOLO_QUEUE_ID = 420
RANKED_FLEX_QUEUE_ID = 440
NORMAL_DRAFT_QUEUE_ID = 400
RANKED_SOLO = "RANKED_SOLO_5x5"
RANKED_FLEX = "RANKED_FLEX_SR"

# The modes RiftWatch analyses: Summoner's Rift with drafted roles.
QUEUE_NAMES = {RANKED_SOLO_QUEUE_ID: "Ranked Solo/Duo", RANKED_FLEX_QUEUE_ID: "Ranked Flex",
               NORMAL_DRAFT_QUEUE_ID: "Normal Draft"}
SUPPORTED_QUEUES = tuple(QUEUE_NAMES)
QUEUE_ALIASES = {"solo": RANKED_SOLO_QUEUE_ID, "flex": RANKED_FLEX_QUEUE_ID,
                 "draft": NORMAL_DRAFT_QUEUE_ID}


def parse_queues(text: str) -> tuple[int, ...]:
    """``"solo,flex"`` -> (420, 440)."""
    out = []
    for part in text.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in QUEUE_ALIASES:
            raise ValueError(f"unknown mode {part!r}; expected solo, flex or draft")
        out.append(QUEUE_ALIASES[key])
    if not out:
        raise ValueError("no modes given; expected solo, flex or draft")
    return tuple(dict.fromkeys(out))

TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND",
         "MASTER", "GRANDMASTER", "CHALLENGER")
APEX_TIERS = ("MASTER", "GRANDMASTER", "CHALLENGER")
DIVISIONS = ("I", "II", "III", "IV")


def _seg(value: str) -> str:
    """URL-encode one path segment (Riot IDs contain spaces and non-ASCII)."""
    return quote(value, safe="")


class RiotApi:
    def __init__(self, client: RiotClient) -> None:
        self.client = client

    # -- account-v1 (regional) -------------------------------------------------------------

    def account_by_riot_id(self, platform: str, riot_id: RiotId) -> dict[str, Any] | None:
        return self.client.get(
            account_region(platform),
            "account-v1.by-riot-id",
            f"/riot/account/v1/accounts/by-riot-id/{_seg(riot_id.game_name)}/{_seg(riot_id.tag_line)}",
        )

    def account_by_puuid(self, platform: str, puuid: str) -> dict[str, Any] | None:
        return self.client.get(
            account_region(platform),
            "account-v1.by-puuid",
            f"/riot/account/v1/accounts/by-puuid/{_seg(puuid)}",
        )

    # -- match-v5 (regional) ---------------------------------------------------------------

    def match_ids(
        self,
        platform: str,
        puuid: str,
        *,
        start: int = 0,
        count: int = 20,
        queue: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[str]:
        """Newest first. ``count`` is capped at 100 by Riot; times are epoch seconds."""
        if not 0 < count <= 100:
            raise ValueError("count must be 1-100")
        params: dict[str, Any] = {"start": start, "count": count}
        if queue is not None:
            params["queue"] = queue
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        ids = self.client.get(
            match_region(platform),
            "match-v5.ids-by-puuid",
            f"/lol/match/v5/matches/by-puuid/{_seg(puuid)}/ids",
            params,
        )
        return ids or []

    def match(self, match_id: str) -> dict[str, Any] | None:
        return self.client.get(
            match_region(platform_of_match(match_id)),
            "match-v5.match",
            f"/lol/match/v5/matches/{_seg(match_id)}",
        )

    def timeline(self, match_id: str) -> dict[str, Any] | None:
        return self.client.get(
            match_region(platform_of_match(match_id)),
            "match-v5.timeline",
            f"/lol/match/v5/matches/{_seg(match_id)}/timeline",
        )

    # -- league (platform) -----------------------------------------------------------------

    def league_entries_by_puuid(self, platform: str, puuid: str) -> list[dict[str, Any]]:
        entries = self.client.get(
            platform_for(platform),
            "league-v4.entries-by-puuid",
            f"/lol/league/v4/entries/by-puuid/{_seg(puuid)}",
        )
        return entries or []

    def league_entries(
        self, platform: str, tier: str, division: str, *, queue: str = RANKED_SOLO, page: int = 1
    ) -> list[dict[str, Any]]:
        """One page (~205 players) of a tier/division. Apex tiers only have division I.

        Uses league-exp-v4, which covers Master+ too, so one code path walks every tier.
        """
        tier, division = tier.upper(), division.upper()
        if tier not in TIERS or division not in DIVISIONS:
            raise ValueError(f"bad tier/division {tier} {division}")
        if tier in APEX_TIERS and division != "I":
            raise ValueError(f"{tier} has only division I")
        entries = self.client.get(
            platform_for(platform),
            "league-exp-v4.entries",
            f"/lol/league-exp/v4/entries/{queue}/{tier}/{division}",
            {"page": page},
        )
        return entries or []

    # -- spectator-v5 and champion-mastery-v4 (platform) -----------------------------------

    def active_game(self, platform: str, puuid: str) -> dict[str, Any] | None:
        """The game this player is in right now, or None if they aren't in one."""
        return self.client.get(
            platform_for(platform),
            "spectator-v5.active-game",
            f"/lol/spectator/v5/active-games/by-summoner/{_seg(puuid)}",
        )

    def champion_mastery(self, platform: str, puuid: str, champion_id: int) -> dict[str, Any] | None:
        """None if the player has never played the champion."""
        return self.client.get(
            platform_for(platform),
            "champion-mastery-v4.by-champion",
            f"/lol/champion-mastery/v4/champion-masteries/by-puuid/{_seg(puuid)}/by-champion/{int(champion_id)}",
        )
