"""Which items count as finished builds, per patch, from Data Dragon.

Riot's item ids say nothing about an item's tier, so the rules come from each patch's item
data:

* **legendary** -- a purchasable Summoner's Rift item that builds into nothing, costs at
  least 2,200 gold, and isn't a consumable, trinket or boots (138 items on 16.19);
* **finished boots** -- any boots item costing at least 900 gold (tier 2 and the tier 3
  upgrades).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LEGENDARY_GOLD = 2200
BOOTS_GOLD = 900
_NOT_LEGENDARY_TAGS = {"Consumable", "Trinket", "Boots"}


@dataclass(frozen=True)
class ItemRules:
    legendary: frozenset[int]
    boots: frozenset[int]

    @classmethod
    def from_ddragon(cls, data: dict[str, Any]) -> ItemRules:
        legendary, boots = set(), set()
        for key, item in data.items():
            gold = item.get("gold") or {}
            on_rift = (item.get("maps") or {}).get("11", False)
            if not (gold.get("purchasable") and on_rift):
                continue
            tags = set(item.get("tags") or [])
            total = gold.get("total", 0)
            if "Boots" in tags and total >= BOOTS_GOLD:
                boots.add(int(key))
            elif not item.get("into") and total >= LEGENDARY_GOLD and not tags & _NOT_LEGENDARY_TAGS:
                legendary.add(int(key))
        return cls(frozenset(legendary), frozenset(boots))


def purchases(events: list[dict[str, Any]], participant_id: int) -> list[tuple[int, int]]:
    """(timestamp ms, item id) bought by the player, minus purchases they undid."""
    bought: list[tuple[int, int]] = []
    for e in events:
        if e.get("participantId") != participant_id:
            continue
        if e["type"] == "ITEM_PURCHASED":
            bought.append((e["timestamp"], e.get("itemId", 0)))
        elif e["type"] == "ITEM_UNDO" and e.get("beforeId"):
            for i in range(len(bought) - 1, -1, -1):
                if bought[i][1] == e["beforeId"]:
                    del bought[i]
                    break
    return bought


def build_timings(events: list[dict[str, Any]], participant_id: int, rules: ItemRules) -> dict[str, float]:
    """Minutes at which the player finished their first and second legendary and boots."""
    out: dict[str, float] = {}
    legendaries: list[int] = []
    for ts, item in purchases(events, participant_id):
        minute = round(ts / 60_000, 2)
        if item in rules.legendary and item not in legendaries:
            legendaries.append(item)
            if len(legendaries) == 1:
                out["first_item_min"] = minute
            elif len(legendaries) == 2:
                out["second_item_min"] = minute
        elif item in rules.boots and "boots_min" not in out:
            out["boots_min"] = minute
    return out
