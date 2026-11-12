"""Static game data for the website: champions, items, summoner spells, runes, and the URLs
of their icons on Riot's Data Dragon CDN (images load straight from Riot, as on op.gg).

JSON is fetched once per game version and cached on disk (riot.ddragon.DataDragon). If Data
Dragon can't be reached, the newest version already cached is used; failing that, pages still
render with names and empty icons.
"""

from __future__ import annotations

import html
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from riftwatch.riot.ddragon import DDRAGON, DataDragon

FALLBACK_VERSION = "16.22.1"
CDRAGON = "https://raw.communitydragon.org/latest/plugins"
TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER",
         "GRANDMASTER", "CHALLENGER")
POSITIONS = {"TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid", "BOTTOM": "Bot",
             "UTILITY": "Support"}
_POSITION_ICON = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "middle", "BOTTOM": "bottom",
                  "UTILITY": "utility"}
_TAG = re.compile(r"<[^>]+>")


def plain(text: str | None) -> str:
    """Data Dragon descriptions carry markup (<br>, <stats>...): text only."""
    text = re.sub(r"<br\s*/?>", " ", text or "")
    return html.unescape(_TAG.sub("", text)).strip()


@dataclass
class Champion:
    id: int
    key: str            # Data Dragon id, the image file name ("MonkeyKing")
    name: str           # display name ("Wukong")
    title: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return self.key.lower()


@dataclass
class Item:
    id: int
    name: str
    text: str
    gold: int
    tags: list[str]
    builds_into: list[str]
    depth: int

    @property
    def is_boots(self) -> bool:
        return "Boots" in self.tags and self.depth >= 2

    @property
    def is_legendary(self) -> bool:
        """A finished item: expensive and building into nothing further."""
        return (not self.builds_into and self.gold >= 2000 and not self.is_boots
                and "Consumable" not in self.tags and "Trinket" not in self.tags)

    @property
    def is_starter(self) -> bool:
        return self.gold <= 500 and not self.builds_into or "Lane" in self.tags


class GameData:
    def __init__(self, ddragon: DataDragon | None = None) -> None:
        self.dd = ddragon or DataDragon()
        self.version = FALLBACK_VERSION
        self.champions: dict[int, Champion] = {}
        self.by_slug: dict[str, Champion] = {}
        self.items: dict[int, Item] = {}
        self.spells: dict[int, dict[str, str]] = {}
        self.runes: dict[int, dict[str, str]] = {}       # perks and styles by id
        self.styles: dict[int, dict[str, Any]] = {}      # style id -> slots of perk ids
        self._details: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.loaded = False

    # -- loading ----------------------------------------------------------------------------

    def load(self) -> GameData:
        with self._lock:
            if self.loaded:
                return self
            try:
                self.version = self.dd.versions()[0]
            except Exception:
                cached = sorted((p.name for p in self.dd.cache_dir.glob("*")
                                 if (p / "champion.json").exists()),
                                key=lambda v: [int(x) for x in v.split(".") if x.isdigit()])
                self.version = cached[-1] if cached else FALLBACK_VERSION
            try:
                self._load_all()
            except Exception:
                pass                            # offline with no cache: names fall back
            self.loaded = True
        return self

    def _json(self, name: str, path: str) -> Any:
        return self.dd._cached_json(self.version, name, f"/cdn/{self.version}/data/en_US/{path}")

    def _load_all(self) -> None:
        for c in self._json("champion.json", "champion.json")["data"].values():
            champ = Champion(int(c["key"]), c["id"], c["name"], c.get("title", ""), c.get("tags", []))
            self.champions[champ.id] = champ
            self.by_slug[champ.slug] = champ
            self.by_slug[re.sub(r"[^a-z]", "", champ.name.lower())] = champ
        for k, v in self._json("item.json", "item.json")["data"].items():
            self.items[int(k)] = Item(int(k), v["name"], plain(v.get("plaintext") or v.get("description")),
                                      v.get("gold", {}).get("total", 0), v.get("tags", []),
                                      v.get("into", []), v.get("depth", 1))
        for s in self._json("summoner.json", "summoner.json")["data"].values():
            self.spells[int(s["key"])] = {"key": s["id"], "name": s["name"],
                                          "text": plain(s.get("description"))}
        for style in self._json("runesReforged.json", "runesReforged.json"):
            self.runes[style["id"]] = {"name": style["name"], "icon": style["icon"], "text": ""}
            self.styles[style["id"]] = {"name": style["name"], "icon": style["icon"],
                                        "slots": [[r["id"] for r in slot["runes"]]
                                                  for slot in style["slots"]]}
            for slot in style["slots"]:
                for r in slot["runes"]:
                    self.runes[r["id"]] = {"name": r["name"], "icon": r["icon"],
                                           "text": plain(r.get("shortDesc"))}

    def champion_detail(self, champ: Champion) -> dict | None:
        """Abilities and lore for one champion (fetched on first use)."""
        if champ.key not in self._details:
            try:
                data = self._json(f"champion-{champ.key}.json", f"champion/{champ.key}.json")
                self._details[champ.key] = data["data"][champ.key]
            except Exception:
                return None
        return self._details[champ.key]

    # -- lookups ----------------------------------------------------------------------------

    def champion(self, champion_id: int | None) -> Champion:
        if champion_id in self.champions:
            return self.champions[champion_id]
        return Champion(champion_id or 0, "", f"Champion {champion_id}")

    def champion_by_slug(self, slug: str) -> Champion | None:
        return self.by_slug.get(re.sub(r"[^a-z]", "", slug.lower()))

    def item(self, item_id: int) -> Item | None:
        return self.items.get(item_id)

    # -- image URLs -------------------------------------------------------------------------

    def champion_icon(self, champion_id: int | None) -> str:
        c = self.champion(champion_id)
        return f"{DDRAGON}/cdn/{self.version}/img/champion/{c.key}.png" if c.key else ""

    def champion_splash(self, champion_id: int) -> str:
        c = self.champion(champion_id)
        return f"{DDRAGON}/cdn/img/champion/splash/{c.key}_0.jpg" if c.key else ""

    def item_icon(self, item_id: int) -> str:
        return f"{DDRAGON}/cdn/{self.version}/img/item/{item_id}.png" if item_id else ""

    def spell_icon(self, spell_id: int) -> str:
        s = self.spells.get(spell_id)
        return f"{DDRAGON}/cdn/{self.version}/img/spell/{s['key']}.png" if s else ""

    def spell_name(self, spell_id: int) -> str:
        return self.spells.get(spell_id, {}).get("name", "")

    def rune_icon(self, rune_id: int | None) -> str:
        r = self.runes.get(rune_id or 0)
        return f"{DDRAGON}/cdn/img/{r['icon']}" if r else ""

    def rune_name(self, rune_id: int | None) -> str:
        return self.runes.get(rune_id or 0, {}).get("name", "")

    def ability_icon(self, image: str) -> str:
        return f"{DDRAGON}/cdn/{self.version}/img/spell/{image}"

    def passive_icon(self, image: str) -> str:
        return f"{DDRAGON}/cdn/{self.version}/img/passive/{image}"

    def profile_icon(self, icon_id: int | None) -> str:
        return f"{DDRAGON}/cdn/{self.version}/img/profileicon/{icon_id or 29}.png"

    @staticmethod
    def tier_crest(tier: str | None) -> str:
        t = (tier or "unranked").lower()
        return (f"{CDRAGON}/rcp-fe-lol-static-assets/global/default/images/ranked-mini-crests/"
                f"{t}.svg")

    @staticmethod
    def tier_emblem(tier: str | None) -> str:
        t = (tier or "unranked").lower()
        return (f"{CDRAGON}/rcp-fe-lol-static-assets/global/default/images/ranked-emblem/"
                f"emblem-{t}.png")

    @staticmethod
    def position_icon(position: str) -> str:
        p = _POSITION_ICON.get(position, "fill")
        return (f"{CDRAGON}/rcp-fe-lol-clash/global/default/assets/images/position-selector/"
                f"positions/icon-position-{p}.png")


_shared: GameData | None = None


def shared() -> GameData:
    """One GameData per process, loaded on first use."""
    global _shared
    if _shared is None:
        _shared = GameData()
    return _shared.load()
