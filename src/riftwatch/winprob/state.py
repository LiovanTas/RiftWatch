"""A game's state at any moment, from its timeline, as the win-probability model's inputs.

Timelines give each player's gold, XP and level once a minute, and every kill, objective and
structure with a timestamp. ``Replay`` rebuilds the state at any time ``t`` from one team's
side: gold and XP leads (interpolated between the minute frames), kills, towers, inhibitors
currently down, dragons and soul, Baron, Elder and their active buffs, heralds, voidgrubs,
Atakhan, and how many players on each side are alive (from respawn timers).

With a ``Catalog`` (what each finished item gives and what each champion is) it also reads:

  * items -- each team's finished (legendary) items so far, counted by what they give: armor,
    magic resist, health, attack damage, ability power, crit, attack speed, armor and magic
    penetration, lifesteal and omnivamp, anti-heal. Bought, from the timeline's purchases.
  * composition -- each team's champions by class (tank, fighter, mage, marksman, assassin,
    support) and how much of its damage is magic, from what those champions deal in stored
    games; and which champions they are (``champions``), so the network can learn that some
    champions win early and others late.

Participant ids 1-5 are blue (team 100) and 6-10 red (team 200), as in every Riot timeline.
Team totals per minute and running counts of every event are worked out once per game, so
asking for a state is cheap (Swing asks a few hundred times a game).
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

BASE_FEATURES = (
    "minute", "gold_diff", "xp_diff", "cs_diff", "level_diff",
    "kills_own", "kills_enemy", "towers_own", "towers_enemy",
    "inhibs_down_enemy", "inhibs_down_own",
    "dragons_own", "dragons_enemy", "soul_own", "soul_enemy",
    "elder_own", "elder_enemy", "baron_own", "baron_enemy", "barons_own", "barons_enemy",
    "heralds_own", "heralds_enemy", "grubs_own", "grubs_enemy", "atakhan_own", "atakhan_enemy",
    "alive_own", "alive_enemy",
)
FEATURES = BASE_FEATURES                       # what the first model (wp1) reads
# What a finished item can give, as the model counts it per team.
ITEM_FLAGS = ("items", "armor", "mr", "health", "ad", "ap", "crit", "attack_speed",
              "armor_pen", "magic_pen", "sustain", "anti_heal")
ITEM_FLAG_NAMES = {"items": "finished items", "armor": "armor", "mr": "magic resist", "health": "health",
                   "ad": "attack damage", "ap": "ability power", "crit": "crit", "attack_speed": "attack speed",
                   "armor_pen": "armor penetration", "magic_pen": "magic penetration",
                   "sustain": "lifesteal", "anti_heal": "anti-heal"}
CLASSES = ("Tank", "Fighter", "Mage", "Marksman", "Assassin", "Support")
ITEM_FEATURES = tuple(f"{f}_{side}" for f in ITEM_FLAGS for side in ("own", "enemy"))
COMP_FEATURES = (tuple(f"{c.lower()}s_{side}" for c in CLASSES for side in ("own", "enemy"))
                 + ("magic_share_own", "magic_share_enemy"))
FULL_FEATURES = BASE_FEATURES + ITEM_FEATURES + COMP_FEATURES
KEPT_EVENTS = ("CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL", "DRAGON_SOUL_GIVEN",
               "TURRET_PLATE_DESTROYED", "GAME_END")
BARON_BUFF_S, ELDER_BUFF_S, INHIB_RESPAWN_S = 180, 150, 300
# Base respawn time by level (seconds), before the late-game increase.
BASE_RESPAWN = (6, 6, 8, 8, 10, 12, 16, 21, 26, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5)
MONSTERS = {"DRAGON": "dragons", "BARON_NASHOR": "barons", "RIFTHERALD": "heralds",
            "HORDE": "grubs", "ATAKHAN": "atakhan"}
OTHER = 0                       # a team id that's neither side (missing in the event)
ANTI_HEAL = {"Thornmail", "Mortal Reminder", "Morellonomicon", "Chempunk Chainsword"}
# Data Dragon item tags behind each flag.
_ITEM_TAGS = {"armor": ("Armor",), "mr": ("SpellBlock", "MagicResist"), "health": ("Health",),
              "ad": ("Damage",), "ap": ("SpellDamage",), "crit": ("CriticalStrike",),
              "attack_speed": ("AttackSpeed",), "armor_pen": ("ArmorPenetration",),
              "magic_pen": ("MagicPenetration",), "sustain": ("LifeSteal", "SpellVamp")}


def team_of(pid: int) -> int:
    return 100 if 1 <= pid <= 5 else 200


def respawn_seconds(level: int, minute: float) -> float:
    """Approximate death timer: the base by level, longer as the game goes on."""
    base = BASE_RESPAWN[max(1, min(level, 18)) - 1]
    if minute < 15:
        extra = 0.0
    elif minute < 30:
        extra = 0.0085 * (minute - 15)
    elif minute < 45:
        extra = 0.1275 + 0.006 * (minute - 30)
    else:
        extra = min(0.5, 0.2175 + 0.029 * (minute - 45))
    return base * (1 + extra)


def compact(timeline: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """(participant frames per minute, the events the model uses) from a full timeline."""
    frames = timeline.get("info", {}).get("frames", [])
    pfs = [f.get("participantFrames") or {} for f in frames]
    events = [e for f in frames for e in f.get("events", []) if e.get("type") in KEPT_EVENTS]
    return pfs, events


def _side(team_id) -> int:
    return team_id if team_id in (100, 200) else OTHER


@dataclass
class Catalog:
    """What the model knows about items and champions, saved with it so a model always reads
    games the way it was trained."""
    items: dict[int, tuple[str, ...]] = field(default_factory=dict)      # finished item -> flags
    champions: dict[int, tuple[str, float]] = field(default_factory=dict)  # id -> (class, magic share)
    index: dict[int, int] = field(default_factory=dict)                  # champion id -> column
    item_names: dict[int, str] = field(default_factory=dict)
    popular: dict[int, list[int]] = field(default_factory=dict)          # champion -> its common items

    @staticmethod
    def item_flags(tags: list[str], text: str, name: str = "") -> tuple[str, ...]:
        flags = ["items"] + [f for f, wanted in _ITEM_TAGS.items() if any(t in tags for t in wanted)]
        # Riot's text says "Wounds" (once "Grievous Wounds"); some items only describe it.
        if re.search(r"Wounds|high Health recovery", text or "") or name in ANTI_HEAL:
            flags.append("anti_heal")
        return tuple(flags)

    def comp(self, champions: list[int]) -> dict[str, float]:
        """Class counts and magic damage share of a team's champions."""
        known = [self.champions.get(c) for c in champions]
        classes = Counter(k[0] for k in known if k)
        shares = [k[1] for k in known if k]
        out = {c: float(classes.get(c, 0)) for c in CLASSES}
        out["magic_share"] = sum(shares) / len(shares) if shares else 0.4
        return out


class Replay:
    def __init__(self, frames: list[dict], events: list[dict], champions: dict[int, int] | None = None,
                 purchases: dict[int, list[tuple[int, int]]] | None = None,
                 catalog: Catalog | None = None) -> None:
        """``champions``: participant id -> champion id. ``purchases``: participant id ->
        [(second, item id), ...] (the timeline's purchases, undos taken out)."""
        self.frames = frames
        self.events = sorted((e for e in events if e.get("type") in KEPT_EVENTS),
                             key=lambda e: e.get("timestamp", 0))
        self.times = [e.get("timestamp", 0) for e in self.events]
        self.winner = next((e.get("winningTeam") for e in self.events if e.get("type") == "GAME_END"), None)
        self.end_ms = max(self.times[-1] if self.times else 0, (len(frames) - 1) * 60_000)
        self.catalog = catalog
        self.champions = champions or {}

        # Each team's totals at each minute frame.
        def total(frame: dict, team: int, *keys: str) -> float:
            pids = range(1, 6) if team == 100 else range(6, 11)
            return float(sum((frame.get(str(p)) or {}).get(k, 0) for p in pids for k in keys))
        self._totals = {(key, team): [total(f, team, *parts) for f in frames]
                        for key, parts in (("gold", ("totalGold",)), ("xp", ("xp",)),
                                           ("cs", ("minionsKilled", "jungleMinionsKilled")),
                                           ("level", ("level",)))
                        for team in (100, 200)}

        # Running counts after each event (index k = after the first k events), by team, and
        # the events whose effect wears off: inhibitors, Baron and Elder buffs, deaths.
        counts: dict[tuple[str, int], int] = {}
        self._prefix = [dict(counts)]
        self._timed: list[tuple[int, str, int, float]] = []          # (index, kind, team, until ms)
        self.deaths: list[tuple[int, int, float]] = []               # (index, victim, back ms)
        for i, e in enumerate(self.events):
            kind, ts = e.get("type"), e.get("timestamp", 0)
            if kind == "CHAMPION_KILL":
                victim = e.get("victimId", 0)
                key = ("kills_by_victim_team", team_of(victim))
                counts[key] = counts.get(key, 0) + 1
                level = int(self._frame_value(victim, "level", ts)) or 1
                self.deaths.append((i, victim, ts + 1000 * respawn_seconds(level, ts / 60_000)))
            elif kind == "BUILDING_KILL":
                lost = _side(e.get("teamId"))
                if e.get("buildingType") == "INHIBITOR_BUILDING":
                    self._timed.append((i, "inhib", lost, ts + INHIB_RESPAWN_S * 1000))
                else:
                    counts[("towers_lost", lost)] = counts.get(("towers_lost", lost), 0) + 1
            elif kind == "ELITE_MONSTER_KILL":
                taker, monster = _side(e.get("killerTeamId")), e.get("monsterType")
                if monster in MONSTERS:
                    counts[(MONSTERS[monster], taker)] = counts.get((MONSTERS[monster], taker), 0) + 1
                if monster == "ELDER_DRAGON":
                    self._timed.append((i, "elder", taker, ts + ELDER_BUFF_S * 1000))
                elif monster == "BARON_NASHOR":
                    self._timed.append((i, "baron", taker, ts + BARON_BUFF_S * 1000))
            elif kind == "DRAGON_SOUL_GIVEN":
                counts[("soul", _side(e.get("teamId")))] = 1
            self._prefix.append(dict(counts))

        # Finished items, in the order they were bought: (ms, participant, item id).
        self.items: list[tuple[int, int, int]] = []
        if catalog is not None:
            for pid, bought in (purchases or {}).items():
                for second, item in bought:
                    if item in catalog.items:
                        self.items.append((1000 * int(second), int(pid), int(item)))
            self.items.sort()
        self._item_times = [t for t, _, _ in self.items]
        self._comp = {}
        if catalog is not None:
            for team in (100, 200):
                pids = range(1, 6) if team == 100 else range(6, 11)
                self._comp[team] = catalog.comp([self.champions.get(p, 0) for p in pids])

    # -- per-player numbers -----------------------------------------------------------------
    def _frame_value(self, pid: int, key: str, t: float) -> float:
        i = min(int(t // 60_000), len(self.frames) - 1)
        return float((self.frames[i].get(str(pid)) or {}).get(key, 0)) if self.frames else 0.0

    def frame_value(self, pid: int, key: str, t: float) -> float:
        """A player's value from the last minute frame at or before ``t``."""
        return self._frame_value(pid, key, t)

    def _total(self, key: str, team: int, t: float, interpolate: bool) -> float:
        values = self._totals[(key, team)]
        if not values:
            return 0.0
        i = int(t // 60_000)
        if i >= len(values) - 1:
            return values[-1]
        if not interpolate:
            return values[i]
        return values[i] + (values[i + 1] - values[i]) * ((t - i * 60_000) / 60_000)

    def item_counts(self, item_time: float) -> dict[int, Counter]:
        """Each team's finished items bought up to ``item_time`` (ms), by flag."""
        out = {100: Counter(), 200: Counter()}
        if self.catalog is None:
            return out
        for _, pid, item in self.items[: bisect_right(self._item_times, item_time)]:
            out[team_of(pid)].update(self.catalog.items[item])
        return out

    def dead_at(self, t: float, upto: int | None = None) -> set[int]:
        """Players dead at ``t`` (counting the first ``upto`` events)."""
        if upto is None:
            upto = bisect_right(self.times, t)
        return {v for i, v, back in self.deaths if i < upto and back > t and 1 <= v <= 10}

    # -- the state ---------------------------------------------------------------------------
    def pair(self, t: float, upto: int | None = None, gold_bonus: dict[int, float] | None = None,
             item_time: float | None = None,
             item_extra: dict[int, Counter] | None = None) -> tuple[dict[str, float], dict[str, float]]:
        """The state at time ``t`` (ms) from blue's side and from red's, counting the first
        ``upto`` events (all events up to ``t`` by default) and items bought up to
        ``item_time`` (``t`` by default). ``gold_bonus`` adds gold to a team (a kill's bounty,
        which the minute frames only show later); ``item_extra`` adds item flags to a team
        (an item someone could have bought instead)."""
        if upto is None:
            upto = bisect_right(self.times, t)
        c = self._prefix[upto]

        def n(key: str, team: int) -> float:
            return float(c.get((key, team), 0))

        gold = {team: self._total("gold", team, t, True) + (gold_bonus or {}).get(team, 0.0) for team in (100, 200)}
        xp = {team: self._total("xp", team, t, True) for team in (100, 200)}
        cs = {team: self._total("cs", team, t, False) for team in (100, 200)}
        level = {team: self._total("level", team, t, False) / 5 for team in (100, 200)}
        active = {(kind, team): 0 for kind in ("inhib", "elder", "baron") for team in (100, 200, OTHER)}
        for i, kind, team, until in self._timed:
            if i < upto and t < until:
                active[(kind, team)] += 1
        dead = {100: set(), 200: set()}
        for v in self.dead_at(t, upto):
            dead[team_of(v)].add(v)
        items = None
        if self.catalog is not None:
            items = self.item_counts(t if item_time is None else item_time)
            for team, extra in (item_extra or {}).items():
                items[team] = items[team] + extra

        out = []
        for own in (100, 200):
            enemy = 300 - own
            f = dict.fromkeys(BASE_FEATURES, 0.0)
            f["minute"] = t / 60_000
            f["gold_diff"] = (gold[own] - gold[enemy]) / 1000
            f["xp_diff"] = (xp[own] - xp[enemy]) / 1000
            f["cs_diff"] = (cs[own] - cs[enemy]) / 10
            f["level_diff"] = level[own] - level[enemy]
            f["kills_own"] = n("kills_by_victim_team", enemy)
            f["kills_enemy"] = n("kills_by_victim_team", own)
            f["towers_own"] = n("towers_lost", enemy)
            f["towers_enemy"] = n("towers_lost", own) + n("towers_lost", OTHER)
            f["inhibs_down_enemy"] = float(active[("inhib", enemy)])
            f["inhibs_down_own"] = float(active[("inhib", own)] + active[("inhib", OTHER)])
            for key in ("dragons", "barons", "heralds", "grubs", "atakhan"):
                f[f"{key}_own"] = n(key, own)
                f[f"{key}_enemy"] = n(key, enemy) + n(key, OTHER)
            f["soul_own"] = float(n("soul", own) > 0)
            f["soul_enemy"] = float(n("soul", enemy) > 0 or n("soul", OTHER) > 0)
            for buff in ("elder", "baron"):
                f[f"{buff}_own"] = float(active[(buff, own)] > 0)
                f[f"{buff}_enemy"] = float(active[(buff, enemy)] > 0 or active[(buff, OTHER)] > 0)
            f["alive_own"] = float(5 - len(dead[own]))
            f["alive_enemy"] = float(5 - len(dead[enemy]))
            if items is not None:
                for flag in ITEM_FLAGS:
                    f[f"{flag}_own"] = float(items[own].get(flag, 0))
                    f[f"{flag}_enemy"] = float(items[enemy].get(flag, 0))
                for side, team in (("own", own), ("enemy", enemy)):
                    comp = self._comp[team]
                    for cls in CLASSES:
                        f[f"{cls.lower()}s_{side}"] = comp[cls]
                    f[f"magic_share_{side}"] = comp["magic_share"]
                f["_own"] = [self.champions.get(p, 0) for p in (range(1, 6) if own == 100 else range(6, 11))]
                f["_enemy"] = [self.champions.get(p, 0) for p in (range(6, 11) if own == 100 else range(1, 6))]
            out.append(f)
        return out[0], out[1]

    def state(self, t: float, team: int, upto: int | None = None,
              gold_bonus: dict[int, float] | None = None) -> dict[str, float]:
        """The state at time ``t`` from ``team``'s side (see ``pair``)."""
        blue, red = self.pair(t, upto, gold_bonus)
        return blue if team == 100 else red

    def minute_states(self, team: int) -> list[dict[str, float]]:
        """The state at every whole minute after the first, from ``team``'s side."""
        return [self.state(m * 60_000, team) for m in range(1, len(self.frames))]
