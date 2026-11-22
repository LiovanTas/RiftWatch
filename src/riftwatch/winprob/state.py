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
# A richer reading of the same game state: which towers are down (an open base isn't two lost
# outer towers), plates, power spikes (players at 6, 11 and 16), gold not yet spent, how long
# the dead are still out (a death at 35 minutes isn't one at 6), and which objectives are up
# right now (a fight beside a live Baron matters more). Spawn times measured from stored
# games: dragon 5:00 and 5:00 after each, Baron 20:00 and 6:00 after, Elder 6:00 after a soul.
TOWER_TIERS = (("OUTER_TURRET", "outer"), ("INNER_TURRET", "inner"), ("BASE_TURRET", "base"),
               ("NEXUS_TURRET", "nexus"))
# Where the gold is: each lane's gold and XP gap against the player in the same position (a
# fed carry isn't a fed support, though the team totals are the same).
LANES = (("TOP", "top"), ("JUNGLE", "jungle"), ("MIDDLE", "mid"), ("BOTTOM", "bot"), ("UTILITY", "support"))
LANE_FEATURES = tuple(f"lane_{what}_{lane}" for _, lane in LANES for what in ("gold", "xp"))
# Vision: wards each team placed and cleared in the last few minutes (control wards over a
# longer window, as they last until killed), and how long until the next dragon, Elder and
# Baron (a fight a minute before Baron spawns isn't one twenty minutes before).
WARD_WINDOW_S, CONTROL_WINDOW_S = 180, 300
VISION_FEATURES = ("wards_own", "wards_enemy", "control_wards_own", "control_wards_enemy",
                   "wards_cleared_own", "wards_cleared_enemy")
TIMER_FEATURES = ("dragon_in", "elder_in", "baron_in")
EXTRA_FEATURES = (tuple(f"towers_{name}_{side}" for _, name in TOWER_TIERS for side in ("own", "enemy"))
                  + ("plates_own", "plates_enemy", "lvl6_own", "lvl6_enemy", "lvl11_own", "lvl11_enemy",
                     "lvl16_own", "lvl16_enemy", "held_gold_own", "held_gold_enemy",
                     "dead_time_own", "dead_time_enemy", "dragon_up", "elder_up", "baron_up")
                  + LANE_FEATURES + VISION_FEATURES + TIMER_FEATURES)
WARD_TYPES = {"YELLOW_TRINKET": "wards", "SIGHT_WARD": "wards", "BLUE_TRINKET": "wards", "CONTROL_WARD": "control"}
RICH_FEATURES = BASE_FEATURES + EXTRA_FEATURES
DRAGON_SPAWN_S, DRAGON_RESPAWN_S, BARON_SPAWN_S, BARON_RESPAWN_S, ELDER_RESPAWN_S = 300, 300, 1200, 360, 360
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
FULL_FEATURES = RICH_FEATURES + ITEM_FEATURES + COMP_FEATURES
KEPT_EVENTS = ("CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL", "DRAGON_SOUL_GIVEN",
               "TURRET_PLATE_DESTROYED", "WARD_PLACED", "WARD_KILL", "GAME_END")
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
                 catalog: Catalog | None = None, positions: dict[int, str] | None = None) -> None:
        """``champions``: participant id -> champion id. ``purchases``: participant id ->
        [(second, item id), ...] (the timeline's purchases, undos taken out). ``positions``:
        participant id -> TOP ... UTILITY (for each lane's gaps)."""
        self.frames = frames
        # Each lane's two players: (blue pid, red pid), when both are known.
        self.lanes: dict[str, tuple[int, int]] = {}
        for pos, lane in LANES:
            pair_ = [p for p, q in (positions or {}).items() if q == pos]
            blue = [p for p in pair_ if 1 <= p <= 5]
            red = [p for p in pair_ if 6 <= p <= 10]
            if len(blue) == 1 and len(red) == 1:
                self.lanes[lane] = (int(blue[0]), int(red[0]))
        self._player = {(pid, key): [float((f.get(str(pid)) or {}).get(src, 0)) for f in frames]
                        for pid in {p for pair_ in self.lanes.values() for p in pair_}
                        for key, src in (("gold", "totalGold"), ("xp", "xp"))}
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
                                           ("level", ("level",)), ("held", ("currentGold",)))
                        for team in (100, 200)}

        def at_least(frame: dict, team: int, level: int) -> float:
            pids = range(1, 6) if team == 100 else range(6, 11)
            return float(sum(1 for p in pids if (frame.get(str(p)) or {}).get("level", 0) >= level))
        for level in (6, 11, 16):
            for team in (100, 200):
                self._totals[(f"lvl{level}", team)] = [at_least(f, team, level) for f in frames]

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
                    tier = dict(TOWER_TIERS).get(e.get("towerType"))
                    if tier:
                        counts[(f"towers_{tier}_lost", lost)] = counts.get((f"towers_{tier}_lost", lost), 0) + 1
            elif kind == "TURRET_PLATE_DESTROYED":
                lost = _side(e.get("teamId"))
                counts[("plates_lost", lost)] = counts.get(("plates_lost", lost), 0) + 1
            elif kind == "ELITE_MONSTER_KILL":
                taker, monster = _side(e.get("killerTeamId")), e.get("monsterType")
                if taker == OTHER:
                    # Neutral (team 300): a summoned herald expiring, nobody's take.
                    self._prefix.append(dict(counts))
                    continue
                if monster in MONSTERS:
                    counts[(MONSTERS[monster], taker)] = counts.get((MONSTERS[monster], taker), 0) + 1
                if monster == "ELDER_DRAGON":
                    self._timed.append((i, "elder", taker, ts + ELDER_BUFF_S * 1000))
                    counts[("last", "elder")] = ts
                elif monster == "BARON_NASHOR":
                    self._timed.append((i, "baron", taker, ts + BARON_BUFF_S * 1000))
                    counts[("last", "baron")] = ts
                elif monster == "DRAGON":
                    counts[("last", "dragon")] = ts
            elif kind == "DRAGON_SOUL_GIVEN":
                counts[("soul", _side(e.get("teamId")))] = 1
                counts[("last", "soul")] = ts
            self._prefix.append(dict(counts))

        # Wards placed (by kind) and cleared, by team, in time order.
        self._wards: dict[tuple[str, int], list[float]] = {(k, team): [] for k in ("wards", "control", "cleared")
                                                           for team in (100, 200)}
        for e in self.events:
            if e.get("type") == "WARD_PLACED" and 1 <= (e.get("creatorId") or 0) <= 10:
                kind = WARD_TYPES.get(e.get("wardType"))
                if kind:
                    self._wards[(kind, team_of(e["creatorId"]))].append(e.get("timestamp", 0))
            elif e.get("type") == "WARD_KILL" and 1 <= (e.get("killerId") or 0) <= 10:
                self._wards[("cleared", team_of(e["killerId"]))].append(e.get("timestamp", 0))

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

    def _player_value(self, pid: int, key: str, t: float) -> float:
        """A player's gold or XP at ``t``, interpolated between minute frames."""
        values = self._player.get((pid, key)) or []
        if not values:
            return 0.0
        i = int(t // 60_000)
        if i >= len(values) - 1:
            return values[-1]
        return values[i] + (values[i + 1] - values[i]) * ((t - i * 60_000) / 60_000)

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
             item_extra: dict[int, Counter] | None = None,
             player_bonus: dict[int, float] | None = None) -> tuple[dict[str, float], dict[str, float]]:
        """The state at time ``t`` (ms) from blue's side and from red's, counting the first
        ``upto`` events (all events up to ``t`` by default) and items bought up to
        ``item_time`` (``t`` by default). ``gold_bonus`` adds gold to a team (a kill's bounty,
        which the minute frames only show later) and ``player_bonus`` the same to a player's
        lane; ``item_extra`` adds item flags to a team (an item someone could have bought
        instead)."""
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
        dead_time = {100: 0.0, 200: 0.0}                       # minutes the dead are still out
        for i, v, back in self.deaths:
            if i < upto and back > t and 1 <= v <= 10:
                dead[team_of(v)].add(v)
                dead_time[team_of(v)] += (back - t) / 60_000
        # Which objectives are up right now (from spawn and respawn timers).
        last = {k: c.get(("last", k)) for k in ("dragon", "elder", "baron", "soul")}
        soul_taken = last["soul"] is not None
        dragon_up = float(not soul_taken and t >= DRAGON_SPAWN_S * 1000
                          and (last["dragon"] is None or t - last["dragon"] >= DRAGON_RESPAWN_S * 1000))
        since = max(x for x in (last["soul"], last["elder"], last["dragon"]) if x is not None) if soul_taken else None
        elder_up = float(soul_taken and t - since >= ELDER_RESPAWN_S * 1000)
        baron_up = float(t >= BARON_SPAWN_S * 1000
                         and (last["baron"] is None or t - last["baron"] >= BARON_RESPAWN_S * 1000))
        # Minutes until each is up again (0 when up), capped: beyond a few minutes it's all "later".
        if soul_taken:
            dragon_in = 5.0
        elif dragon_up:
            dragon_in = 0.0
        else:
            due = DRAGON_SPAWN_S * 1000 if last["dragon"] is None else last["dragon"] + DRAGON_RESPAWN_S * 1000
            dragon_in = min(5.0, max(0.0, (due - t) / 60_000))
        elder_in = (6.0 if not soul_taken else 0.0 if elder_up
                    else min(6.0, max(0.0, (since + ELDER_RESPAWN_S * 1000 - t) / 60_000)))
        if baron_up:
            baron_in = 0.0
        else:
            due = BARON_SPAWN_S * 1000 if last["baron"] is None else last["baron"] + BARON_RESPAWN_S * 1000
            baron_in = min(6.0, max(0.0, (due - t) / 60_000))

        def recent(kind: str, team: int, window_s: int) -> float:
            times = self._wards[(kind, team)]
            return float(bisect_right(times, t) - bisect_right(times, t - window_s * 1000))
        lane_gap = {}                                           # blue's gap in each lane
        for lane, (b, r) in self.lanes.items():
            extra = (player_bonus or {}).get(b, 0.0) - (player_bonus or {}).get(r, 0.0)
            lane_gap[lane] = ((self._player_value(b, "gold", t) - self._player_value(r, "gold", t) + extra) / 1000,
                              (self._player_value(b, "xp", t) - self._player_value(r, "xp", t)) / 1000)
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
            for _, tier in TOWER_TIERS:
                f[f"towers_{tier}_own"] = n(f"towers_{tier}_lost", enemy)
                f[f"towers_{tier}_enemy"] = n(f"towers_{tier}_lost", own) + n(f"towers_{tier}_lost", OTHER)
            f["plates_own"] = n("plates_lost", enemy)
            f["plates_enemy"] = n("plates_lost", own) + n("plates_lost", OTHER)
            for spike in (6, 11, 16):
                f[f"lvl{spike}_own"] = self._total(f"lvl{spike}", own, t, False)
                f[f"lvl{spike}_enemy"] = self._total(f"lvl{spike}", enemy, t, False)
            f["held_gold_own"] = self._total("held", own, t, False) / 1000
            f["held_gold_enemy"] = self._total("held", enemy, t, False) / 1000
            f["dead_time_own"] = dead_time[own]
            f["dead_time_enemy"] = dead_time[enemy]
            f["dragon_up"], f["elder_up"], f["baron_up"] = dragon_up, elder_up, baron_up
            f["dragon_in"], f["elder_in"], f["baron_in"] = dragon_in, elder_in, baron_in
            for side, team in (("own", own), ("enemy", enemy)):
                f[f"wards_{side}"] = recent("wards", team, WARD_WINDOW_S)
                f[f"control_wards_{side}"] = recent("control", team, CONTROL_WINDOW_S)
                f[f"wards_cleared_{side}"] = recent("cleared", team, WARD_WINDOW_S)
            sign = 1.0 if own == 100 else -1.0
            for _, lane in LANES:
                gold_gap, xp_gap = lane_gap.get(lane, (0.0, 0.0))
                f[f"lane_gold_{lane}"] = sign * gold_gap
                f[f"lane_xp_{lane}"] = sign * xp_gap
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
