"""What a player did well, and less well, in one game: short notes for the match card and
the game's Insights tab.

Each note compares the player with something concrete:
  * the other nine players in the game (most damage, most vision, kill participation);
  * their lane opponent (CS, gold and, from the timeline, the lead at 10 and 15 minutes);
  * players at their rank in their role, from every stored ranked game (``baselines``):
    farm, vision, damage and deaths per minute, control wards.

A rank comparison only counts when the player is in the top or bottom fifth, so the notes
point at what stood out, not at everything that was a little above or below average.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import psycopg

ROLE_NAME = {"TOP": "top laners", "JUNGLE": "junglers", "MIDDLE": "mid laners",
             "BOTTOM": "bot laners", "UTILITY": "supports"}
TIER_NAME = {"IRON": "Iron", "BRONZE": "Bronze", "SILVER": "Silver", "GOLD": "Gold",
             "PLATINUM": "Platinum", "EMERALD": "Emerald", "DIAMOND": "Diamond",
             "MASTER_PLUS": "Master+", "ALL": "ranked"}
APEX = ("MASTER", "GRANDMASTER", "CHALLENGER")

# metric -> SQL over participant_builds rows (per-minute rates unless noted)
METRICS = {
    "cs_min": "cs / (duration_s / 60.0)",
    "vision_min": "vision_score / (duration_s / 60.0)",
    "dmg_min": "damage / (duration_s / 60.0)",
    "deaths_10": "deaths / (duration_s / 600.0)",
    "kp": "(kills + assists)::float8 / greatest(team_kills, 1)",
    "control_wards": "control_wards::float8",
}
LOW, HIGH = 0.2, 0.8                 # the quantiles that count as standing out
MIN_GAMES = 150                      # fewer games in a rank and role -> use all ranks
TTL_S = 1800

_cache: dict[str, Any] = {}
_lock = threading.Lock()


def bucket(tier: str | None) -> str:
    """The match_samples tier bucket for a ranked tier."""
    if not tier:
        return "ALL"
    tier = tier.upper()
    return "MASTER_PLUS" if tier in APEX else tier


def baselines(conn: psycopg.Connection) -> dict[tuple[str, str], dict[str, tuple[float, ...]]]:
    """(tier bucket or 'ALL', role) -> metric -> (20th, 50th, 80th percentile, games),
    over ranked solo games longer than 15 minutes. Cached for half an hour."""
    with _lock:
        hit = _cache.get("baselines")
        if hit and time.monotonic() - hit[0] < TTL_S:
            return hit[1]
    cols = ", ".join(f"percentile_cont(ARRAY[{LOW}, 0.5, {HIGH}]) WITHIN GROUP (ORDER BY {sql}) {name}"
                     for name, sql in METRICS.items())
    rows = conn.execute(f"""
        SELECT coalesce(s.tier_bucket, 'ALL'), b.position, count(*), {cols}
          FROM participant_builds b JOIN match_samples s USING (match_id)
         WHERE b.queue_id = 420 AND NOT b.remake AND b.duration_s > 900 AND b.position <> ''
         GROUP BY GROUPING SETS ((s.tier_bucket, b.position), (b.position))""").fetchall()
    out: dict[tuple[str, str], dict[str, tuple[float, ...]]] = {}
    for tier, position, n, *quantiles in rows:
        out[(tier, position)] = {name: (*q, n) for name, q in zip(METRICS, quantiles, strict=True)}
    with _lock:
        _cache["baselines"] = (time.monotonic(), out)
    return out


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _baseline(table: dict, tier: str, position: str) -> tuple[str, dict[str, tuple[float, ...]] | None]:
    own = table.get((tier, position))
    if own and own["cs_min"][3] >= MIN_GAMES:
        return tier, own
    return "ALL", table.get(("ALL", position))


@dataclass
class Note:
    kind: str           # "good" | "bad"
    label: str          # short, for the match card
    why: str            # one sentence of evidence, for the tooltip and the Insights tab
    weight: float       # how strongly it stood out, for ordering

    def view(self) -> dict[str, str]:
        return {"kind": self.kind, "label": self.label, "why": self.why}

    @property
    def pattern(self) -> str | None:
        """Which recurring pattern this note is an instance of (see PATTERNS)."""
        return next((key for key, _, test, _ in PATTERNS if test(self.label)), None)


# Recurring patterns across games: key, kind, how to spot the note, how to say it.
PATTERNS = (
    ("most_damage", "good", lambda s: s == "Most damage in the game", "most damage in the game"),
    ("damage_share", "good", lambda s: s.endswith("% of team damage"), "30%+ of the team's damage"),
    ("kp_high", "good", lambda s: s.startswith("In on "), "in on two thirds of kills"),
    ("kp_low", "bad", lambda s: s.startswith("Missed fights"), "missed most of the team's kills"),
    ("vision_most", "good", lambda s: s == "Most vision in the game", "most vision in the game"),
    ("deathless", "good", lambda s: s == "Didn't die", "no deaths"),
    ("first_blood", "good", lambda s: s == "First blood", "first blood"),
    ("multikill", "good", lambda s: s.endswith(" kill") and s.split()[0] in ("Triple", "Quadra", "Penta"),
     "a triple kill or better"),
    ("cs_up", "good", lambda s: s.startswith("Out-farmed "), "out-farmed the lane opponent by 30+"),
    ("cs_down", "bad", lambda s: " CS behind " in s, "30+ CS behind the lane opponent"),
    ("gold_up", "good", lambda s: " gold on " in s, "ahead of the lane opponent at 15"),
    ("gold_down", "bad", lambda s: " gold behind " in s, "behind the lane opponent at 15"),
    ("farm_high", "good", lambda s: s.startswith("Farm: "), "top-20% farm for the rank"),
    ("farm_low", "bad", lambda s: s.startswith("Low farm: "), "bottom-20% farm for the rank"),
    ("vision_high", "good", lambda s: s.startswith("Vision: "), "top-20% vision for the rank"),
    ("vision_low", "bad", lambda s: s.startswith("Little vision: "), "bottom-20% vision for the rank"),
    ("damage_high", "good", lambda s: s.startswith("Damage: "), "top-20% damage for the rank"),
    ("damage_low", "bad", lambda s: s.startswith("Low damage: "), "bottom-20% damage for the rank"),
    ("deaths_high", "bad", lambda s: s.startswith("Died "), "died far more than the rank does"),
    ("no_wards", "bad", lambda s: s == "No control wards", "no control wards bought"),
)


def patterns(per_game: list[list[Note]], min_games: int = 3) -> dict[str, list[dict]]:
    """The patterns seen in at least ``min_games`` of the games, most frequent first."""
    from collections import Counter

    seen: Counter[str] = Counter()
    for notes in per_game:
        seen.update({n.pattern for n in notes if n.pattern})
    out: dict[str, list[dict]] = {"good": [], "bad": []}
    for key, kind, _, phrase in PATTERNS:
        if seen[key] >= min_games:
            out[kind].append({"key": key, "phrase": phrase, "games": seen[key], "of": len(per_game)})
    for kind in out:
        out[kind].sort(key=lambda p: -p["games"])
    return out


def _k(v: float) -> str:
    return f"{v / 1000:.1f}k"


def game_notes(me: dict, players: list[dict], table: dict, tier: str,
               champion_name, lane: dict | None = None) -> list[Note]:
    """Notes for ``me`` (a participant_builds row) given everyone in the game.
    ``champion_name(id)`` names a champion; ``lane`` holds timeline leads (see lane_leads)."""
    if me["remake"]:
        return []
    notes: list[Note] = []
    minutes = max(1.0, me["duration_s"] / 60)
    position = me["position"] or ""
    team = [p for p in players if p["team_id"] == me["team_id"]]
    opp = next((p for p in players if position and p["team_id"] != me["team_id"]
                and p["position"] == position), None)
    opp_name = champion_name(opp["champion_id"]) if opp else ""

    # -- against the other nine --------------------------------------------------------------
    team_damage = sum(p["damage"] for p in team) or 1
    share = me["damage"] / team_damage
    if me["damage"] >= max(p["damage"] for p in players):
        notes.append(Note("good", "Most damage in the game",
                          f"{me['damage']:,} damage to champions, more than any of the other nine "
                          f"players ({round(100 * share)}% of your team's).", 3.0))
    elif share >= 0.30:
        notes.append(Note("good", f"{round(100 * share)}% of team damage",
                          f"{me['damage']:,} of your team's {team_damage:,} damage to champions.", 2.2))
    kp = (me["kills"] + me["assists"]) / max(1, me["team_kills"])
    if me["team_kills"] >= 10 and kp >= 0.68:
        notes.append(Note("good", f"In on {round(100 * kp)}% of kills",
                          f"A kill or assist in {me['kills'] + me['assists']} of your team's "
                          f"{me['team_kills']} kills.", 1.6 + 3 * (kp - 0.68)))
    elif me["team_kills"] >= 10 and kp < 0.35 and position != "TOP":
        notes.append(Note("bad", f"Missed fights: {round(100 * kp)}% KP",
                          f"Involved in only {me['kills'] + me['assists']} of your team's "
                          f"{me['team_kills']} kills.", 1.8 + 3 * (0.35 - kp)))
    if me["vision_score"] >= max(p["vision_score"] for p in players) and me["vision_score"] >= minutes:
        notes.append(Note("good", "Most vision in the game",
                          f"Vision score {me['vision_score']}, the highest of all ten players.", 1.7))
    if me["deaths"] == 0 and minutes >= 15:
        notes.append(Note("good", "Didn't die", f"No deaths in {round(minutes)} minutes.", 2.6))
    if me["first_blood"]:
        notes.append(Note("good", "First blood", "Got the game's first kill.", 1.2))
    if me["largest_multikill"] >= 3:
        word = {3: "Triple", 4: "Quadra"}.get(me["largest_multikill"], "Penta")
        notes.append(Note("good", f"{word} kill", f"A {word.lower()} kill.",
                          1.5 + me["largest_multikill"] - 3))

    # -- against the lane opponent --------------------------------------------------------------
    if opp and position != "UTILITY":
        cs_diff = me["cs"] - opp["cs"]
        if cs_diff >= 30:
            notes.append(Note("good", f"Out-farmed {opp_name} by {cs_diff} CS",
                              f"{me['cs']} CS against {opp['cs']} for the enemy {opp_name}.",
                              1.4 + min(cs_diff, 120) / 60))
        elif cs_diff <= -30:
            notes.append(Note("bad", f"{-cs_diff} CS behind {opp_name}",
                              f"{me['cs']} CS against {opp['cs']} for the enemy {opp_name}.",
                              1.4 + min(-cs_diff, 120) / 60))
    if lane and opp:
        for minute in (15, 10):
            lead = lane.get(minute)
            if lead is None:
                continue
            gold = lead["gold"]
            if gold >= 800:
                notes.append(Note("good", f"+{_k(gold)} gold on {opp_name} at {minute}",
                                  f"At {minute} minutes you had {lead['me_gold']:,} gold to "
                                  f"{opp_name}'s {lead['opp_gold']:,}, and {lead['cs']:+d} CS.",
                                  1.5 + min(gold, 3000) / 1500))
            elif gold <= -800:
                notes.append(Note("bad", f"{_k(-gold)} gold behind {opp_name} at {minute}",
                                  f"At {minute} minutes you had {lead['me_gold']:,} gold to "
                                  f"{opp_name}'s {lead['opp_gold']:,}, and {lead['cs']:+d} CS.",
                                  1.5 + min(-gold, 3000) / 1500))
            break                                   # the 15-minute lead if there is one

    # -- against players at this rank, in this role ----------------------------------------------
    used, base = _baseline(table, tier, position)
    if base and minutes >= 15:
        who = f"{TIER_NAME.get(used, used.title())} {ROLE_NAME.get(position, 'players')}"
        values = {"cs_min": me["cs"] / minutes, "vision_min": me["vision_score"] / minutes,
                  "dmg_min": me["damage"] / minutes, "deaths_10": me["deaths"] / (minutes / 10),
                  "control_wards": float(me["control_wards"])}
        checks = (
            ("cs_min", "Farm", "Low farm", "{:.1f} CS/min", position != "UTILITY", 1.0),
            ("vision_min", "Vision", "Little vision", "{:.1f} vision/min", True, 0.9),
            ("dmg_min", "Damage", "Low damage", "{:,.0f} damage/min", position != "UTILITY", 0.9),
        )
        for key, good_word, bad_word, fmt, applies, scale in checks:
            if not applies:
                continue
            lo, mid, hi, _ = base[key]
            v = values[key]
            shown, typical = fmt.format(v), fmt.format(mid)
            if v >= hi > mid:
                notes.append(Note("good", f"{good_word}: {shown}",
                                  f"Top 20% of {who} (typical {typical}).",
                                  scale * (1.4 + min((v - hi) / (hi - mid), 2))))
            elif v <= lo < mid:
                notes.append(Note("bad", f"{bad_word}: {shown}",
                                  f"Bottom 20% of {who} (typical {typical}).",
                                  scale * (1.4 + min((lo - v) / (mid - lo), 2))))
        lo, mid, hi, _ = base["deaths_10"]
        if me["deaths"] >= 6 and values["deaths_10"] >= hi > mid:
            notes.append(Note("bad", f"Died {me['deaths']} times",
                              f"{values['deaths_10']:.1f} deaths per 10 minutes; typical {who} "
                              f"die {mid:.1f}.", 1.6 + min((values['deaths_10'] - hi) / (hi - mid), 2)))
        typical_wards = base["control_wards"][1]
        if me["control_wards"] == 0 and minutes >= 20 and typical_wards >= 1:
            notes.append(Note("bad", "No control wards",
                              f"None bought in {round(minutes)} minutes; {who} typically buy "
                              f"{typical_wards:.0f}.", 1.5 + min(typical_wards, 4) / 4))
    elif me["deaths"] >= 10:
        notes.append(Note("bad", f"Died {me['deaths']} times", "Ten or more deaths.", 1.8))

    # The strongest few of each kind; a label never repeats.
    seen: set[str] = set()
    out = []
    for n in sorted(notes, key=lambda n: -n.weight):
        if n.label not in seen:
            seen.add(n.label)
            out.append(n)
    return out


def card(notes: list[Note], each: int = 2) -> list[dict[str, str]]:
    """Up to ``each`` strengths then ``each`` weaknesses, for a match card."""
    goods = [n.view() for n in notes if n.kind == "good"][:each]
    bads = [n.view() for n in notes if n.kind == "bad"][:each]
    return goods + bads


def lane_leads(timeline: dict | None, me_pid: int, opp_pid: int | None) -> dict[int, dict]:
    """Gold and CS leads over the lane opponent at minutes 10 and 15, from the timeline."""
    if not timeline or opp_pid is None:
        return {}
    frames = timeline.get("info", {}).get("frames", [])
    out = {}
    for minute in (10, 15):
        if minute >= len(frames):
            continue
        pf = frames[minute].get("participantFrames") or {}
        a, b = pf.get(str(me_pid)), pf.get(str(opp_pid))
        if not a or not b:
            continue
        cs_a = a.get("minionsKilled", 0) + a.get("jungleMinionsKilled", 0)
        cs_b = b.get("minionsKilled", 0) + b.get("jungleMinionsKilled", 0)
        out[minute] = {"gold": a.get("totalGold", 0) - b.get("totalGold", 0),
                       "me_gold": a.get("totalGold", 0), "opp_gold": b.get("totalGold", 0),
                       "cs": cs_a - cs_b, "xp": a.get("xp", 0) - b.get("xp", 0)}
    return out
