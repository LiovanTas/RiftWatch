"""What each choice did beyond the game state when it was made.

For every player in every stored game, a choice is compared with the win chance the network
gave their team at the moment it was made:

  * items -- each of the first three finished items, against the chance when it was bought
    (a team that's ahead buys differently from one that's behind, so a raw win rate by item
    mostly measures who was winning)
  * skill -- the first ability maxed, against the chance at 3 minutes
  * keystone and summoner spells -- against the chance at the start (the draft)

The result minus that chance is what the choice added on average ("value over state"), in
percentage points. Averages over few games are noisy, so each is pulled towards zero by how
much real difference there is between choices (empirical Bayes: the spread of true values is
estimated from how much the averages vary beyond what their sample sizes explain). Items are
also measured against the enemy team: mostly magic damage, mostly physical, or two or more
tanks.

These are associations with the game state taken out, not proof that a choice causes a win:
players who pick an unusual item may differ in other ways. A split-half check (the same
numbers from two random halves of the games) says how much of it is signal.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import psycopg

from riftwatch.winprob.swing import version as swing_version

CONTEXTS = {"all": "every game", "ap_heavy": "against mostly magic damage",
            "ad_heavy": "against mostly physical damage", "tanks": "against two or more tanks"}
KINDS = ("item", "skill", "keystone", "spells")
KIND_NAMES = {"item": "Items", "skill": "First ability maxed", "keystone": "Keystone", "spells": "Summoner spells"}
MIN_STORED = 10                 # cells with fewer games aren't stored
MIN_SHOWN = 30                  # nor shown as evidence
SKILL_MINUTE = 3
CHOICES_FILE = "choices.json"


def first_max(order: str | None) -> str | None:
    """The basic ability that reached five points first."""
    counts = {"Q": 0, "W": 0, "E": 0}
    for k in order or "":
        if k in counts:
            counts[k] += 1
            if counts[k] == 5:
                return k
    return None


def _half(match_id: str) -> int:
    return hashlib.md5(match_id.encode()).digest()[0] & 1


def collect(conn: psycopg.Connection, model) -> list[tuple]:
    """(champion, kind, slot, choice, contexts, expected chance, won, match id) for every
    choice in every stored game with Swing."""
    catalog = getattr(model, "catalog", None)
    ver = swing_version(model)
    curves = {m: c for m, c in conn.execute(
        "SELECT match_id, data->'curve' FROM match_swing WHERE version = %s", (ver,))}
    teams: dict[tuple[str, int], list[int]] = defaultdict(list)
    rows = conn.execute("""SELECT match_id, team_id, champion_id, win, purchases, purchase_s, skill_order,
                                  keystone, spells FROM participant_builds
                            WHERE NOT remake AND queue_id IN (420, 440) AND match_id = ANY(%s)""",
                        (list(curves),)).fetchall()
    for r in rows:
        teams[(r[0], r[1])].append(r[2])

    def context(match_id: str, team: int) -> list[str]:
        out = ["all"]
        if catalog is None:
            return out
        enemy = catalog.comp(teams.get((match_id, 300 - team), []))
        if enemy["magic_share"] >= 0.5:
            out.append("ap_heavy")
        elif enemy["magic_share"] <= 0.3:
            out.append("ad_heavy")
        if enemy["Tank"] >= 2:
            out.append("tanks")
        return out

    out = []
    for match_id, team, champ, win, items, seconds, order, keystone, spells in rows:
        curve = curves.get(match_id) or []
        if not curve:
            continue

        def chance(minute: int) -> float:
            v = curve[min(max(minute, 0), len(curve) - 1)] / 100
            return v if team == 100 else 1 - v
        ctx = context(match_id, team)
        won = 1.0 if win else 0.0
        seen, slot = set(), 0
        for item, second in zip(items or [], seconds or [], strict=False):
            if catalog is not None and item in catalog.items and item not in seen:
                seen.add(item)
                slot += 1
                out.append((champ, "item", slot, str(item), ctx, chance(int(second // 60)), won, match_id))
                if slot == 3:
                    break
        skill = first_max(order)
        if skill:
            out.append((champ, "skill", 0, skill, ["all"], chance(SKILL_MINUTE), won, match_id))
        if keystone:
            out.append((champ, "keystone", 0, str(keystone), ["all"], chance(0), won, match_id))
        if spells:
            out.append((champ, "spells", 0, "+".join(str(s) for s in sorted(spells)), ["all"], chance(0), won, match_id))
    return out


def _cells(rows, half: int | None = None) -> dict[tuple, list[tuple[float, float]]]:
    cells: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for champ, kind, slot, choice, contexts, p, won, match_id in rows:
        if half is not None and _half(match_id) != half:
            continue
        for ctx in contexts:
            cells[(champ, kind, slot, choice, ctx)].append((p, won))
    return cells


def _signal(cells: dict[tuple, list], kind: str) -> tuple[float, float]:
    """(spread of true values, typical noise) for one kind of choice, by method of moments."""
    means, noise = [], []
    for key, obs in cells.items():
        if key[1] != kind or key[4] != "all" or len(obs) < 20:
            continue
        means.append(fmean(w - p for p, w in obs))
        noise.append(fmean(p * (1 - p) for p, _ in obs) / len(obs))
    if len(means) < 5:
        return 0.0001, 0.2
    grand = fmean(means)
    var = fmean((m - grand) ** 2 for m in means)
    return max(var - fmean(noise), 0.004 ** 2), fmean(p for p in noise)


def build(conn: psycopg.Connection, model, models_dir: Path | str | None = None, log=print) -> dict:
    """Work out every choice's value and store them (choice_values), with a summary of how
    much signal each kind of choice carries."""
    rows = collect(conn, model)
    log(f"  {len(rows):,} choices from stored games")
    cells = _cells(rows)
    halves = (_cells(rows, 0), _cells(rows, 1))
    ver = swing_version(model)
    summary, records = {"version": ver, "choices": len(rows), "kinds": {}}, []
    for kind in KINDS:
        tau2, _ = _signal(cells, kind)
        # Split-half check: the same cell's raw average in each half of the games.
        a, b = [], []
        for key, obs in halves[0].items():
            other = halves[1].get(key)
            if key[1] == kind and key[4] == "all" and len(obs) >= MIN_SHOWN and other and len(other) >= MIN_SHOWN:
                a.append(fmean(w - p for p, w in obs))
                b.append(fmean(w - p for p, w in other))
        reliability = _corr(a, b) if len(a) >= 10 else None
        n_cells = 0
        for key, obs in cells.items():
            if key[1] != kind or len(obs) < MIN_STORED:
                continue
            n = len(obs)
            raw = fmean(w - p for p, w in obs)
            noise = fmean(p * (1 - p) for p, _ in obs) / n
            weight = tau2 / (tau2 + noise)
            records.append((key[0], kind, key[2], key[3], key[4], n, 100 * raw, 100 * raw * weight,
                            100 * math.sqrt(tau2 * noise / (tau2 + noise)), 100 * fmean(p for p, _ in obs),
                            100 * fmean(w for _, w in obs), ver))
            n_cells += 1
        summary["kinds"][kind] = {"name": KIND_NAMES[kind], "signal_sd": round(100 * math.sqrt(tau2), 2),
                                  "reliability": round(reliability, 2) if reliability is not None else None,
                                  "pairs": len(a), "cells": n_cells}
        log(f"  {kind:<9} true spread ±{100 * math.sqrt(tau2):.1f} pts, split-half reliability "
            f"{'n/a' if reliability is None else f'{reliability:.2f}'} ({len(a)} choices in both halves)")
    with conn.transaction():
        conn.execute("DELETE FROM choice_values")
        with conn.cursor() as cur:
            cur.executemany("""INSERT INTO choice_values (champion_id, kind, slot, choice, context, games, raw, value,
                                                          sd, expected, win_rate, version)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", records)
    summary["stored"] = len(records)
    if models_dir is not None:
        out = Path(models_dir) / "winprob" / CHOICES_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _corr(a: list[float], b: list[float]) -> float:
    ma, mb = fmean(a), fmean(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / (sa * sb) if sa and sb else 0.0


def summary(models_dir: Path | str) -> dict | None:
    path = Path(models_dir) / "winprob" / CHOICES_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# -- reading them back ---------------------------------------------------------------------

def _row(r) -> dict:
    return {"choice": r[0], "games": r[1], "value": round(r[2], 1), "sd": round(r[3], 1),
            "expected": round(r[4]), "win_rate": round(r[5]), "raw": round(r[6], 1)}


def options(conn: psycopg.Connection, model, champion_id: int, kind: str, slot: int = 0,
            context: str = "all", min_games: int = MIN_SHOWN, limit: int = 6) -> list[dict]:
    """The common choices for a champion at one decision, best value first."""
    rows = conn.execute("""SELECT choice, games, value, sd, expected, win_rate, raw FROM choice_values
                            WHERE champion_id = %s AND kind = %s AND slot = %s AND context = %s
                              AND version = %s AND games >= %s
                            ORDER BY value DESC LIMIT %s""",
                        (champion_id, kind, slot, context, swing_version(model), min_games, limit)).fetchall()
    return [_row(r) for r in rows]


def one(conn: psycopg.Connection, model, champion_id: int, kind: str, slot: int, choice: str,
        context: str = "all") -> dict | None:
    r = conn.execute("""SELECT choice, games, value, sd, expected, win_rate, raw FROM choice_values
                         WHERE champion_id = %s AND kind = %s AND slot = %s AND choice = %s AND context = %s
                           AND version = %s""",
                     (champion_id, kind, slot, choice, context, swing_version(model))).fetchone()
    return _row(r) if r else None


def enemy_context(catalog, enemy_champions: list[int]) -> str:
    """The most specific context the enemy team falls in (tanks first, then damage type)."""
    if catalog is None:
        return "all"
    comp = catalog.comp(enemy_champions)
    if comp["Tank"] >= 2:
        return "tanks"
    if comp["magic_share"] >= 0.5:
        return "ap_heavy"
    if comp["magic_share"] <= 0.3:
        return "ad_heavy"
    return "all"
