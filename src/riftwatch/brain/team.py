"""Team awareness from the replay's side columns (vision.hud.sides, kept in the archive).

The side columns show all ten champions every frame, on screen or not, in role order (top,
jungle, mid, bot, support) for each team. Which one is the followed champion isn't labelled,
so it is found: the slot whose health bar tracks the followed champion's HUD panel health
most closely. Its lane opponent is the same role on the other team.

From there each moment gets the opponent's health (known even when the camera has moved
off them), whether their ultimate and summoner spells are up (icons dark on cooldown, judged
per video against each icon's own bright level, as with the panel), both junglers' health,
how many champions on each side are dead (their bars blank), and the two teams' health
difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg

READY_SHARE = 0.6
MIN_MATCHED = 50          # frames where both healths are read, to trust a slot match
MAX_SLOT_ERROR = 0.08     # mean health difference above this: no slot is the followed one
NEAREST_S = 0.5           # a moment takes the side reading within this much game time

FEATURES = ("opp_hp_side", "opp_r_ready", "opp_summoners_ready", "my_jungler_hp",
            "enemy_jungler_hp", "allies_down", "enemies_down", "team_hp_diff")


def frames(conn: psycopg.Connection, video_id: int, offset: float | None) -> pd.DataFrame | None:
    """Per archived frame: game time, the panel's health, and every side slot's readings."""
    from riftwatch.vision import archive

    rows = archive.load(conn, video_id)
    if not rows or offset is None:
        return None
    t, panel, sides = [], [], []
    for r in rows:
        extra = r[4]
        if extra is None or not extra.sides:
            continue
        t.append(r[0] + offset)
        panel.append(np.nan if extra.panel is None or extra.panel.hp is None else extra.panel.hp)
        sides.append([[np.nan if v is None else v for v in champ] for champ in extra.sides])
    if not t:
        return None
    s = np.array(sides, float)                    # (frames, 10, 5): hp, res, r, d, f
    out = {"t": np.array(t), "panel_hp": np.array(panel, float)}
    for k in range(s.shape[1]):
        for j, name in enumerate(("hp", "res", "r", "d", "f")):
            out[f"s{k}_{name}"] = s[:, k, j]
    return pd.DataFrame(out)


def followed_slot(f: pd.DataFrame) -> int | None:
    """The side slot whose health tracks the followed champion's panel health."""
    best, best_err = None, np.inf
    for k in range(10):
        both = f["panel_hp"].notna() & f[f"s{k}_hp"].notna()
        if both.sum() < MIN_MATCHED:
            continue
        err = float((f.loc[both, "panel_hp"] - f.loc[both, f"s{k}_hp"]).abs().mean())
        if err < best_err:
            best, best_err = k, err
    return best if best_err <= MAX_SLOT_ERROR else None


def _ready(values: pd.Series) -> np.ndarray:
    v = values.to_numpy(float)
    seen = v[~np.isnan(v)]
    if len(seen) < 20:
        return np.full(len(v), np.nan)
    bright = np.percentile(seen, 90)
    return np.where(np.isnan(v), np.nan, (v >= READY_SHARE * bright).astype(float))


def _mean(columns: list[np.ndarray]) -> np.ndarray:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all five blank: NaN
        return np.nanmean(columns, axis=0)


def features(f: pd.DataFrame | None, t: np.ndarray) -> pd.DataFrame:
    """The team features at game times ``t`` (all NaN when the video has no side columns)."""
    out = pd.DataFrame({name: np.full(len(t), np.nan) for name in FEATURES + ("self_hp_side",)})
    if f is None or not len(t):
        return out
    me = followed_slot(f)
    if me is None:
        return out
    team = range(0, 5) if me < 5 else range(5, 10)
    enemy = range(5, 10) if me < 5 else range(0, 5)
    opp = (me + 5) % 10
    hp = {k: f[f"s{k}_hp"].to_numpy(float) for k in range(10)}
    per_frame = pd.DataFrame({
        "t": f["t"].to_numpy(float),
        "opp_hp_side": hp[opp],
        "opp_r_ready": _ready(f[f"s{opp}_r"]),
        "opp_summoners_ready": _ready(f[f"s{opp}_d"]) + _ready(f[f"s{opp}_f"]),
        "my_jungler_hp": hp[team[1]],
        "enemy_jungler_hp": hp[enemy[1]],
        # A blank bar is a dead champion (or a misread; the panel's own rule needs seconds of
        # blank, but one frame either way matters little in a count).
        "allies_down": np.sum([np.isnan(hp[k]) for k in team if k != me], axis=0).astype(float),
        "enemies_down": np.sum([np.isnan(hp[k]) for k in enemy], axis=0).astype(float),
        "team_hp_diff": _mean([hp[k] for k in team]) - _mean([hp[k] for k in enemy]),
        # Not a model input: the followed champion's own side bar, a second witness for deaths.
        "self_hp_side": hp[me],
    }).sort_values("t")
    target = pd.DataFrame({"t": np.asarray(t, float), "_i": np.arange(len(t))}).sort_values("t")
    merged = pd.merge_asof(target, per_frame, on="t", direction="nearest", tolerance=NEAREST_S)
    merged = merged.sort_values("_i")
    for name in FEATURES + ("self_hp_side",):
        out[name] = merged[name].to_numpy(float)
    return out
