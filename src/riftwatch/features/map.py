"""Summoner's Rift geometry: which part of the map a position is in.

Coordinates run from (0, 0) at the blue fountain corner to roughly (14870, 14980) at the
red one. Mid lane runs along the main diagonal; the river crosses it along the other
diagonal. The zones are approximations from that geometry -- good enough to tell "died in
the river" from "died under your own tower", not to trace exact walls.
"""

from __future__ import annotations

MAP_X = 14870
MAP_Y = 14980

ZONES = (
    "blue_base", "red_base", "top_lane", "bot_lane", "mid_lane", "river",
    "blue_jungle", "red_jungle",
)


def zone(x: float, y: float) -> str:
    u, v = x / MAP_X, y / MAP_Y
    if u < 0.26 and v < 0.26:
        return "blue_base"
    if u > 0.74 and v > 0.74:
        return "red_base"
    # Side lanes hug the map edges: top along the left and top edges, bot along the
    # bottom and right edges.
    if u < 0.15 or v > 0.85:
        return "top_lane"
    if v < 0.15 or u > 0.85:
        return "bot_lane"
    if abs(u - v) < 0.07:
        return "mid_lane"
    if abs(u + v - 1) < 0.07:
        return "river"
    return "blue_jungle" if u + v < 1 else "red_jungle"


def own_side(zone_name: str, team_id: int) -> bool | None:
    """Whether a base/jungle zone is on this team's half. None for lanes and river."""
    blue = team_id == 100
    if zone_name in ("blue_base", "blue_jungle"):
        return blue
    if zone_name in ("red_base", "red_jungle"):
        return not blue
    return None


def readable(zone_name: str, team_id: int) -> str:
    """``"red_jungle"`` for a blue player -> ``"enemy jungle"``."""
    side = own_side(zone_name, team_id)
    if side is None:
        return zone_name.replace("_", " ")
    kind = "base" if zone_name.endswith("base") else "jungle"
    return f"{'own' if side else 'enemy'} {kind}"
