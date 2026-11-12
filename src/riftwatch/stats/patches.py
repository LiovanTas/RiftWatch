"""Patch notes on the tier board.

Win rates come from stored games, and most of those were played a few patches ago. A champion
buffed or nerfed since then is placed by numbers from before the change. This module reads
``patch_notes.json`` (Riot's notes for recent patches, plus the next patch's preview) and
nudges each changed champion's tier score:

  * a buff adds ``SHIFT`` win-rate points and a nerf subtracts them -- about what Riot aims a
    typical balance change at -- scaled by the share of the games in scope that were played
    before that patch (a change the data already includes moves nothing);
  * a previewed change counts ``PREVIEW_WEIGHT`` of that, since previews are often adjusted;
  * mixed changes and reworks move nothing but are flagged, because the old numbers may no
    longer describe the champion.

The nudge is shown, never hidden: every changed champion carries its changes, and the board
can be switched to the raw data.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SHIFT = 1.0
PREVIEW_WEIGHT = 0.6
WEIGHT = {"buff": 1.0, "nerf": -1.0, "adjust": 0.0}
NOTES_PATH = Path(__file__).with_name("patch_notes.json")


@lru_cache(maxsize=1)
def notes(path: str = str(NOTES_PATH)) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def game_patch(patch: str) -> tuple[int, int]:
    """'26.22' (a patch name) or '16.22' (a game version) -> (26, 22), for comparing them.
    Patch names run 10 ahead of the game version's major number."""
    major, minor = (int(x) for x in patch.split(".")[:2])
    return (major if major >= 20 else major + 10, minor)


def by_champion(name_to_id: dict[str, int]) -> dict[int, list[dict[str, Any]]]:
    """Champion id -> its changes, newest patch first."""
    out: dict[int, list[dict[str, Any]]] = {}
    for p in notes()["patches"]:
        for c in p["changes"]:
            cid = name_to_id.get(c["champion"].lower())
            if cid is None:
                continue
            out.setdefault(cid, []).append({**c, "patch": p["patch"], "status": p["status"],
                                            "source": p["source"]})
    return out


def share_before(patch: str, counts: list[tuple[str, int]]) -> float:
    """Share of the games in ``counts`` (game version, games) played before ``patch``."""
    total = sum(n for _, n in counts)
    if not total:
        return 0.0
    key = game_patch(patch)
    return sum(n for v, n in counts if game_patch(v) < key) / total


def shift(changes: list[dict[str, Any]], counts: list[tuple[str, int]]) -> float:
    """Tier-score points to add for these changes, given the data's patch mix."""
    total = 0.0
    for c in changes:
        weight = WEIGHT.get(c["kind"], 0.0) * (PREVIEW_WEIGHT if c["status"] == "preview" else 1.0)
        total += SHIFT * weight * share_before(c["patch"], counts)
    return total


def mark(changes: list[dict[str, Any]]) -> str | None:
    """'buff', 'nerf' or 'adjust' for a champion's badge: the net direction of its changes."""
    if not changes:
        return None
    net = sum(WEIGHT.get(c["kind"], 0.0) for c in changes)
    return "buff" if net > 0 else "nerf" if net < 0 else "adjust"
