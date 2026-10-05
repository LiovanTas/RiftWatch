"""Post-game analysis of a live recording: trades, recalls and deaths, second by second.

The recording has the player's own health every second but not the enemy's, so a "trade"
here is the player's side of it: a sharp health loss, and what it led to -- a death, a
forced recall, or nothing (they stayed and recovered).

Recalls aren't in the API either. A purchase made while alive, outside the first 90 seconds
and not just after respawning, means the player was in base, so that's what marks a recall;
health and gold are read from just before it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

BIG_DROP = 0.20          # health lost, as a share of max, that counts as a lost trade
DROP_WINDOW_S = 15.0     # ...within this long
QUIET_S = 3.0            # a drop ends after this long without further loss
RECALL_AFTER_S = 60.0    # a recall this soon after a drop counts as forced by it
DEATH_AFTER_S = 20.0
BURST_S = 3.0            # from 60%+ health to dead this fast is a burst death
EARLY_GAME_S = 14 * 60


@dataclass
class Drop:
    start: float
    end: float
    lost: float              # share of max health
    hp_after: float          # share of max health
    led_to: str              # "death" | "recall" | "stayed"


@dataclass
class Recall:
    t: float
    hp: float                # share of max health just before
    gold: float              # unspent gold just before


@dataclass
class DeathInfo:
    t: float
    hp_10s_before: float
    seconds_from_60pct: float | None
    burst: bool


@dataclass
class LiveSummary:
    duration_s: float
    drops: list[Drop] = field(default_factory=list)
    recalls: list[Recall] = field(default_factory=list)
    deaths: list[DeathInfo] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        early = [d for d in self.drops if d.start < EARLY_GAME_S]
        m: dict[str, float] = {
            "early_big_hp_losses": len(early),
            "early_losses_to_death": sum(d.led_to == "death" for d in early),
            "early_losses_to_recall": sum(d.led_to == "recall" for d in early),
            "burst_deaths": sum(d.burst for d in self.deaths),
            "recalls": len(self.recalls),
        }
        if self.recalls:
            m["avg_recall_hp"] = round(sum(r.hp for r in self.recalls) / len(self.recalls), 3)
            m["avg_recall_gold"] = round(sum(r.gold for r in self.recalls) / len(self.recalls))
        return m

    def to_json(self) -> dict[str, Any]:
        return {"duration_s": self.duration_s, "metrics": self.metrics(),
                "drops": [asdict(d) for d in self.drops],
                "recalls": [asdict(r) for r in self.recalls],
                "deaths": [asdict(d) for d in self.deaths]}


def _mine(sample: dict[str, Any], riot_id: str) -> dict[str, Any]:
    return next((p for p in sample.get("players", []) if p["riot_id"] == riot_id), {})


def _pct(s: dict[str, Any]) -> float:
    return s["hp"] / s["hp_max"] if s.get("hp_max") else 0.0


def analyze(head: dict[str, Any], samples: list[dict[str, Any]]) -> LiveSummary:
    samples = sorted((s for s in samples if s.get("hp_max")), key=lambda s: s["t"])
    if not samples:
        return LiveSummary(0.0)
    riot_id = head.get("riot_id", "")
    summary = LiveSummary(samples[-1]["t"])
    dead = [bool(_mine(s, riot_id).get("dead")) for s in samples]

    # -- deaths ---------------------------------------------------------------------------
    death_times = [samples[i]["t"] for i in range(1, len(samples)) if dead[i] and not dead[i - 1]]
    respawn_times = [samples[i]["t"] for i in range(1, len(samples)) if dead[i - 1] and not dead[i]]
    for t in death_times:
        before = [s for s in samples if t - 10.5 <= s["t"] < t]
        hp_10 = _pct(before[0]) if before else 0.0
        above = [s["t"] for s in before if _pct(s) >= 0.6]
        from_60 = round(t - above[-1], 1) if above else None
        summary.deaths.append(DeathInfo(t, round(hp_10, 3), from_60,
                                        from_60 is not None and from_60 <= BURST_S))

    # -- recalls (a purchase while alive, outside the opening and respawn shopping) --------
    for i in range(1, len(samples)):
        prev, cur = samples[i - 1], samples[i]
        if dead[i] or cur["t"] < 90:
            continue
        if any(0 <= cur["t"] - r <= 20 for r in respawn_times):
            continue
        if _mine(cur, riot_id).get("items") == _mine(prev, riot_id).get("items"):
            continue
        if summary.recalls and cur["t"] - summary.recalls[-1].t < 45:
            continue        # several purchases in one shopping trip
        # Read state from before the 8 s recall channel and the fountain heal.
        window = [s for s in samples if cur["t"] - 15 <= s["t"] <= cur["t"] - 9]
        ref = window[0] if window else prev
        gold = max(s["gold"] for s in samples if cur["t"] - 15 <= s["t"] < cur["t"]) \
            if any(cur["t"] - 15 <= s["t"] < cur["t"] for s in samples) else prev["gold"]
        summary.recalls.append(Recall(cur["t"], round(_pct(ref), 3), round(gold)))

    # -- big health losses and what they led to -----------------------------------------------
    i = 1
    while i < len(samples):
        if dead[i] or _pct(samples[i]) >= _pct(samples[i - 1]):
            i += 1
            continue
        start_i = i - 1
        peak = _pct(samples[start_i])
        last_drop_t = samples[i]["t"]
        j = i
        while j < len(samples) and not dead[j] and samples[j]["t"] - samples[start_i]["t"] <= DROP_WINDOW_S:
            if _pct(samples[j]) < _pct(samples[j - 1]):
                last_drop_t = samples[j]["t"]
            elif samples[j]["t"] - last_drop_t >= QUIET_S:
                break
            j += 1
        end_i = min(j, len(samples) - 1)
        low = min(_pct(s) for s in samples[start_i:end_i + 1])
        lost = peak - low
        if lost >= BIG_DROP:
            start_t = samples[start_i]["t"]
            if any(0 <= t - start_t <= DEATH_AFTER_S for t in death_times):
                led_to = "death"
            elif any(0 <= r.t - start_t <= RECALL_AFTER_S for r in summary.recalls):
                led_to = "recall"
            else:
                led_to = "stayed"
            summary.drops.append(Drop(start_t, samples[end_i]["t"], round(lost, 3),
                                      round(low, 3), led_to))
        i = max(end_i, i) + 1
    return summary
