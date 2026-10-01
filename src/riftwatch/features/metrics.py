"""The metric catalogue: what each feature means, which way is good, and how to show it.

Baselines, scoring, evidence and reports all read from here, so a metric is defined once.
"""

from __future__ import annotations

from dataclasses import dataclass

AREAS = ("farming", "laning", "fighting", "survival", "vision", "objectives", "economy")


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    area: str
    higher_is_better: bool = True
    kind: str = "game"            # "game" = one value per game, "curve" = one per minute
    fmt: str = "{:.1f}"
    skip_roles: tuple[str, ...] = ()   # roles where this metric isn't a fair yardstick
    coachable: bool = True        # False = context only, never a finding on its own

    def show(self, value: float) -> str:
        return self.fmt.format(value)

    def applies_to(self, role: str) -> bool:
        return role not in self.skip_roles


PCT = "{:.0%}"
INT = "{:.0f}"
SIGNED = "{:+.0f}"
NOT_SUPPORT = ("UTILITY",)

_GAME = [
    Metric("cs_per_min", "CS per minute", "farming", skip_roles=NOT_SUPPORT),
    Metric("cs_at_10", "CS at 10 min", "farming", fmt=INT, skip_roles=NOT_SUPPORT),
    Metric("cs_at_15", "CS at 15 min", "farming", fmt=INT, skip_roles=NOT_SUPPORT),
    Metric("cs_at_20", "CS at 20 min", "farming", fmt=INT, skip_roles=NOT_SUPPORT),
    Metric("cs_diff_at_10", "CS lead over lane opponent at 10 min", "laning", fmt=SIGNED,
           skip_roles=NOT_SUPPORT),
    Metric("cs_diff_at_15", "CS lead over lane opponent at 15 min", "laning", fmt=SIGNED,
           skip_roles=NOT_SUPPORT),
    Metric("gold_diff_at_10", "gold lead over lane opponent at 10 min", "laning", fmt=SIGNED),
    Metric("gold_diff_at_15", "gold lead over lane opponent at 15 min", "laning", fmt=SIGNED),
    Metric("xp_diff_at_10", "XP lead over lane opponent at 10 min", "laning", fmt=SIGNED),
    Metric("xp_diff_at_15", "XP lead over lane opponent at 15 min", "laning", fmt=SIGNED),
    Metric("gold_diff_at_20", "gold lead over lane opponent at 20 min", "laning", fmt=SIGNED),
    Metric("xp_diff_at_20", "XP lead over lane opponent at 20 min", "laning", fmt=SIGNED,
           coachable=False),
    Metric("cs_diff_at_20", "CS lead over lane opponent at 20 min", "laning", fmt=SIGNED,
           skip_roles=NOT_SUPPORT),
    Metric("turret_plates", "turret plates taken", "laning", fmt=INT, skip_roles=("JUNGLE", "UTILITY")),
    Metric("solo_kills", "solo kills", "laning", fmt=INT),
    Metric("solo_deaths", "solo deaths", "survival", higher_is_better=False, fmt=INT),
    Metric("kill_participation", "kill participation", "fighting", fmt=PCT),
    Metric("damage_share", "share of team damage to champions", "fighting", fmt=PCT,
           skip_roles=NOT_SUPPORT),
    Metric("damage_per_min", "damage to champions per minute", "fighting", fmt=INT),
    Metric("kda", "KDA", "fighting", fmt="{:.2f}", coachable=False),
    Metric("kills", "kills", "fighting", fmt=INT, coachable=False),
    Metric("assists", "assists", "fighting", fmt=INT, coachable=False),
    Metric("deaths", "deaths", "survival", higher_is_better=False, fmt=INT),
    Metric("early_deaths", "deaths before 14 min", "survival", higher_is_better=False, fmt=INT),
    Metric("deaths_while_ahead", "deaths while 500+ gold ahead of lane opponent", "survival",
           higher_is_better=False, fmt=INT),
    Metric("first_death_min", "minute of first death", "survival", fmt="{:.1f}", coachable=False),
    Metric("vision_per_min", "vision score per minute", "vision", fmt="{:.2f}"),
    Metric("wards_placed_per_min", "wards placed per minute", "vision", fmt="{:.2f}"),
    Metric("control_wards", "control wards placed", "vision", fmt=INT),
    Metric("wards_killed", "enemy wards cleared", "vision", fmt=INT),
    Metric("objective_participation", "share of team's epic monsters you took part in",
           "objectives", fmt=PCT),
    Metric("tower_participation", "share of team's towers you took part in", "objectives", fmt=PCT),
    Metric("gold_per_min", "gold per minute", "economy", fmt=INT),
    Metric("gold_at_10", "gold at 10 min", "economy", fmt=INT, coachable=False),
    Metric("gold_at_15", "gold at 15 min", "economy", fmt=INT, coachable=False),
    Metric("gold_at_20", "gold at 20 min", "economy", fmt=INT, coachable=False),
    Metric("xp_at_10", "XP at 10 min", "economy", fmt=INT, coachable=False),
    Metric("xp_at_15", "XP at 15 min", "economy", fmt=INT, coachable=False),
    Metric("xp_at_20", "XP at 20 min", "economy", fmt=INT, coachable=False),
]

# Per-minute curves; names match participant_minute_features columns.
_CURVES = [
    Metric("cs", "CS", "farming", kind="curve", fmt=INT, skip_roles=NOT_SUPPORT),
    Metric("gold", "total gold", "economy", kind="curve", fmt=INT),
    Metric("xp", "XP", "economy", kind="curve", fmt=INT, coachable=False),
    Metric("gold_diff", "gold lead over lane opponent", "laning", kind="curve", fmt=SIGNED),
    Metric("xp_diff", "XP lead over lane opponent", "laning", kind="curve", fmt=SIGNED,
           coachable=False),
    Metric("cs_diff", "CS lead over lane opponent", "laning", kind="curve", fmt=SIGNED,
           skip_roles=NOT_SUPPORT),
    Metric("damage_to_champions", "damage to champions", "fighting", kind="curve", fmt=INT,
           coachable=False),
    Metric("wards_placed", "wards placed", "vision", kind="curve", fmt=INT),
]

GAME_METRICS: dict[str, Metric] = {m.name: m for m in _GAME}
CURVE_METRICS: dict[str, Metric] = {m.name: m for m in _CURVES}
ALL_METRICS: dict[str, Metric] = {**GAME_METRICS, **CURVE_METRICS}

# Minutes for which curve baselines are built: enough to cover nearly every ranked game's
# laning phase and mid game without thinning the late-game samples to nothing.
CURVE_MINUTES = range(1, 31)
