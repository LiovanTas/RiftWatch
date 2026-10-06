"""Settings, read from the environment (and a ``.env`` file loaded by the CLI)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from riftwatch.riot.routing import platform_for

DEFAULT_DATABASE_URL = "postgresql://riftwatch:riftwatch@127.0.0.1:5432/riftwatch"
DEFAULT_COACH_MODEL = "claude-sonnet-5-5"
DEFAULT_COACH_EFFORT = "low"
DEFAULT_COACH_THINKING = "adaptive"
DEFAULT_KILL_HOTKEY = "ctrl+alt+k"
DEFAULT_MODELS_DIR = "out/ml/models"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    riot_api_key: str | None
    database_url: str
    default_platform: str
    anthropic_api_key: str | None
    coach_model: str
    coach_effort: str = DEFAULT_COACH_EFFORT
    coach_thinking: str = DEFAULT_COACH_THINKING
    kill_hotkey: str = DEFAULT_KILL_HOTKEY
    models_dir: str = DEFAULT_MODELS_DIR
    rate_limits: str | None = None          # website: "sync=30,scout=60,coach=10" per hour
    coach_daily_budget_usd: float = 5.0     # website: LLM coaching stops for the day past this

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            riot_api_key=env.get("RIOT_API_KEY") or None,
            database_url=env.get("RIFTWATCH_DATABASE_URL") or DEFAULT_DATABASE_URL,
            default_platform=platform_for(env.get("RIFTWATCH_DEFAULT_REGION") or "na"),
            anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
            coach_model=env.get("RIFTWATCH_COACH_MODEL") or DEFAULT_COACH_MODEL,
            coach_effort=env.get("RIFTWATCH_COACH_EFFORT") or DEFAULT_COACH_EFFORT,
            coach_thinking=env.get("RIFTWATCH_COACH_THINKING") or DEFAULT_COACH_THINKING,
            kill_hotkey=env.get("RIFTWATCH_KILL_HOTKEY") or DEFAULT_KILL_HOTKEY,
            models_dir=env.get("RIFTWATCH_MODELS_DIR") or DEFAULT_MODELS_DIR,
            rate_limits=env.get("RIFTWATCH_RATE_LIMITS") or None,
            coach_daily_budget_usd=float(env.get("RIFTWATCH_COACH_DAILY_BUDGET_USD") or 5.0),
        )

    def require_riot_key(self) -> str:
        if not self.riot_api_key:
            raise ConfigError(
                "RIOT_API_KEY is not set. Development keys expire every 24h -- "
                "get a fresh one at https://developer.riotgames.com and put it in .env"
            )
        return self.riot_api_key
