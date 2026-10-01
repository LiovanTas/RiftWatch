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
        )

    def require_riot_key(self) -> str:
        if not self.riot_api_key:
            raise ConfigError(
                "RIOT_API_KEY is not set. Development keys expire every 24h -- "
                "get a fresh one at https://developer.riotgames.com and put it in .env"
            )
        return self.riot_api_key
