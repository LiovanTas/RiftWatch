import pytest

from riftwatch.config import DEFAULT_DATABASE_URL, ConfigError, Settings
from riftwatch.riot.routing import UnknownRegion


def test_defaults_from_empty_env():
    s = Settings.from_env({})
    assert s.riot_api_key is None
    assert s.database_url == DEFAULT_DATABASE_URL
    assert s.default_platform == "na1"


def test_blank_values_count_as_unset():
    # .env.example ships with `RIOT_API_KEY=` -- that must not read as a key.
    s = Settings.from_env({"RIOT_API_KEY": "", "RIFTWATCH_DEFAULT_REGION": ""})
    assert s.riot_api_key is None
    assert s.default_platform == "na1"


def test_region_is_normalised():
    assert Settings.from_env({"RIFTWATCH_DEFAULT_REGION": "EUW"}).default_platform == "euw1"


def test_bad_region_fails_fast():
    with pytest.raises(UnknownRegion):
        Settings.from_env({"RIFTWATCH_DEFAULT_REGION": "moon"})


def test_require_riot_key():
    with pytest.raises(ConfigError, match="24h"):
        Settings.from_env({}).require_riot_key()
    assert Settings.from_env({"RIOT_API_KEY": "RGAPI-x"}).require_riot_key() == "RGAPI-x"


def test_website_guard_settings():
    s = Settings.from_env({"RIFTWATCH_RATE_LIMITS": "coach=3", "RIFTWATCH_COACH_DAILY_BUDGET_USD": "2.5"})
    assert s.rate_limits == "coach=3" and s.coach_daily_budget_usd == 2.5
    assert Settings.from_env({}).coach_daily_budget_usd == 5.0
