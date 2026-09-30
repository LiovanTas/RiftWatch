import pytest

from riftwatch.riot.routing import (
    RiotId,
    UnknownRegion,
    account_region,
    match_region,
    parse_riot_id,
    platform_for,
    platform_host,
    platform_of_match,
)


@pytest.mark.parametrize(
    "typed, platform",
    [("na", "na1"), ("NA1", "na1"), (" EUW ", "euw1"), ("eune", "eun1"), ("kr", "kr"),
     ("oce", "oc1"), ("lan", "la1"), ("las", "la2")],
)
def test_platform_aliases(typed, platform):
    assert platform_for(typed) == platform


def test_unknown_region_lists_choices():
    with pytest.raises(UnknownRegion, match="euw"):
        platform_for("atlantis")


@pytest.mark.parametrize(
    "platform, match, account",
    [("na1", "americas", "americas"), ("euw1", "europe", "europe"), ("kr", "asia", "asia"),
     ("oc1", "sea", "asia"), ("vn2", "sea", "asia")],
)
def test_regional_routing(platform, match, account):
    # SEA has a match-v5 cluster but no account-v1 cluster.
    assert match_region(platform) == match
    assert account_region(platform) == account


def test_hosts():
    assert platform_host("na") == "https://na1.api.riotgames.com"


def test_platform_of_match():
    assert platform_of_match("NA1_5123456789") == "na1"
    assert platform_of_match("EUW1_7000000000") == "euw1"
    with pytest.raises(ValueError):
        platform_of_match("5123456789")


@pytest.mark.parametrize(
    "text, expected",
    [("Zven#S16XD", RiotId("Zven", "S16XD")),
     ("C9 Loki#kr3", RiotId("C9 Loki", "kr3")),
     ("  Faker #KR1 ", RiotId("Faker", "KR1"))],
)
def test_parse_riot_id(text, expected):
    assert parse_riot_id(text) == expected


@pytest.mark.parametrize("text", ["Zven", "#NA1", "Zven#", ""])
def test_parse_riot_id_rejects(text):
    with pytest.raises(ValueError):
        parse_riot_id(text)
