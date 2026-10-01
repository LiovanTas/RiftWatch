import json

import httpx
import pytest

from riftwatch.riot.api import RiotApi
from riftwatch.riot.client import RiotClient
from riftwatch.riot.ddragon import ChampionNames, DataDragon, patch_of
from riftwatch.riot.ratelimit import RateLimiter
from riftwatch.riot.routing import RiotId


def api_with(handler):
    limiter = RateLimiter("1000:1", pad_s=0)
    client = RiotClient("RGAPI-test", limiter, transport=httpx.MockTransport(handler))
    return RiotApi(client)


def recorder(body):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=body)

    return seen, handler


def test_account_by_riot_id_routes_and_encodes():
    seen, handler = recorder({"puuid": "p1", "gameName": "C9 Loki", "tagLine": "kr3"})
    api = api_with(handler)
    assert api.account_by_riot_id("na", RiotId("C9 Loki", "kr3"))["puuid"] == "p1"
    url = seen[0].url
    assert url.host == "americas.api.riotgames.com"
    assert url.raw_path == b"/riot/account/v1/accounts/by-riot-id/C9%20Loki/kr3"


def test_sea_accounts_resolve_through_asia():
    seen, handler = recorder({"puuid": "p"})
    api_with(handler).account_by_riot_id("vn", RiotId("Name", "VN2"))
    assert seen[0].url.host == "asia.api.riotgames.com"


def test_match_ids_params():
    seen, handler = recorder(["NA1_2", "NA1_1"])
    ids = api_with(handler).match_ids("na1", "puuid", start=100, count=100, queue=420,
                                       start_time=1_700_000_000)
    assert ids == ["NA1_2", "NA1_1"]
    params = dict(seen[0].url.params)
    assert params == {"start": "100", "count": "100", "queue": "420", "startTime": "1700000000"}


def test_match_ids_count_bounds():
    api = api_with(lambda r: httpx.Response(200, json=[]))
    with pytest.raises(ValueError):
        api.match_ids("na1", "p", count=101)


def test_match_and_timeline_route_by_match_id_prefix():
    seen, handler = recorder({})
    api = api_with(handler)
    api.match("EUW1_123")
    api.timeline("KR_456")
    assert seen[0].url.host == "europe.api.riotgames.com"
    assert seen[1].url.path == "/lol/match/v5/matches/KR_456/timeline"
    assert seen[1].url.host == "asia.api.riotgames.com"


def test_league_endpoints_use_platform_hosts():
    seen, handler = recorder([])
    api = api_with(handler)
    api.league_entries_by_puuid("euw", "p")
    api.league_entries("kr", "emerald", "ii", page=3)
    assert seen[0].url.host == "euw1.api.riotgames.com"
    assert seen[1].url.path == "/lol/league-exp/v4/entries/RANKED_SOLO_5x5/EMERALD/II"
    assert seen[1].url.params["page"] == "3"


def test_league_entries_validates_apex_divisions():
    api = api_with(lambda r: httpx.Response(200, json=[]))
    with pytest.raises(ValueError):
        api.league_entries("na1", "MASTER", "II")
    with pytest.raises(ValueError):
        api.league_entries("na1", "WOOD", "I")


def test_404s_become_empty_results():
    api = api_with(lambda r: httpx.Response(404))
    assert api.match_ids("na1", "p") == []
    assert api.match("NA1_1") is None
    assert api.league_entries_by_puuid("na1", "p") == []


# -- Data Dragon --------------------------------------------------------------------------

CHAMPS = {"data": {"Jhin": {"key": "202", "name": "Jhin"},
                   "KSante": {"key": "897", "name": "K'Sante"}}}


def ddragon_transport(calls):
    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/versions.json":
            return httpx.Response(200, json=["16.19.1", "16.18.1", "16.17.1"])
        if request.url.path.endswith("champion.json"):
            return httpx.Response(200, text=json.dumps(CHAMPS))
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_patch_of():
    assert patch_of("16.19.712.3456") == "16.19"
    with pytest.raises(ValueError):
        patch_of("garbage")


def test_version_for_patch_and_fallback(tmp_path):
    dd = DataDragon(tmp_path, ddragon_transport([]))
    assert dd.version_for("16.18.555.1") == "16.18.1"
    assert dd.version_for("16.20") == "16.19.1"   # not released on ddragon yet -> newest


def test_champions_cached_on_disk(tmp_path):
    calls = []
    dd = DataDragon(tmp_path, ddragon_transport(calls))
    assert dd.champions("16.19.1") == {202: "Jhin", 897: "K'Sante"}
    dd2 = DataDragon(tmp_path, ddragon_transport(calls))
    dd2.champions("16.19.1")
    assert sum(p.endswith("champion.json") for p in calls) == 1


def test_champion_names_offline_fallback():
    def broken(request):
        raise httpx.ConnectError("offline", request=request)

    names = ChampionNames(DataDragon(transport=httpx.MockTransport(broken)))
    assert names(202) == "Champion 202"


def test_champion_name_lookup_is_forgiving(tmp_path):
    names = ChampionNames(DataDragon(tmp_path, ddragon_transport([])), "16.19.1")
    assert names(897) == "K'Sante"
    assert names.id_for("ksante") == 897
    assert names.id_for("nobody") is None
