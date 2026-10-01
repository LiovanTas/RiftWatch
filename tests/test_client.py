import httpx
import pytest

from riftwatch.riot.client import RiotApiError, RiotAuthError, RiotClient
from riftwatch.riot.ratelimit import RateLimiter

KEY = "RGAPI-00000000-test-key"


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class SpyLimiter(RateLimiter):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__("1000:1", pad_s=0.0, clock=clock, sleep=clock.sleep)
        self.blocks: list[tuple[str, float, str | None]] = []
        self.header_updates = 0

    def block(self, routing, seconds, method=None):
        self.blocks.append((routing, seconds, method))
        super().block(routing, seconds, method)

    def update_from_headers(self, routing, method, headers):
        self.header_updates += 1
        super().update_from_headers(routing, method, headers)


def make(responses, rng=lambda: 1.0):
    """Client whose transport replays `responses` in order. Each item is an
    httpx.Response, or an exception class to raise instead."""
    clock = FakeClock()
    limiter = SpyLimiter(clock)
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, type) and issubclass(item, Exception):
            raise item("boom", request=request)
        # Fresh copy: the same scripted response may be replayed several times.
        return httpx.Response(item.status_code, headers=item.headers, content=item.content)

    client = RiotClient(KEY, limiter, transport=httpx.MockTransport(handler), rng=rng)
    return client, limiter, clock, seen


def ok(body=None, **headers):
    return httpx.Response(200, json=body if body is not None else {"ok": True}, headers=headers)


def status(code, **headers):
    return httpx.Response(code, json={"status": {"status_code": code}}, headers=headers)


def h(**kw):
    """Python kwargs can't contain dashes: x_rate_limit_type -> X-Rate-Limit-Type."""
    return {k.replace("_", "-"): v for k, v in kw.items()}


# -- happy path ---------------------------------------------------------------------------

def test_200_returns_json_and_sends_key_and_params():
    client, limiter, clock, seen = make([ok({"puuid": "abc"})])
    body = client.get("americas", "account-v1.by-riot-id", "/riot/account/v1/x", {"count": 5})
    assert body == {"puuid": "abc"}
    req = seen[0]
    assert str(req.url) == "https://americas.api.riotgames.com/riot/account/v1/x?count=5"
    assert req.headers["X-Riot-Token"] == KEY
    assert client.stats["requests"] == 1 and client.stats["retries"] == 0


def test_404_is_none_not_an_error():
    client, *_ = make([status(404)])
    assert client.get("na1", "league-v4.entries", "/x") is None


def test_unknown_routing_rejected_before_any_request():
    client, _, _, seen = make([])
    with pytest.raises(ValueError):
        client.get("evil.example.com/", "m", "/x")
    assert seen == []


def test_response_headers_feed_the_limiter():
    client, limiter, clock, _ = make([ok(**h(X_App_Rate_Limit="1:10"))])
    client.get("americas", "m", "/x")
    assert limiter.header_updates == 1
    assert limiter.acquire("americas", "m") == pytest.approx(10)  # learned 1:10


# -- 429 ----------------------------------------------------------------------------------

def test_application_429_blocks_the_whole_host_for_retry_after():
    client, limiter, clock, _ = make(
        [status(429, **h(Retry_After="7", X_Rate_Limit_Type="application")), ok()]
    )
    assert client.get("americas", "match-v5.match", "/x") == {"ok": True}
    assert limiter.blocks == [("americas", 7.0, None)]
    assert clock() == pytest.approx(7)
    assert client.stats["429:application"] == 1 and client.stats["retries"] == 1


def test_method_429_blocks_only_that_method():
    client, limiter, clock, _ = make(
        [status(429, **h(Retry_After="3", X_Rate_Limit_Type="method")), ok()]
    )
    client.get("americas", "match-v5.timeline", "/x")
    assert limiter.blocks == [("americas", 3.0, "match-v5.timeline")]


def test_service_429_without_retry_after_backs_off_exponentially():
    client, limiter, clock, _ = make(
        [status(429, **h(X_Rate_Limit_Type="service"))] * 3 + [ok()]
    )
    client.get("americas", "m", "/x")
    # rng pinned to 1.0, so full-jitter backoff hits its ceiling: 1, 2, 4 s.
    assert [b[1] for b in limiter.blocks] == [1.0, 2.0, 4.0]
    assert clock() == pytest.approx(7)
    assert client.stats["429:service"] == 3


def test_429_without_type_header_is_treated_as_service():
    client, limiter, *_ = make([status(429), ok()])
    client.get("na1", "m", "/x")
    assert client.stats["429:service"] == 1
    assert limiter.blocks[0][2] == "m"


# -- 5xx and network ----------------------------------------------------------------------

def test_5xx_retried_with_jitter_then_succeeds():
    client, limiter, clock, _ = make([status(503), status(500), ok()], rng=lambda: 0.5)
    assert client.get("europe", "m", "/x") == {"ok": True}
    assert [b[1] for b in limiter.blocks] == [0.5, 1.0]
    assert client.stats["5xx"] == 2


def test_backoff_is_capped():
    client, limiter, *_ = make([status(503)] * 8 + [ok()])
    client._max_retries = 8
    client.get("europe", "m", "/x")
    assert max(b[1] for b in limiter.blocks) == 30.0


def test_gives_up_after_max_retries():
    client, _, _, seen = make([status(502)] * 5)
    with pytest.raises(RiotApiError, match="gave up after 5 attempts; last: m /x: 502"):
        client.get("europe", "m", "/x")
    assert len(seen) == 5


def test_network_error_is_retried():
    client, *_ = make([httpx.ConnectError, httpx.ReadTimeout, ok()])
    assert client.get("asia", "m", "/x") == {"ok": True}
    assert client.stats["network_errors"] == 2


# -- non-retryable ------------------------------------------------------------------------

@pytest.mark.parametrize("code", [401, 403])
def test_auth_errors_fail_fast_with_a_hint(code):
    client, _, _, seen = make([status(code)])
    with pytest.raises(RiotAuthError, match="expire every 24h") as exc:
        client.get("na1", "m", "/x")
    assert exc.value.status == code
    assert len(seen) == 1


def test_400_fails_fast():
    client, _, _, seen = make([status(400)])
    with pytest.raises(RiotApiError, match="unexpected status 400"):
        client.get("na1", "m", "/x")
    assert len(seen) == 1


def test_key_never_appears_in_errors_or_repr():
    client, *_ = make([status(403)])
    with pytest.raises(RiotAuthError) as exc:
        client.get("na1", "m", "/x")
    assert KEY not in str(exc.value)
    assert KEY not in repr(client)
