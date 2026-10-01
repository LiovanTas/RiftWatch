import threading
import time

import pytest

from riftwatch.riot.ratelimit import Bucket, RateLimiter, Window, parse_counts, parse_limits


class FakeClock:
    """Time only moves when someone sleeps, so a test's timeline is exact."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def limiter(app="20:1,100:120", pad_s=0.0):
    clock = FakeClock()
    return RateLimiter(app, pad_s=pad_s, clock=clock, sleep=clock.sleep), clock


def max_in_any_window(times: list[float], seconds: float) -> int:
    """Largest number of timestamps inside any half-open interval [t, t + seconds)."""
    times = sorted(times)
    best, lo = 0, 0
    for hi, t in enumerate(times):
        while times[lo] <= t - seconds:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


# -- header parsing ---------------------------------------------------------------------

def test_parse_limits():
    assert parse_limits("20:1,100:120") == [Window(20, 1), Window(100, 120)]
    assert parse_limits(" 2000:10 ") == [Window(2000, 10)]


@pytest.mark.parametrize("bad", ["", "20", "20:1,", "a:1", "0:1", "5:0"])
def test_parse_limits_rejects(bad):
    with pytest.raises(ValueError):
        parse_limits(bad)


def test_parse_counts():
    assert parse_counts("3:1,57:120") == {1.0: 3, 120.0: 57}


# -- the core guarantee ---------------------------------------------------------------------

def test_never_exceeds_any_window_over_a_long_run():
    rl, clock = limiter("20:1,100:120")
    sent = []
    for _ in range(350):
        rl.acquire("americas", "m")
        sent.append(clock())
    assert max_in_any_window(sent, 1) <= 20
    assert max_in_any_window(sent, 120) <= 100


def test_throughput_is_not_needlessly_low():
    rl, clock = limiter("20:1,100:120")
    sent = []
    for _ in range(101):
        rl.acquire("americas", "m")
        sent.append(clock())
    # 100 requests in five 1 s bursts of 20 ...
    assert sent[99] == pytest.approx(4.0)
    # ... then the 101st waits for the first token to come back, not a whole extra window.
    assert sent[100] == pytest.approx(120.0)


def test_continuous_refill_would_have_429d():
    """Why tokens refill per-token: a 100/120 s bucket refilling continuously at 100/120 per
    second, started full, sends ~199 requests in the first 120 s. Ours sends 100."""
    rl, clock = limiter("100:120")
    count = 0
    while True:
        rl.acquire("americas", "m")
        if clock() >= 120:
            break
        count += 1
    assert count == 100


def test_pad_extends_each_window():
    rl, clock = limiter("2:1", pad_s=0.25)
    for _ in range(3):
        rl.acquire("na1", "m")
    assert clock() == pytest.approx(1.25)


# -- isolation -----------------------------------------------------------------------------

def test_routing_values_have_separate_quotas():
    rl, clock = limiter("2:10")
    rl.acquire("americas", "m")
    rl.acquire("americas", "m")
    assert rl.acquire("europe", "m") == 0  # europe unaffected by americas being full
    assert rl.acquire("americas", "m") == pytest.approx(10)


def test_method_limit_only_holds_back_that_method():
    rl, clock = limiter("100:10")
    rl.acquire("na1", "league")
    rl.update_from_headers("na1", "league", {"X-Method-Rate-Limit": "1:10"})
    assert rl.acquire("na1", "summoner") == 0
    assert rl.acquire("na1", "league") == pytest.approx(10)


# -- learning from headers -----------------------------------------------------------------

def test_app_limits_learned_from_headers():
    rl, clock = limiter("20:1")
    rl.acquire("americas", "m")
    rl.update_from_headers("americas", "m", {"x-app-rate-limit": "2:5"})
    rl.acquire("americas", "m")
    assert rl.acquire("americas", "m") == pytest.approx(5)


def test_raising_a_limit_keeps_history():
    rl, clock = limiter("2:10")
    rl.acquire("na1", "m")
    rl.acquire("na1", "m")
    rl.update_from_headers("na1", "m", {"X-App-Rate-Limit": "3:10"})
    assert rl.acquire("na1", "m") == 0          # one more slot, not three
    assert rl.acquire("na1", "m") == pytest.approx(10)


def test_server_count_ahead_of_ours_is_adopted():
    # Another process using the same key already spent 19 of the 20.
    rl, clock = limiter("20:1,100:120")
    rl.acquire("americas", "m")
    rl.update_from_headers("americas", "m", {"X-App-Rate-Limit-Count": "19:1,19:120"})
    assert rl.acquire("americas", "m") == 0
    assert rl.acquire("americas", "m") == pytest.approx(1)


def test_server_count_behind_ours_is_ignored():
    rl, clock = limiter("3:10")
    for _ in range(3):
        rl.acquire("na1", "m")
    rl.update_from_headers("na1", "m", {"X-App-Rate-Limit-Count": "1:10"})
    assert rl.acquire("na1", "m") == pytest.approx(10)


def test_first_method_header_counts_the_request_it_came_from():
    rl, clock = limiter("100:10")
    rl.acquire("na1", "match")
    rl.update_from_headers("na1", "match", {"X-Method-Rate-Limit": "2:10"})
    assert rl.acquire("na1", "match") == 0
    assert rl.acquire("na1", "match") == pytest.approx(10)


# -- 429 blocks ---------------------------------------------------------------------------

def test_block_holds_whole_routing_value():
    rl, clock = limiter()
    rl.block("americas", 7)
    assert rl.acquire("americas", "anything") == pytest.approx(7)
    assert rl.acquire("europe", "anything") == 0


def test_block_on_one_method():
    rl, clock = limiter()
    rl.block("na1", 3, method="league")
    assert rl.acquire("na1", "summoner") == 0
    assert rl.acquire("na1", "league") == pytest.approx(3)


def test_overlapping_blocks_keep_the_later_one():
    b = Bucket([Window(10, 1)])
    b.block(5, now=0)
    b.block(2, now=0)
    assert b.wait(0) == 5


# -- threads ------------------------------------------------------------------------------

def test_concurrent_threads_respect_the_limit():
    rl = RateLimiter("5:0.2", pad_s=0.0)
    sent, lock = [], threading.Lock()

    def worker():
        for _ in range(4):
            rl.acquire("americas", "m")
            with lock:
                sent.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sent) == 24
    # Small tolerance for the gap between acquire() returning and the timestamp being taken.
    assert max_in_any_window(sent, 0.19) <= 5
