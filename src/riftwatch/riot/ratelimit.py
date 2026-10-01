"""Client-side rate limiting for the Riot API.

Riot enforces two layers of limits, separately for every routing value (``na1``,
``americas``, ...):

* an **application** limit shared by every call made with the key, and
* a **method** limit per endpoint (match-v5 ``/matches/{id}`` has its own quota).

Each layer is several windows at once -- a dev key is ``20:1,100:120``: at most 20 requests
in any 1 s *and* 100 in any 120 s. Responses announce the limits in ``X-App-Rate-Limit`` /
``X-Method-Rate-Limit`` and Riot's own tally in the matching ``-Count`` headers, so limits
are learned from responses instead of hardcoded.

Each window is a token bucket whose tokens refill **one full window after they were spent**
rather than continuously. A continuously refilling bucket is not safe here: drain all 100
tokens of a 100/120 s bucket at t=0, keep going at the refill rate, and by t=119 s it has
sent ~199 requests inside one 120 s window -- a 429. Per-token refill guarantees no interval
of the window's length ever holds more than ``limit`` requests, which satisfies Riot whether
it counts fixed or sliding windows.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

# Riot dev-key application limits, used for a routing value until its first response
# tells us the real ones. Method limits are unknown until then and not enforced.
DEV_KEY_APP_LIMITS = "20:1,100:120"

# Requests are timestamped when sent, but Riot counts them when they arrive. Padding each
# window absorbs that latency so a request sent "just in time" can't land early.
DEFAULT_PAD_S = 0.1


@dataclass(frozen=True)
class Window:
    limit: int
    seconds: float


def parse_limits(header: str) -> list[Window]:
    """``"20:1,100:120"`` -> ``[Window(20, 1), Window(100, 120)]``."""
    windows = []
    for part in header.split(","):
        limit, sep, seconds = part.strip().partition(":")
        if not sep:
            raise ValueError(f"malformed rate-limit header {header!r}")
        window = Window(int(limit), float(seconds))
        if window.limit <= 0 or window.seconds <= 0:
            raise ValueError(f"non-positive rate limit in {header!r}")
        windows.append(window)
    return windows


def parse_counts(header: str) -> dict[float, int]:
    """``"3:1,57:120"`` -> ``{1.0: 3, 120.0: 57}`` (window seconds -> requests counted)."""
    return {w.seconds: w.limit for w in parse_limits(header)}


class Bucket:
    """All the windows of one limit (one routing value's app limit, or one method's).

    One send history is checked against every window, so changing the limits -- even a
    window's length -- never forgets requests already sent.
    """

    def __init__(self, windows: list[Window], pad_s: float = DEFAULT_PAD_S) -> None:
        self.windows = windows
        self.pad_s = pad_s
        self.blocked_until = 0.0
        self._spent: deque[float] = deque()  # send times, oldest first

    def set_limits(self, windows: list[Window]) -> None:
        self.windows = windows

    def _span(self, window: Window) -> float:
        return window.seconds + self.pad_s

    def _in_window(self, window: Window, now: float) -> list[float]:
        start = now - self._span(window)
        return [t for t in self._spent if t > start]

    def _expire(self, now: float) -> None:
        # Keep enough history for the longest window, plus a margin in case a later
        # header announces a longer one.
        keep = max((self._span(w) for w in self.windows), default=0.0) * 2
        while self._spent and self._spent[0] <= now - keep:
            self._spent.popleft()

    def wait(self, now: float) -> float:
        self._expire(now)
        wait = max(self.blocked_until - now, 0.0)
        for w in self.windows:
            sent = self._in_window(w, now)
            if len(sent) >= w.limit:
                # A slot frees when the request `limit` places back from the newest ages out.
                wait = max(wait, sent[len(sent) - w.limit] + self._span(w) - now)
        return wait

    def consume(self, now: float) -> None:
        self._spent.append(now)

    def sync_counts(self, counts: Mapping[float, int], now: float) -> None:
        """Riot saw more requests than we sent (another process on the same key, a call we
        didn't track): record the difference as sent just now, the conservative choice."""
        missing = 0
        for w in self.windows:
            if w.seconds in counts:
                missing = max(missing, counts[w.seconds] - len(self._in_window(w, now)))
        self._spent.extend([now] * missing)

    def block(self, seconds: float, now: float) -> None:
        self.blocked_until = max(self.blocked_until, now + seconds)


class RateLimiter:
    """Thread-safe limiter keyed by routing value and method.

    Usage, once per request::

        limiter.acquire("americas", "match-v5.match")      # blocks until allowed
        response = send(...)
        limiter.update_from_headers("americas", "match-v5.match", response.headers)
    """

    def __init__(
        self,
        default_app_limits: str = DEV_KEY_APP_LIMITS,
        pad_s: float = DEFAULT_PAD_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._default_app = parse_limits(default_app_limits)
        self._pad_s = pad_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._app: dict[str, Bucket] = {}
        self._method: dict[tuple[str, str], Bucket] = {}

    def _buckets(self, routing: str, method: str) -> list[Bucket]:
        app = self._app.get(routing)
        if app is None:
            app = self._app[routing] = Bucket(self._default_app, self._pad_s)
        buckets = [app]
        if (routing, method) in self._method:
            buckets.append(self._method[routing, method])
        return buckets

    def acquire(self, routing: str, method: str) -> float:
        """Block until a request may be sent, and count it. Returns seconds spent waiting."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                buckets = self._buckets(routing, method)
                wait = max(b.wait(now) for b in buckets)
                if wait <= 0:
                    for b in buckets:
                        b.consume(now)
                    return waited
            # Sleep outside the lock so other routing values keep flowing; re-check after,
            # since another thread may have taken the slot we were waiting for.
            self._sleep(wait)
            waited += wait

    def update_from_headers(self, routing: str, method: str, headers: Mapping[str, str]) -> None:
        """Learn limits and Riot's request tally from a response's headers."""
        h = {k.lower(): v for k, v in headers.items()}
        with self._lock:
            now = self._clock()
            self._buckets(routing, method)  # make sure the app bucket exists
            app = self._app[routing]
            if "x-app-rate-limit" in h:
                app.set_limits(parse_limits(h["x-app-rate-limit"]))
            if "x-app-rate-limit-count" in h:
                app.sync_counts(parse_counts(h["x-app-rate-limit-count"]), now)

            if "x-method-rate-limit" in h:
                windows = parse_limits(h["x-method-rate-limit"])
                bucket = self._method.get((routing, method))
                if bucket is None:
                    bucket = self._method[routing, method] = Bucket(windows, self._pad_s)
                    # This response was already counted by acquire() before the bucket existed.
                    bucket.consume(now)
                else:
                    bucket.set_limits(windows)
            if "x-method-rate-limit-count" in h and (routing, method) in self._method:
                self._method[routing, method].sync_counts(
                    parse_counts(h["x-method-rate-limit-count"]), now
                )

    def block(self, routing: str, seconds: float, method: str | None = None) -> None:
        """Hold back all calls on ``routing`` (or just ``method`` there) for ``seconds`` --
        what a 429's ``Retry-After`` asks for."""
        with self._lock:
            now = self._clock()
            if method is None:
                self._buckets(routing, "")
                self._app[routing].block(seconds, now)
            else:
                bucket = self._method.get((routing, method))
                if bucket is None:
                    # No known limits yet; an empty bucket still carries the block.
                    bucket = self._method[routing, method] = Bucket([], self._pad_s)
                bucket.block(seconds, now)
