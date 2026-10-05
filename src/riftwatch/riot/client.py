"""HTTP client for the Riot API.

Every request goes through the :class:`RateLimiter` first, and every response's headers
go back into it. The client itself never sleeps: a 429's ``Retry-After`` and the backoff
after a 5xx or a network error are turned into limiter blocks, so other threads calling
the same endpoint hold off too, and the next ``acquire()`` does the waiting.

Status handling:

* 200 -> parsed JSON
* 404 -> ``None`` (unknown Riot ID, match not found -- an answer, not a failure)
* 429 -> wait and retry. ``X-Rate-Limit-Type`` says whose limit it was: ``application``
  (all our calls on that routing value), ``method`` (this endpoint), or ``service`` (Riot's
  backend is overloaded; no ``Retry-After``, so we back off exponentially)
* 500/502/503/504 and connection errors -> exponential backoff with jitter, then retry
* 401/403 -> :class:`RiotAuthError` immediately (almost always an expired dev key)
* any other 4xx -> :class:`RiotApiError` immediately; retrying won't change the answer

The API key is sent as the ``X-Riot-Token`` header and appears in no message or repr.
"""

from __future__ import annotations

import random
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from riftwatch.riot.ratelimit import RateLimiter
from riftwatch.riot.routing import ROUTING_VALUES

RETRYABLE_STATUS = frozenset({500, 502, 503, 504})
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0


class RiotApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RiotAuthError(RiotApiError):
    pass


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after")
    try:
        return max(float(value), 0.0) if value is not None else None
    except ValueError:
        return None  # Retry-After may be an HTTP date; Riot sends seconds, so treat as absent


class RiotClient:
    def __init__(
        self,
        api_key: str,
        limiter: RateLimiter | None = None,
        *,
        max_retries: int = 4,
        timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._limiter = limiter or RateLimiter()
        self._max_retries = max_retries
        self._rng = rng
        self._http = httpx.Client(
            headers={"X-Riot-Token": api_key, "Accept": "application/json"},
            timeout=timeout_s,
            transport=transport,
        )
        # requests, retries, 429:application / 429:method / 429:service, 5xx, network_errors
        self.stats: Counter[str] = Counter()
        # Every 429, so a rate-limit hit can be traced to whose limit it was.
        self.rate_limited: list[dict[str, Any]] = []
        self._stats_lock = threading.Lock()  # get() runs on several threads at once

    def __repr__(self) -> str:
        return f"RiotClient(requests={self.stats['requests']})"

    def __enter__(self) -> RiotClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _count(self, key: str) -> None:
        with self._stats_lock:
            self.stats[key] += 1

    def _backoff(self, attempt: int) -> float:
        """Full jitter: uniform in [0, min(cap, base * 2**attempt)]."""
        return self._rng() * min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2**attempt)

    def get(
        self,
        routing: str,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        """GET ``path`` on ``routing``'s host.

        ``method`` names the endpoint for method-level rate limits, e.g.
        ``"match-v5.match"``. Path segments must already be URL-encoded by the caller.
        """
        if routing not in ROUTING_VALUES:
            raise ValueError(f"unknown routing value {routing!r}")
        url = f"https://{routing}.api.riotgames.com{path}"
        where = f"{method} {path}"

        for attempt in range(self._max_retries + 1):
            if attempt:
                self._count("retries")
            self._limiter.acquire(routing, method)
            self._count("requests")
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                self._count("network_errors")
                last_error = f"{where}: {type(exc).__name__}"
                self._limiter.block(routing, self._backoff(attempt), method=method)
                continue

            self._limiter.update_from_headers(routing, method, response.headers)
            status = response.status_code

            if status == 200:
                return response.json()
            if status == 404:
                return None
            if status == 429:
                kind = response.headers.get("x-rate-limit-type", "service").lower()
                self._count(f"429:{kind}")
                wait = _retry_after(response.headers)
                with self._stats_lock:
                    self.rate_limited.append({
                        "type": kind, "routing": routing, "method": method,
                        "retry_after": wait, "at": time.time(),
                        "app_count": response.headers.get("x-app-rate-limit-count"),
                        "method_count": response.headers.get("x-method-rate-limit-count"),
                    })
                if wait is None:
                    wait = self._backoff(attempt)
                # An application 429 means every endpoint on this host is over quota.
                self._limiter.block(
                    routing, wait, method=None if kind == "application" else method
                )
                last_error = f"{where}: 429 ({kind})"
                continue
            if status in RETRYABLE_STATUS:
                self._count("5xx")
                self._limiter.block(routing, self._backoff(attempt), method=method)
                last_error = f"{where}: {status}"
                continue
            if status in (401, 403):
                raise RiotAuthError(
                    f"{where}: {status} -- the Riot API key was rejected. Development keys "
                    "expire every 24h; get a new one at https://developer.riotgames.com",
                    status,
                )
            raise RiotApiError(f"{where}: unexpected status {status}", status)

        raise RiotApiError(f"gave up after {self._max_retries + 1} attempts; last: {last_error}")
