from types import SimpleNamespace

import pytest

from riftwatch import cli


def test_every_command_parses():
    parser = cli.build_parser()
    for argv in (["doctor"], ["lookup", "A#B"], ["sync", "A#B", "-n", "5"],
                 ["backfill", "A#B", "--since", "2026-01-01"], ["crawl", "--tiers", "GOLD"],
                 ["crawl", "--high-elo", "--regions", "na,euw,kr", "--players", "50"],
                 ["baselines"], ["features"], ["coach", "A#B", "--last", "--offline"],
                 ["cache"], ["watchdog", "--status"], ["watchdog", "--record"],
                 ["record"], ["record", "--import"], ["ml", "dataset"], ["ml", "train", "--roles", "JUNGLE"], ["serve", "--port", "9000"],
                 ["db", "migrate"]):
        assert callable(parser.parse_args(argv).func)


def test_version_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_client_stats_lists_each_429():
    stats = {"requests": 10, "retries": 1, "429:method": 1}
    hit = {"type": "method", "routing": "americas", "method": "match-v5.timeline",
           "retry_after": 2.0, "app_count": "5:1,90:120", "method_count": "2001:10"}
    api = SimpleNamespace(client=SimpleNamespace(stats=stats, rate_limited=[hit]))
    text = cli._client_stats(api)
    first, second = text.split("\n")
    assert first == "10 API requests, 1 rate-limited (429), 1 retries"
    assert "429 from method limit on americas match-v5.timeline" in second
