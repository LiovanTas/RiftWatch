import json
import re

from riftwatch.coach.evidence import game_evidence
from riftwatch.coach.pipeline import CoachResult, offline_coach
from riftwatch.report.html import game_html, recent_html
from riftwatch.report.terminal import bar, render
from tests.test_coach import make_score


def result(champion_name="Ahri"):
    game, gs = make_score()
    gs.participant.champion_name = champion_name
    ev = game_evidence(game, gs)
    return CoachResult("game", gs.participant.puuid, "PLATINUM", ev, offline_coach(ev),
                       "offline", games=[(game, gs.participant)], scores=[gs])


def embedded_data(page: str) -> dict:
    raw = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1)
    return json.loads(raw.replace("<\\/", "</"))


def test_game_html_embeds_chart_data():
    page = game_html(result())
    data = embedded_data(page)
    cs = next(c for c in data["curves"] if c["name"] == "cs")
    assert cs["points"][0]["m"] == 1 and cs["points"][-1]["m"] == 26
    assert cs["points"][5]["p50"] == 7          # baseline median carried per minute
    assert len(data["deaths"]) == 2
    assert "<title>Ahri middle review</title>" in page


def test_untrusted_names_cannot_break_out():
    page = game_html(result('</script><script>alert(1)</script>'))
    assert "<script>alert(1)</script>" not in page
    assert page.count("</script>") == 2         # only our own two script tags close


def test_recent_html_renders():
    r = result()
    r.scope = "recent"
    assert "Last 1 ranked games" in recent_html(r)


def test_terminal_render_and_bar():
    assert bar(0) == "[" + "." * 20 + "]" and bar(100) == "[" + "#" * 20 + "]"
    text = render(result())
    assert "offline template coach" in text and "Ahri middle" in text
