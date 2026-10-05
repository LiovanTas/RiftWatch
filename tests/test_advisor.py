import pytest

from riftwatch.ml import advisor as adv
from riftwatch.ml.advisor import Advisor, Option
from tests.fixtures import build_game


def opt(decision, share, objective=0.3, death=0.3, gold=0.0):
    return Option(decision, share, objective, death, gold)


# -- the key-moment rules, on crafted options ------------------------------------------------------

class Stub(Advisor):
    """An advisor whose model answers are scripted per minute."""

    def __init__(self, per_minute):
        super().__init__(models_dir=None)
        self.per_minute = per_minute

    def available(self, role):
        return True

    def model(self, role):
        return None

    def _options(self, trained, minutes):
        # Unscripted minutes: an even split, so no consensus and nothing to flag.
        even = [opt(d, 1 / 9) for d in ("farm", "gank_top", "gank_mid", "gank_bot", "objective",
                                         "invade", "rotate", "base", "other")]
        return [self.per_minute.get(e.minute, even) for e in minutes]


def review_with(per_minute, decisions=None):
    """Run the stub on a fixture game, forcing the jungler's decision at chosen minutes."""
    match, timeline = build_game(events=[])
    puuid = match["info"]["participants"][1]["puuid"]          # blue jungler
    from riftwatch.ml import situations

    real = situations.examples

    def forced(m, t, role):
        out = real(m, t, role)
        for e in out:
            if decisions and e.participant_id == 2 and e.minute in decisions:
                e.decision = decisions[e.minute]
        return out

    adv.examples, saved = forced, adv.examples
    try:
        return Stub(per_minute).review(match, timeline, puuid)
    finally:
        adv.examples = saved


def full(**shares):
    """Options for every jungle decision, with given (share, objective, death)."""
    names = ("farm", "gank_top", "gank_mid", "gank_bot", "objective", "invade", "rotate", "base", "other")
    return [opt(d, *shares.get(d, (0.01, 0.3, 0.3))) for d in names]


def test_going_against_a_high_elo_consensus_is_a_key_moment():
    r = review_with({6: full(farm=(0.62, 0.30, 0.30), base=(0.04, 0.30, 0.30))},
                    decisions={6: "base"})
    [m] = r.moments
    assert (m.minute, m.did.decision, m.best.decision) == (6, "base", "farm")
    assert m.gain == pytest.approx(0.62 - 0.04)
    assert not m.outcomes_meaningful            # same predicted outcomes either way


def test_no_moment_without_a_consensus():
    # The player's choice was rare, but high-elo players were split -- nothing to point at.
    r = review_with({6: full(farm=(0.35, 0.3, 0.3), gank_top=(0.30, 0.3, 0.3), base=(0.05, 0.3, 0.3))},
                    decisions={6: "base"})
    assert r.moments == []


def test_no_moment_when_the_choice_was_not_rare():
    r = review_with({6: full(farm=(0.55, 0.3, 0.3), base=(0.25, 0.3, 0.3))}, decisions={6: "base"})
    assert r.moments == []


def test_other_is_never_coached():
    r = review_with({6: full(other=(0.02, 0.0, 0.9), farm=(0.8, 0.9, 0.1))},
                    decisions={6: "other"})
    assert r.moments == []


def test_matching_a_consensus_is_a_strength():
    r = review_with({7: full(objective=(0.6, 0.9, 0.1), farm=(0.3, 0.2, 0.3))},
                    decisions={7: "objective"})
    assert [m.minute for m in r.good] == [7] and r.moments == []


def test_outcomes_are_quoted_only_when_the_gap_is_meaningful():
    from riftwatch.coach.evidence import EvidenceSet, _Builder, _review_evidence
    from riftwatch.coach.grounding import validate
    from riftwatch.coach.pipeline import offline_coach

    flat = review_with({6: full(farm=(0.62, 0.30, 0.30), base=(0.04, 0.30, 0.30))},
                       decisions={6: "base"})
    big = review_with({6: full(farm=(0.62, 0.70, 0.20), base=(0.04, 0.20, 0.40))},
                      decisions={6: "base"})
    for review, quoted in ((flat, False), (big, True)):
        b = _Builder()
        _review_evidence(b, review)
        ev = EvidenceSet(b.items)
        text = " ".join(e.text for e in ev.items)
        assert ("At 6:00 you went back to base. In similar situations only 4% of "
                "Grandmaster/Challenger junglers did that; 62% farmed camps.") in text
        assert ("took an objective within the next 3 minutes 70% of the time" in text) is quoted
        assert validate(offline_coach(ev), ev) == []


# -- end to end with real (tiny) models ----------------------------------------------------------

def test_train_then_review(tmp_path):
    import pandas as pd

    from riftwatch.ml.situations import examples
    from riftwatch.ml.train import save, train

    rows = []
    for i in range(40):
        match, timeline = build_game(f"NA1_{9000 + i}", cs_bonus=i * 0.05)
        rows += [e.row() for e in examples(match, timeline, "JUNGLE")]
    pd.DataFrame(rows).to_parquet(tmp_path / "JUNGLE.parquet", index=False)
    trained = train(tmp_path / "JUNGLE.parquet")
    save(trained, tmp_path / "models")
    assert (tmp_path / "models" / "JUNGLE.joblib").exists()
    assert trained.metrics["test_games"] > 0

    match, timeline = build_game("NA1_9999")
    puuid = match["info"]["participants"][1]["puuid"]
    review = Advisor(tmp_path / "models").review(match, timeline, puuid)
    assert review.role == "JUNGLE" and review.minutes == 14 and 0 <= review.agreement <= 1
    # A role with no trained model is simply skipped.
    mid = match["info"]["participants"][2]["puuid"]
    assert Advisor(tmp_path / "models").review(match, timeline, mid) is None


def test_moments_appear_in_the_html_report():
    from riftwatch.coach.evidence import game_evidence
    from riftwatch.coach.pipeline import offline_coach
    from riftwatch.report.html import game_html
    from riftwatch.web.serialize import _review_json
    from tests.test_report import result

    r = result()
    r.review = review_with({6: full(farm=(0.62, 0.30, 0.30), base=(0.04, 0.30, 0.30))},
                           decisions={6: "base"})
    game, _ = r.games[0]
    r.evidence = game_evidence(game, r.scores[0], review=r.review)
    r.output = offline_coach(r.evidence)
    page = game_html(r)
    assert "Compared with high-elo play" in page and "At 6:00 you went back to base" in page
    js = _review_json(r.review)
    assert js["moments"][0]["alternative"]["decision"] == "farm"
