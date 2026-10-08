import os
import random

import numpy as np
import pandas as pd
import pytest

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")
pytest.importorskip("sklearn")


# -- features and labels on a hand-built video (no database) -------------------------------

def tiny():
    """20 s at 4 readings a second: in range, a trade the player starts at 5 s (won), quiet,
    one the opponent starts at 12 s, then the player dies at 16 s."""
    t = np.arange(0, 20, 0.25) + 300
    n = len(t)
    s = pd.DataFrame({"t": t, "me": 0.8, "opponent": 0.6, "distance": 0.2, "others": 0.0,
                      "my_minions": 4.0, "their_minions": 2.0, "mana": 0.5,
                      "ready": float(0b111111), "level": 6.0, "depth": 0.1, "in_base": 0.0,
                      "dead": 0.0})
    s.loc[(t > 305) & (t <= 306), "opponent"] = np.linspace(0.55, 0.4, 4)
    s.loc[t > 306, "opponent"] = 0.4
    s.loc[(t > 312) & (t <= 313), "me"] = np.linspace(0.75, 0.6, 4)
    s.loc[t > 313, "me"] = 0.6
    s.loc[t >= 316, ["me", "mana", "ready", "level", "depth", "in_base"]] = np.nan
    s.loc[t >= 316, "dead"] = 1.0
    s.loc[t < 301, "level"] = 5.0                       # level 6 reached at 301 s
    trades = pd.DataFrame([
        (305.0, 306.0, 0.0, 0.2, "you", False, "won", False, None),
        (312.0, 313.0, 0.2, 0.0, "opponent", False, "lost", True, None)],
        columns=["start_s", "end_s", "me_lost", "opponent_lost", "started_by", "skirmish",
                 "result", "died", "back_after_s"])
    assert n == 80
    return s, trades


def test_features_carry_the_recent_past_and_nothing_after():
    from riftwatch.brain import features

    s, trades = tiny()
    f = features.derive(s, trades, "TOP", "Irelia", "Renekton")
    at = dict(zip(s["t"], range(len(s)), strict=True))
    r = f.iloc[at[308.0]]
    assert r["since_trade_s"] == pytest.approx(2.0) and r["trades_so_far"] == 1
    assert r["last_trade_net"] == pytest.approx(0.2) and r["net_so_far"] == pytest.approx(0.2)
    assert r["opp_trend_3s"] == pytest.approx(-0.2) and r["me_trend_3s"] == pytest.approx(0)
    assert r["in_range_s"] == pytest.approx(8.0) and r["minion_edge"] == 2
    assert r["r_ready"] == 1 and r["basics_ready"] == 3 and r["since_level_up_s"] == pytest.approx(7)
    assert r["role_TOP"] == 1 and r["champion"] == "Irelia"
    before = f.iloc[at[304.0]]
    assert before["trades_so_far"] == 0 and before["since_trade_s"] == features.CAP_S
    # The future never leaks in: features at 304 s are the same without the later readings.
    cut = features.derive(s[s["t"] <= 304.0], trades.iloc[:0], "TOP")
    cols = [c for c in features.VARIANTS["full"]]
    assert np.allclose(cut.iloc[-1][cols].to_numpy(float), before[cols].to_numpy(float),
                       equal_nan=True)


def test_labels_look_ahead_from_each_moment():
    from riftwatch.brain import data

    s, trades = tiny()
    lab = data.labels(s, trades)
    mask = data.situations_mask(s, trades)
    at = dict(zip(s["t"], range(len(s)), strict=True))
    assert lab["trade"].iloc[at[304.25]] == 1 and lab["trade"].iloc[at[303.75]] == 0
    assert lab["trade_won"].iloc[at[304.5]] == 1 and lab["trade_net"].iloc[at[304.5]] == pytest.approx(0.2)
    assert np.isnan(lab["trade_net"].iloc[at[303.0]])          # no trade started from there
    assert lab["traded_on"].iloc[at[311.5]] == 1 and lab["trade"].iloc[at[311.5]] == 0
    assert lab["died_15s"].iloc[at[302.0]] == 1 and lab["died_15s"].iloc[at[300.0]] == 0
    assert np.isnan(lab["died_15s"].iloc[at[306.0]])          # the video ends before 15 s pass
    # Health swing over 10 s from 303 s: the opponent lost 20 points in the trade.
    assert lab["swing_10s"].iloc[at[303.0]] == pytest.approx(0.2 - 0.2)
    # In trades, dead, or out of range: not situations.
    assert not mask[at[305.5]] and not mask[at[312.5]] and not mask[at[317.0]] and mask[at[308.0]]


# -- a library of synthetic replays with planted habits -------------------------------------

def high_elo(st):
    """Trades taken when ahead on health, with the ultimate up, or with the bigger wave."""
    return (0.02 + 0.22 * (st["me"] - st["opp"] > 0.1) + 0.2 * (st["r_up"] and st["level"] >= 6)
            + 0.1 * (st["mine"] - st["theirs"] >= 2))


def replay(conn, vid, seed, *, role="TOP", view="spectator", match_id=None, policy=high_elo,
           forced=(), panel=True):
    """A processed video's samples and trades, simulated second by second. ``forced``:
    (second, won) trades the player starts at that game second regardless of ``policy``."""
    from riftwatch.vision import library

    rng = random.Random(seed)
    conn.execute(
        """INSERT INTO videos (id, path, fingerprint, title, champion, opponent, role, tier,
                               view, match_id, status, analyzer_version, game_offset_s,
                               clock_agreement)
           VALUES (%s, %s, %s, %s, %s, 'Renekton', %s, 'GRANDMASTER', %s, %s, 'done', %s, 60, 0.95)""",
        (vid, f"v{vid}.mp4", f"fp{vid}", f"video {vid}", rng.choice(["Irelia", "Fiora", "Jax"]),
         role, view, match_id, library.ANALYZER_VERSION))
    st = {"me": 1.0, "opp": 1.0, "dist": 0.25, "mine": 3, "theirs": 3, "mana": 1.0,
          "r_up": True, "r_cd": 0.0, "level": 1}
    samples, trades = [], []
    forced = dict(forced)
    t = 90.0
    away_until, dead_until, base_until, opp_away = 0.0, 0.0, 0.0, 0.0

    def emit(t, me=None, opp=None, dist=None, dead=False, in_base=False, others=0):
        hud = panel and not dead
        ready = (0b110111 | (8 if st["r_up"] and st["level"] >= 6 else 0)) if hud else None
        samples.append((vid, round(t, 3), me, opp, dist, others,
                        st["mine"] if not dead else None, st["theirs"] if not dead else None,
                        round(st["mana"], 3) if hud else None, ready,
                        st["level"] if hud else None, 0.5, 0.5,
                        (-0.9 if in_base else rng.uniform(-0.1, 0.15)) if panel and not dead else None,
                        in_base if panel and not dead else None, dead))

    while t < 840:
        st["level"] = min(18, 1 + int((t - 90) // 55))
        st["r_cd"] = max(0.0, st["r_cd"] - 1)
        st["r_up"] = st["r_cd"] == 0
        st["mana"] = min(1.0, st["mana"] + 0.004)
        st["mine"] = max(0, min(6, st["mine"] + rng.choice((-1, 0, 0, 1))))
        st["theirs"] = max(0, min(6, st["theirs"] + rng.choice((-1, 0, 0, 1))))
        if t < dead_until:
            for k in range(4):
                emit(t + 0.25 * k, dead=True)
            t += 1
            continue
        if t < base_until or t < away_until:
            in_base = t < base_until
            for k in range(4):
                emit(t + 0.25 * k, me=st["me"], in_base=in_base)
            t += 1
            continue
        opp_here = t >= opp_away
        if not opp_here and st["opp"] < 1:
            st["opp"] = 1.0
        st["dist"] = rng.choice((0.15, 0.25, 0.3, 0.3, 0.5, 0.6)) if opp_here else None
        others = 1 if rng.random() < 0.04 else 0
        in_range = st["dist"] is not None and st["dist"] <= 0.35
        starter = None
        if in_range and int(t) in forced:
            starter, win = "you", forced.pop(int(t))
        elif in_range and rng.random() < policy(st):
            starter = "you"
            good = st["me"] - st["opp"] > 0.1 or (st["r_up"] and st["level"] >= 6)
            win = rng.random() < (0.8 if good else 0.3)
        elif in_range and rng.random() < 0.03 + 0.1 * (st["dist"] < 0.2):
            starter, win = "opponent", rng.random() < 0.3
        for k in range(4):
            emit(t + 0.25 * k, st["me"], st["opp"] if opp_here else None, st["dist"], others=others)
        t += 1
        if starter is None:
            st["me"] = min(1.0, st["me"] + 0.002)
            continue
        # The trade: four readings of both losing health, then three quiet seconds apart.
        start = t - 0.25
        mine_loss, their_loss = (0.06, 0.18) if win else (0.18, 0.06)
        if starter == "you" and st["r_up"] and st["level"] >= 6:
            st["r_cd"] = 60.0
        st["mana"] = max(0.0, st["mana"] - 0.15)
        me0, opp0 = st["me"], st["opp"]
        for k in range(1, 5):
            emit(start + 0.25 * k, max(0.01, me0 - mine_loss * k / 4),
                 max(0.01, opp0 - their_loss * k / 4), st["dist"])
        st["me"], st["opp"] = max(0.01, me0 - mine_loss), max(0.01, opp0 - their_loss)
        t = start + 1.25
        for k in range(12):
            emit(t + 0.25 * k, st["me"], st["opp"], 0.55)
        t += 3
        net = (opp0 - st["opp"]) - (me0 - st["me"])
        died = st["me"] < 0.15 and rng.random() < 0.6
        back = None
        if died:
            dead_until, base_until = t + 2 + 20, t + 2 + 30
            for k in range(8):
                emit(t + 0.25 * k, st["me"], st["opp"], 0.55)
            t += 2
            st["me"] = 1.0
            back = round(t + 20 - start, 1)
        elif st["me"] < 0.3:
            away_until, base_until = t + 25, t + 18
            back = round(t + 8 - start, 1)
            st["me"] = 1.0
        if st["opp"] < 0.2:
            opp_away = t + 25
        trades.append((vid, start, start + 1.0, me0 - st["me"] if not died and back is None
                       else mine_loss, their_loss, starter, False, st["mine"] - st["theirs"],
                       "won" if net >= 0.05 else "lost" if net <= -0.05 else "even",
                       None, None, None, None, died if panel else None,
                       back if panel and back is not None and back <= 45 else None, None))
        t = round(t * 4) / 4
    with conn.cursor() as cur:
        with cur.copy("COPY video_samples (video_id, t, me, opponent, distance, others, "
                      "my_minions, their_minions, mana, ready, level, map_x, map_y, depth, "
                      "in_base, dead) FROM STDIN") as copy:
            seen = set()
            for row in samples:
                if row[1] not in seen and row[1] <= 840:
                    seen.add(row[1])
                    copy.write_row(row)
        cur.executemany(
            "INSERT INTO video_trades VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s, %s, %s, %s) ON CONFLICT DO NOTHING", trades)


GAMES = 8


@pytest.fixture(scope="module")
def library_db():
    if not TEST_DB:
        pytest.skip("RIFTWATCH_TEST_DATABASE_URL not set")
    from riftwatch.db import migrate
    from riftwatch.db.connection import connect

    with connect(TEST_DB) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(c)
        for vid in range(1, GAMES + 1):
            replay(c, vid, seed=vid)
        yield c


@pytest.fixture(scope="module")
def trained(library_db, tmp_path_factory):
    from riftwatch.brain import registry
    from riftwatch.brain.train import TrainConfig, train

    lines = []
    brain = train(library_db, TrainConfig.quick(learning_curve=True), progress=lines.append)
    models = tmp_path_factory.mktemp("models")
    registry.save(brain, models)
    return brain, models, lines


def test_situations_from_the_library(library_db):
    from riftwatch.brain import data

    df = data.load(library_db)
    assert set(df["video_id"]) == set(range(1, GAMES + 1))
    assert 0.02 < df["trade"].mean() < 0.4 and df["traded_on"].sum() > 50
    assert df["trade_net"].notna().sum() > 100 and df["died_15s"].notna().all() is not None
    ahead = df[df["hp_diff"] > 0.1]["trade"].mean()
    behind = df[df["hp_diff"] < -0.1]["trade"].mean()
    assert ahead > 2 * behind                       # the planted habit is in the data


def test_check_keeps_bad_readings_out(library_db):
    from riftwatch.brain import data

    library_db.execute(
        "INSERT INTO videos (id, path, fingerprint, title, role, view, status, analyzer_version) "
        "VALUES (500, 'x.mp4', 'fpx', 'unreadable', NULL, 'spectator', 'done', 1)")
    checks = {c.video_id: c for c in data.check(library_db)}
    assert checks[1].ok and not checks[500].ok
    assert any("older analyser" in p for p in checks[500].problems)
    assert any("role unknown" in w for w in checks[500].warnings)
    assert 500 not in set(data.load(library_db)["video_id"])
    library_db.execute("DELETE FROM videos WHERE id = 500")


def test_training_judges_every_head_on_held_out_games(trained):
    brain, _, lines = trained
    trade = brain.unit("trade", "full")
    assert trade.usable and trade.metrics["auc"] > 0.65
    assert trade.metrics["log_loss"] < trade.metrics["base_log_loss"]
    assert trade.metrics["folds"] == 3 and trade.metrics["games"] == GAMES
    families = {c["family"] for c in trade.metrics["candidates"]}
    assert families == {"trees", "neural"}          # both kinds of model were tried
    assert len(trade.models) == 2 and trade.calibrator is not None
    assert "r_ready" in trade.columns
    assert "champion" in trade.categories or trade.family == "neural"   # the network: numbers only
    assert trade.metrics["importance"]["health"]["loss_increase"] > 0
    basic = brain.unit("trade", "basic")
    assert basic.usable and "mana" not in basic.columns and "r_ready" not in basic.columns
    assert brain.unit("trade_net", "full") is not None
    assert brain.learning_curve and len(brain.learning_curve["points"]) == 4
    assert any("trade/full/all" in line for line in lines)


def test_the_brain_learned_the_planted_habits(trained):
    brain, _, _ = trained
    found = {(p["head"], p["key"]): p for p in brain.patterns}
    ult = found[("trade", "ultimate")]
    assert ult["value_a"] > 1.3 * ult["value_b"] and ult["a"] == "with the ultimate ready"
    health = found[("trade", "health")]
    assert health["value_a"] > health["value_b"] and "x as often" in health["text"]


def test_predictions_and_reasons(trained, library_db):
    from riftwatch.brain import data, explain

    brain, _, _ = trained
    df = data.load(library_db, video_ids=[1])
    pred = brain.predict(df)
    assert {"trade", "trade_sd", "traded_on", "trade_net"} <= set(pred.columns)
    assert pred["trade"].between(0, 1).all() and (pred["trade_sd"] >= 0).all()
    unit = brain.unit("trade", "full")
    row = df[(df["hp_diff"] > 0.3) & (df["r_ready"] == 1)].head(1)
    words = explain.reasons(unit, row, sign=1)[0]
    assert words and any("health" in w or "R ready" in w for w in words)


def test_registry_keeps_versions_and_switches_back(trained):
    from riftwatch.brain import registry

    brain, models, _ = trained
    first = registry.current(models)
    second = registry.save(brain, models, promote=False)
    assert registry.current(models) == first and second != first
    registry.use(models, second)
    assert registry.load(models).version == second
    cards = registry.versions(models)
    assert [c["current"] for c in cards] == [False, True]
    assert cards[0]["units"] and cards[0]["patterns"]
    registry.use(models, first)
    with pytest.raises(ValueError):
        registry.use(models, "nope")


def test_review_finds_missed_and_rare_trades_in_the_players_game(trained, library_db):
    from riftwatch.brain.review import review

    brain, _, _ = trained
    # A passive player who never trades on their own -- except once, far behind, and loses.
    replay(library_db, 900, seed=900, view="player", match_id="NA1_900", panel=False,
           policy=lambda st: 0.0, forced={})
    df = library_db.execute("SELECT t, me, opponent FROM video_samples WHERE video_id = 900 "
                            "AND distance <= 0.35 AND opponent - me > 0.3 ORDER BY t").fetchall()
    assert df, "the simulation should put the player behind at some point"
    library_db.execute("DELETE FROM video_samples WHERE video_id = 900")
    library_db.execute("DELETE FROM video_trades WHERE video_id = 900")
    library_db.execute("DELETE FROM videos WHERE id = 900")
    replay(library_db, 900, seed=900, view="player", match_id="NA1_900", panel=False,
           policy=lambda st: 0.0, forced={int(df[0][0]): False})
    rv = review(library_db, brain, 900)
    assert rv.variant == "basic" and rv.role == "TOP"
    assert rv.expected_trades > 3 * max(rv.trades, 1) and rv.style == "passive"
    kinds = {m.kind for m in rv.moments}
    assert "missed_trade" in kinds
    missed = next(m for m in rv.moments if m.kind == "missed_trade")
    assert "didn't trade" in missed.text and "would have started about" in missed.text
    assert all(m.polarity in ("weakness", "strength") for m in rv.moments)
    assert len(rv.moments) <= 7
    text = rv.summary_text()
    assert "you started" in text and "high-elo top laners" in text


def test_coach_evidence_from_the_brain(trained, library_db):
    from riftwatch.coach.evidence import Evidence, EvidenceSet
    from riftwatch.coach.pipeline import _video_evidence

    brain, _, _ = trained
    ev = EvidenceSet([Evidence("E1", "context", "context", "neutral", "Game NA1_900.")])
    _video_evidence(library_db, ev, "NA1_900", brain)
    texts = [e.text for e in ev.items]
    assert any(t.startswith("Laning brain (trained on 8 high-elo games") for t in texts)
    moments = [e for e in ev.items if e.kind == "decision"]
    assert moments and all(e.area == "laning" and e.data["source"] == "video" for e in moments)
    assert any(t.startswith("What the laning brain learned") for t in texts)


def test_training_refuses_with_too_few_games(tmp_path):
    if not TEST_DB:
        pytest.skip("RIFTWATCH_TEST_DATABASE_URL not set")
    from riftwatch.brain.train import TrainConfig, train
    from riftwatch.db import migrate
    from riftwatch.db.connection import connect

    with connect(TEST_DB) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(c)
        replay(c, 1, seed=1)
        with pytest.raises(ValueError, match="need at least 5"):
            train(c, TrainConfig.quick())
