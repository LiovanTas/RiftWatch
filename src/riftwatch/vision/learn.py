"""Learn laning decisions from the video library.

Every moment a video's player stands within trading range of their lane opponent, outside
a trade, is a *situation*: game time, both health bars, the distance between them, the
minions on each side, other enemies close by. Two questions are learned from situations:

* **decision** -- does this player start a trade within the next DECISION_S? A classifier
  over all situations: how often high-elo players take the trade from a spot like this.
* **outcome** -- when they do, how does it go? A regressor over the situations where a trade
  started, predicting its net (health points taken minus lost).

Models are evaluated the way the rest of RiftWatch's models are: whole videos are held out,
so a game's moments are never split between training and testing, and each model is
compared with the plain base rate it has to beat. With few videos the numbers say so --
``train`` refuses below MIN_VIDEOS.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import psycopg

from riftwatch.vision import lane

DECISION_S = 1.0          # a trade the player starts within this long counts as "took it"
MIN_VIDEOS = 5            # fewer games than this and a held-out score means nothing
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
FEATURES = (["t_min", "me", "opponent", "hp_diff", "distance", "my_minions", "their_minions",
             "minion_edge", "others", "mana", "level", "depth"]
            + [f"{k.lower()}_ready" for k in lane.SLOTS] + [f"role_{r}" for r in ROLES])
MODEL_FILE = "lane_trades.joblib"


@dataclass
class Situation:
    video_id: int
    t: float
    features: dict
    started: bool                 # the player started a trade within DECISION_S
    net: float | None             # that trade's net, health points (taken - lost)
    died: bool | None = None      # ... and whether the player died right after it
    back: bool | None = None      # ... or had to go back to base within 45 s


def _features(t: float, me: float, opp: float, distance: float, mine: int | None,
              theirs: int | None, others: int, role: str | None, *, mana: float | None = None,
              ready: int | None = None, level: int | None = None,
              depth: float | None = None) -> dict:
    """The situation as the models see it. Anything the video didn't show is NaN: the
    gradient-boosted models learn what an unknown means rather than reading it as a value."""
    nan = float("nan")
    f = {"t_min": t / 60, "me": me, "opponent": opp, "hp_diff": me - opp, "distance": distance,
         "my_minions": -1 if mine is None else mine, "their_minions": -1 if theirs is None else theirs,
         "minion_edge": 0 if mine is None or theirs is None else mine - theirs, "others": others,
         "mana": nan if mana is None else mana, "level": nan if level is None else level,
         "depth": nan if depth is None else depth}
    for i, k in enumerate(lane.SLOTS):
        f[f"{k.lower()}_ready"] = nan if ready is None else float(ready >> i & 1)
    f.update({f"role_{r}": float(role == r) for r in ROLES})
    return f


def situations(conn: psycopg.Connection, view: str = "spectator") -> list[Situation]:
    """Every in-range, out-of-trade moment in the processed videos, labelled."""
    roles = dict(conn.execute("SELECT id, role FROM videos WHERE status = 'done' AND view = %s",
                              (view,)).fetchall())
    trades: dict[int, list[tuple]] = defaultdict(list)
    for vid, start, end, me_lost, opp_lost, started_by, died, back in conn.execute(
            """SELECT t.video_id, t.start_s, t.end_s, t.me_lost, t.opponent_lost, t.started_by,
                      t.died, t.back_after_s
                 FROM video_trades t JOIN videos v ON v.id = t.video_id
                WHERE v.status = 'done' AND v.view = %s AND NOT t.skirmish""", (view,)):
        trades[vid].append((start, end, me_lost, opp_lost, started_by, died, back))
    out: list[Situation] = []
    for (vid, t, me, opp, dist, others, mine, theirs, mana, ready, level,
         dep) in conn.execute(
            """SELECT s.video_id, s.t, s.me, s.opponent, s.distance, s.others, s.my_minions,
                      s.their_minions, s.mana, s.ready, s.level, s.depth
                 FROM video_samples s JOIN videos v ON v.id = s.video_id
                WHERE v.status = 'done' AND v.view = %s AND s.me IS NOT NULL
                  AND s.opponent IS NOT NULL AND s.distance <= %s
                ORDER BY s.video_id, s.t""", (view, lane.TRADE_RANGE)):
        vt = trades.get(vid, [])
        if any(start <= t <= end for start, end, *_ in vt):
            continue                                  # mid-trade: not a decision point
        nxt = next(((ml, ol, by, died, back) for start, _, ml, ol, by, died, back in vt
                    if t < start <= t + DECISION_S), None)
        started = nxt is not None and nxt[2] in ("you", "both")
        features = _features(t, me, opp, dist, mine, theirs, others, roles.get(vid),
                             mana=mana, ready=ready, level=level, depth=dep)
        out.append(Situation(vid, t, features, started, (nxt[1] - nxt[0]) if started else None,
                             nxt[3] if started else None,
                             (nxt[4] is not None) if started else None))
    return out


def to_frame(rows: list[Situation]):
    import pandas as pd

    return pd.DataFrame([{**r.features, "video_id": r.video_id, "t": r.t,
                          "started": r.started, "net": r.net, "died": r.died, "back": r.back}
                         for r in rows])


def _seen(df) -> list[str]:
    """The features the data shows at least once (a video type that never shows one -- the
    HUD panel in your own recordings -- leaves it all unknown, and there's nothing to learn)."""
    return [f for f in FEATURES if df[f].notna().any()]


def train(conn: psycopg.Connection, models_dir: Path, view: str = "spectator",
          seed: int = 0) -> dict:
    """Fit both models, report held-out scores, save them. Raises when there are too few
    videos for the scores to mean anything."""
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score

    df = to_frame(situations(conn, view))
    videos = sorted(df["video_id"].unique()) if len(df) else []
    if len(videos) < MIN_VIDEOS:
        raise ValueError(f"{len(videos)} processed video(s) with laning data; need at least "
                         f"{MIN_VIDEOS} before held-out scores mean anything")
    rng = np.random.default_rng(seed)
    test_videos = set(rng.choice(videos, size=max(1, len(videos) // 5), replace=False).tolist())
    test = df["video_id"].isin(test_videos)
    y = df["started"].astype(int)

    decision = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
    cols = _seen(df[~test])
    decision.fit(df.loc[~test, cols], y[~test])
    p = decision.predict_proba(df.loc[test, cols])[:, 1]
    base = np.full(test.sum(), y[~test].mean())
    metrics = {"videos": len(videos), "test_videos": len(test_videos), "situations": len(df),
               "trade_rate": round(float(y.mean()), 4)}
    if y[test].nunique() > 1:
        metrics["decision_auc"] = round(float(roc_auc_score(y[test], p)), 3)
    metrics["decision_logloss"] = round(float(log_loss(y[test], p, labels=[0, 1])), 4)
    metrics["decision_logloss_base"] = round(float(log_loss(y[test], base, labels=[0, 1])), 4)

    started = df[df["started"]]
    outcome, outcome_cols = None, []
    if len(started) >= 20:
        st = started["video_id"].isin(test_videos)
        outcome = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed)
        cols = _seen(started[~st])
        outcome.fit(started.loc[~st, cols], started.loc[~st, "net"])
        if st.any():
            pred = outcome.predict(started.loc[st, cols])
            metrics["outcome_mae"] = round(float(mean_absolute_error(started.loc[st, "net"], pred)), 4)
            metrics["outcome_mae_base"] = round(float(mean_absolute_error(
                started.loc[st, "net"], np.full(st.sum(), started.loc[~st, "net"].mean()))), 4)
        # Refit on everything for use; the scores above are from the held-out games.
        outcome_cols = _seen(started)
        outcome.fit(started[outcome_cols], started["net"])
    decision_cols = _seen(df)
    decision.fit(df[decision_cols], y)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"decision": decision, "outcome": outcome, "features": decision_cols,
                 "outcome_features": outcome_cols, "metrics": metrics}, models_dir / MODEL_FILE)
    (models_dir / "lane_trades.metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load(models_dir: Path) -> dict | None:
    path = Path(models_dir) / MODEL_FILE
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


def judge(model: dict, rows: list[dict]) -> list[tuple[float, float | None]]:
    """(chance a high-elo player starts a trade here, expected net if they do) per situation."""
    import pandas as pd

    X = pd.DataFrame(rows)
    p = model["decision"].predict_proba(X[model["features"]])[:, 1]
    nets = (model["outcome"].predict(X[model.get("outcome_features", model["features"])])
            if model["outcome"] is not None else [None] * len(rows))
    return [(float(a), None if b is None else float(b)) for a, b in zip(p, nets, strict=True)]
