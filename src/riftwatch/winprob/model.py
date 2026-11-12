"""Train, evaluate and load the win-probability network.

Data: every stored game with a timeline, at every whole minute, from both teams' sides (the
same moment seen from the other side is the same game with the label flipped, which teaches
the network that the game is symmetric). Label: whether that team won.

Model: a small multilayer perceptron (scikit-learn MLPClassifier, two hidden layers, ReLU,
early stopping) on standardised inputs, then isotonic calibration on held-out games so a
"70%" means 70%. Games, not rows, are split into train / calibration / test, so no minute of a
test game was seen in training.

It's measured against a baseline that only knows the gold lead and the minute (logistic
regression): the network has to beat it to be worth using. At prediction time both teams'
sides are asked and averaged, so the two probabilities always add up to one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from riftwatch.winprob.state import FEATURES, Replay

DIRNAME = "winprob"
MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"
VERSION = "wp1"
LAYERS = (64, 32)


def load_dataset(conn, limit: int | None = None, log=print) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, game index, minute) for every stored game with a timeline. Only the parts of
    each timeline the model reads are fetched (per-minute player frames and kill, objective
    and structure events), which keeps it to a few minutes for ~10,000 games."""
    sql = """
        SELECT match_id,
               jsonb_path_query_array(t.raw, '$.info.frames[*].participantFrames'),
               jsonb_path_query_array(t.raw, '$.info.frames[*].events[*] ? (@.type == "CHAMPION_KILL"
                   || @.type == "ELITE_MONSTER_KILL" || @.type == "BUILDING_KILL"
                   || @.type == "DRAGON_SOUL_GIVEN" || @.type == "GAME_END")')
          FROM match_timelines t JOIN matches m USING (match_id)
         WHERE m.queue_id IN (420, 440) ORDER BY match_id""" + (f" LIMIT {int(limit)}" if limit else "")
    xs, ys, games, minutes = [], [], [], []
    started = time.monotonic()
    with conn.transaction(), conn.cursor(name="winprob_dataset") as cur:   # server-side, streamed
        cur.itersize = 200
        cur.execute(sql)
        for g, (_, frames, events) in enumerate(cur):
            replay = Replay(frames, events)
            if replay.winner not in (100, 200) or len(frames) < 6:
                continue
            for m in range(1, len(frames)):               # every whole minute, both sides
                for team, state in zip((100, 200), replay.pair(m * 60_000), strict=True):
                    xs.append([state[k] for k in FEATURES])
                    ys.append(float(replay.winner == team))
                    games.append(g)
                    minutes.append(state["minute"])
            if g and g % 1000 == 0:
                log(f"  {g:,} games read ({time.monotonic() - started:.0f}s)")
    return (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
            np.asarray(games), np.asarray(minutes, dtype=np.float32))


def _split(games: np.ndarray, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.unique(games)
    rng.shuffle(ids)
    n = len(ids)
    train_ids, cal_ids = set(ids[: int(0.7 * n)]), set(ids[int(0.7 * n): int(0.85 * n)])
    train = np.isin(games, list(train_ids))
    cal = np.isin(games, list(cal_ids))
    return train, cal, ~(train | cal)


def _calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """For each band of predicted chance, how often that team actually won."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.sum() >= 50:
            out.append({"predicted": round(float(p[m].mean()), 3), "actual": round(float(y[m].mean()), 3),
                        "rows": int(m.sum())})
    return out


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


class WinModel:
    """The fitted network with its scaler and calibration."""

    def __init__(self, scaler, network, calibrator) -> None:
        self.scaler, self.network, self.calibrator = scaler, network, calibrator
        self.version = VERSION

    def raw(self, X: np.ndarray) -> np.ndarray:
        return self.network.predict_proba(self.scaler.transform(X))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.calibrator.predict(self.raw(X)), 0.001, 0.999)

    def blue(self, blue_states: list[dict], red_states: list[dict]) -> np.ndarray:
        """Blue's win probability for paired states (the same moments from each side)."""
        xb = np.asarray([[s[k] for k in FEATURES] for s in blue_states], dtype=np.float32)
        xr = np.asarray([[s[k] for k in FEATURES] for s in red_states], dtype=np.float32)
        return (self.predict(xb) + 1 - self.predict(xr)) / 2


def train(X: np.ndarray, y: np.ndarray, games: np.ndarray, minutes: np.ndarray, log=print,
          layers: tuple[int, ...] = LAYERS) -> tuple[WinModel, dict]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    train_m, cal_m, test_m = _split(games)
    scaler = StandardScaler().fit(X[train_m])
    network = MLPClassifier(hidden_layer_sizes=layers, activation="relu", alpha=1e-4,
                            batch_size=1024, learning_rate_init=1e-3, max_iter=60,
                            early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                            random_state=7)
    started = time.monotonic()
    network.fit(scaler.transform(X[train_m]), y[train_m])
    fit_s = time.monotonic() - started
    log(f"  network trained in {fit_s:.0f}s ({network.n_iter_} epochs)")
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        network.predict_proba(scaler.transform(X[cal_m]))[:, 1], y[cal_m])
    model = WinModel(scaler, network, calibrator)

    # Baseline: only the gold lead and the minute.
    gi, mi = FEATURES.index("gold_diff"), FEATURES.index("minute")

    def base_x(A):
        return np.column_stack([A[:, gi], A[:, mi], A[:, gi] * A[:, mi]])
    baseline = LogisticRegression(max_iter=1000).fit(base_x(X[train_m]), y[train_m])

    yt, xt, mt = y[test_m], X[test_m], minutes[test_m]
    p_net, p_base = model.predict(xt), baseline.predict_proba(base_x(xt))[:, 1]

    def scores(p, mask=slice(None)):
        return {"log_loss": round(float(log_loss(yt[mask], p[mask], labels=[0, 1])), 4),
                "brier": round(float(brier_score_loss(yt[mask], p[mask])), 4),
                "auc": round(float(roc_auc_score(yt[mask], p[mask])), 4) if len(set(yt[mask])) > 1 else None}
    by_minute = {}
    for lo, hi in ((0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 99)):
        m = (mt >= lo) & (mt < hi)
        if m.sum() > 200:
            by_minute[f"{lo}-{hi if hi < 99 else ''}"] = {"rows": int(m.sum()), "network": scores(p_net, m)["auc"],
                                                           "baseline": scores(p_base, m)["auc"]}
    metrics = {
        "version": VERSION, "features": list(FEATURES),
        "games": int(len(np.unique(games))), "rows": int(len(y)),
        "test_games": int(len(np.unique(games[test_m]))), "test_rows": int(test_m.sum()),
        "network": {**scores(p_net), "ece": round(_ece(p_net, yt), 4), "epochs": int(network.n_iter_),
                    "fit_seconds": round(fit_s, 1), "layers": list(layers)},
        "baseline": {**scores(p_base), "ece": round(_ece(p_base, yt), 4), "inputs": "gold lead, minute"},
        "auc_by_minute": by_minute,
        "calibration": _calibration(p_net, yt),
        "trained_at": time.strftime("%Y-%m-%d"),
    }
    return model, metrics


def save(model: WinModel, metrics: dict, models_dir: Path | str) -> Path:
    import joblib

    out = Path(models_dir) / DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / MODEL_FILE)
    (out / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out


_loaded: dict[str, Any] = {}


def load(models_dir: Path | str) -> WinModel | None:
    """The trained model, or None when it hasn't been trained (cached per process)."""
    path = Path(models_dir) / DIRNAME / MODEL_FILE
    key = str(path)
    if key not in _loaded:
        if not path.exists():
            return None
        import joblib

        model = joblib.load(path)
        model.stamp = f"{path.stat().st_mtime_ns // 1_000_000_000:x}"   # changes on every retrain
        _loaded[key] = model
    return _loaded[key]


def metrics(models_dir: Path | str) -> dict | None:
    path = Path(models_dir) / DIRNAME / METRICS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
