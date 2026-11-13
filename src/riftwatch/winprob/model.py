"""Train, evaluate and load the win-probability network.

Data: every stored game with a timeline, at every whole minute, from both teams' sides (the
same moment seen from the other side is the same game with the label flipped, which teaches
the network that the game is symmetric). Label: whether that team won.

Model: a small multilayer perceptron (scikit-learn MLPClassifier, two hidden layers, ReLU,
stopped on held-out games) on standardised inputs, then Platt calibration (a logistic curve on
its log-odds, smooth so small changes still count) on other held-out games, so a "70%" means
70%. Games, not rows, are split into train / calibration / test, so no minute of a
test game was seen in training.

Four versions are trained on the same split; each network is stopped on held-out training
games, and the version kept is the best on every game none of them trained on (calibration and
test games, each half calibrated on the other half), with its improvement over the game-state
network and a 95% interval from resampling games:

  * state -- the 29 inputs of the first model: gold, XP, kills, objectives, who's alive
  * + items and composition -- each team's finished items by what they give, and its
    champions by class and magic damage share
  * + champions -- which ten champions are playing (one input per champion per side, the
    network's first layer acting as a learned embedding), so it can learn that some
    champions win early and others late
  * stacked -- the game-state network with a regularised logistic layer on top for items,
    item-against-team matchups, classes and each champion's strength and scaling. Inputs that
    stay the same all game let a network recognise each game and memorise who won (it does
    worse on new games); a linear layer can't, so it keeps only what holds across games.

All of them are measured against a baseline that only knows the gold lead and the minute
(logistic regression). At prediction time both teams' sides are asked and averaged, so the
two probabilities always add up to one.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from riftwatch.winprob.state import BASE_FEATURES, CLASSES, FULL_FEATURES, ITEM_FLAGS, Catalog, Replay

DIRNAME = "winprob"
MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"
VERSION = "wp2"
LAYERS = (64, 32)
ALPHAS = (1e-4, 1e-3, 1e-2)         # weight penalties tried for each version
MAX_EPOCHS, PATIENCE = 40, 4
FEATURES = BASE_FEATURES            # kept for callers of the first version

# One game: per-minute player frames, the events the model reads, and each player's champion
# and purchases (from participant_builds).
GAME_SQL = """
    SELECT t.match_id,
           jsonb_path_query_array(t.raw, '$.info.frames[*].participantFrames'),
           jsonb_path_query_array(t.raw, '$.info.frames[*].events[*] ? (@.type == "CHAMPION_KILL"
               || @.type == "ELITE_MONSTER_KILL" || @.type == "BUILDING_KILL"
               || @.type == "DRAGON_SOUL_GIVEN" || @.type == "GAME_END")'),
           (SELECT jsonb_agg(jsonb_build_array(b.participant_id, b.champion_id, b.purchases, b.purchase_s))
              FROM participant_builds b WHERE b.match_id = t.match_id)
      FROM match_timelines t"""


def players(rows) -> tuple[dict[int, int], dict[int, list[tuple[int, int]]]]:
    """(participant -> champion, participant -> [(second, item)]) from GAME_SQL's last column."""
    champions, purchases = {}, {}
    for pid, champ, items, seconds in rows or []:
        champions[int(pid)] = int(champ)
        purchases[int(pid)] = list(zip(seconds or [], items or [], strict=False))
    return champions, purchases


def build_catalog(conn, gd, log=print) -> Catalog:
    """Items and champions as the model reads them: every finished item's flags (Data Dragon),
    every champion's class (Data Dragon) and magic damage share (stored games), and the items
    each champion builds most."""
    cat = Catalog()
    for item in gd.items.values():
        if item.is_legendary:
            cat.items[item.id] = Catalog.item_flags(item.tags, item.text, item.name)
            cat.item_names[item.id] = item.name
    started = time.monotonic()
    shares = {c: (m / t if t else 0.4) for c, m, t in conn.execute(
        """SELECT (p->>'championId')::int, sum((p->>'magicDamageDealtToChampions')::bigint),
                  sum((p->>'totalDamageDealtToChampions')::bigint)
             FROM matches m, jsonb_array_elements(m.raw->'info'->'participants') p
            WHERE m.queue_id IN (420, 440) GROUP BY 1""")}
    log(f"  champion damage types from stored games ({time.monotonic() - started:.0f}s)")
    for cid, champ in gd.champions.items():
        cat.champions[cid] = ((champ.tags or ["Fighter"])[0], round(float(shares.get(cid, 0.4)), 3))
    for cid in shares:
        cat.champions.setdefault(cid, ("Fighter", round(float(shares[cid]), 3)))
    cat.index = {cid: i for i, cid in enumerate(sorted(cat.champions))}
    counts: dict[int, Counter] = defaultdict(Counter)
    for champ, items in conn.execute("SELECT champion_id, purchases FROM participant_builds "
                                     "WHERE queue_id IN (420, 440) AND NOT remake"):
        counts[champ].update({i for i in items or [] if i in cat.items})
    cat.popular = {c: [i for i, _ in n.most_common(8)] for c, n in counts.items()}
    return cat


def load_dataset(conn, catalog: Catalog, limit: int | None = None, log=print) -> dict[str, np.ndarray]:
    """Every stored game with a timeline at every minute from both sides: inputs (all of
    FULL_FEATURES), labels, game index, minute and each side's champions."""
    sql = GAME_SQL + """ JOIN matches m USING (match_id)
         WHERE m.queue_id IN (420, 440) ORDER BY match_id""" + (f" LIMIT {int(limit)}" if limit else "")
    xs, ys, games, minutes, own, enemy = [], [], [], [], [], []
    started = time.monotonic()
    with conn.transaction(), conn.cursor(name="winprob_dataset") as cur:   # server-side, streamed
        cur.itersize = 200
        cur.execute(sql)
        for g, (_, frames, events, rows) in enumerate(cur):
            champions, purchases = players(rows)
            replay = Replay(frames, events, champions, purchases, catalog)
            if replay.winner not in (100, 200) or len(frames) < 6:
                continue
            for m in range(1, len(frames)):               # every whole minute, both sides
                for team, state in zip((100, 200), replay.pair(m * 60_000), strict=True):
                    xs.append([state[k] for k in FULL_FEATURES])
                    ys.append(float(replay.winner == team))
                    games.append(g)
                    minutes.append(state["minute"])
                    own.append(state["_own"])
                    enemy.append(state["_enemy"])
            if g and g % 1000 == 0:
                log(f"  {g:,} games read ({time.monotonic() - started:.0f}s)")
    return {"X": np.asarray(xs, dtype=np.float32), "y": np.asarray(ys, dtype=np.float32),
            "games": np.asarray(games), "minutes": np.asarray(minutes, dtype=np.float32),
            "own": np.asarray(own, dtype=np.int16), "enemy": np.asarray(enemy, dtype=np.int16)}


def _fit(design_train, y_train, design_val, y_val, layers, alpha, log=print):
    """Train a network one epoch at a time and keep the epoch with the lowest log loss on
    validation *games*. scikit-learn's own early stopping holds out random rows, which come
    from the same games as the training rows: a network that memorises each game's outcome
    from its champions and items looks fine there and fails on new games."""
    import copy

    from sklearn.metrics import log_loss
    from sklearn.neural_network import MLPClassifier

    net = MLPClassifier(hidden_layer_sizes=layers, activation="relu", alpha=alpha, batch_size=1024,
                        learning_rate_init=1e-3, random_state=7)
    best, best_loss, bad, epochs = None, float("inf"), 0, 0
    for epoch in range(MAX_EPOCHS):
        net.partial_fit(design_train, y_train, classes=np.array([0.0, 1.0]))
        loss = float(log_loss(y_val, net.predict_proba(design_val)[:, 1], labels=[0, 1]))
        if loss < best_loss - 1e-4:
            best, best_loss, bad, epochs = copy.deepcopy(net), loss, 0, epoch + 1
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best, best_loss, epochs


class Platt:
    """Calibration by a logistic curve on the network's log-odds: smooth and always
    increasing. Isotonic regression calibrates as well on average but is a staircase, so a
    small change in the state (a finished item, a ward of gold) can land on a flat step and
    move the chance by exactly nothing -- and Swing is made of small changes."""

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Platt:
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw).reshape(-1, 1), y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(np.asarray(raw)).reshape(-1, 1))[:, 1]


def _held_out_losses(raw: np.ndarray, y: np.ndarray, games: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-game log loss (summed over its moments) and moment counts on the held-out games,
    with each half of the games calibrated on the other half: how good a version is once
    calibrated, on games neither its training nor its calibration saw."""
    ids, inv = np.unique(games, return_inverse=True)
    fold = (inv % 2).astype(bool)
    out = np.empty_like(raw)
    for a, b in ((fold, ~fold), (~fold, fold)):
        out[b] = Platt().fit(raw[a], y[a]).predict(raw[b])
    p = np.clip(out, 0.001, 0.999)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return np.bincount(inv, loss), np.bincount(inv)


def _paired(base: tuple, other: tuple, seed: int = 0) -> dict:
    """How much ``other`` beats ``base`` in log loss per moment, with a 95% interval from
    resampling games."""
    diff, n = base[0] - other[0], base[1]
    rng = np.random.default_rng(seed)
    boots = [diff[i].sum() / n[i].sum() for i in (rng.integers(0, len(diff), len(diff)) for _ in range(1000))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"improvement": round(float(diff.sum() / n.sum()), 4), "low": round(float(lo), 4), "high": round(float(hi), 4)}


def _split(games: np.ndarray, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = np.unique(games)
    rng.shuffle(ids)
    n = len(ids)
    train_ids, cal_ids = set(ids[: int(0.7 * n)]), set(ids[int(0.7 * n): int(0.85 * n)])
    train = np.isin(games, list(train_ids))
    cal = np.isin(games, list(cal_ids))
    return train, cal, ~(train | cal)


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


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


def _one_hot(own: np.ndarray, enemy: np.ndarray, index: dict[int, int]):
    """One column per champion on each side: the network's first layer on these is a learned
    embedding of each champion, added up over the team."""
    from scipy import sparse

    n, k = len(own), len(index)
    size = max(max(index, default=0), int(own.max(initial=0)), int(enemy.max(initial=0))) + 1
    lookup = np.full(size, -1)
    for cid, col in index.items():
        lookup[cid] = col
    rows = np.repeat(np.arange(n), own.shape[1] if own.ndim == 2 else 5)
    out = None
    for cols, offset in ((lookup[own].ravel(), 0), (lookup[enemy].ravel(), k)):
        keep = cols >= 0
        part = sparse.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32), (rows[keep], cols[keep] + offset)),
                                 shape=(n, 2 * k))
        out = part if out is None else out + part
    return out


CONTEXT_SCALE = 10.0     # the network's logit is scaled up so the layer's penalty barely touches it
CONTEXT_C = (0.003, 0.03, 0.3)
# Champion columns are scaled down before the layer's penalty, which penalises them harder
# than the item and class terms: there are ~170 of them per side and drafts are noisy.
CHAMPION_SCALES = (0.1, 0.3, 1.0)


def context_names(index: dict[int, int]) -> list[str]:
    """Names of the context layer's inputs, for reading its weights."""
    names = [f"{f}_diff" for f in ITEM_FLAGS]
    names += ["armor vs physical", "magic resist vs magic", "armor pen vs armor", "magic pen vs magic resist",
              "anti-heal vs lifesteal"]
    names += [f"{c.lower()}s_diff" for c in CLASSES] + [f"{c.lower()}s_diff x minute" for c in CLASSES]
    by_col = {col: cid for cid, col in index.items()}
    names += [f"champion {by_col[i]}" for i in range(len(index))]
    names += [f"champion {by_col[i]} x minute" for i in range(len(index))]
    return names


def context_matrix(X: np.ndarray, own: np.ndarray, enemy: np.ndarray, features: tuple[str, ...],
                   index: dict[int, int], logit: np.ndarray, champion_scale: float = 1.0):
    """What the context layer reads beside the network's logit: item differences by what they
    give, item-against-team matchups (armor against physical damage, magic resist against
    magic, penetration against resistances, anti-heal against lifesteal), class differences
    and how they change with the minute, and each champion's own strength and scaling (+1 on
    one's own side, -1 on the enemy's, so the layer is the same seen from either side)."""
    from scipy import sparse

    col = {f: i for i, f in enumerate(features)}

    def f(name):
        return X[:, col[name]].astype(np.float64)
    minute = f("minute") / 30
    parts = [f(f"{flag}_own") - f(f"{flag}_enemy") for flag in ITEM_FLAGS]
    ms_own, ms_enemy = f("magic_share_own"), f("magic_share_enemy")
    parts += [f("armor_own") * (1 - ms_enemy) - f("armor_enemy") * (1 - ms_own),
              f("mr_own") * ms_enemy - f("mr_enemy") * ms_own,
              f("armor_pen_own") * f("armor_enemy") - f("armor_pen_enemy") * f("armor_own"),
              f("magic_pen_own") * f("mr_enemy") - f("magic_pen_enemy") * f("mr_own"),
              f("anti_heal_own") * f("sustain_enemy") - f("anti_heal_enemy") * f("sustain_own")]
    classes = [f(f"{c.lower()}s_own") - f(f"{c.lower()}s_enemy") for c in CLASSES]
    parts += classes + [c * minute for c in classes]
    dense = sparse.csr_matrix(np.column_stack([CONTEXT_SCALE * logit, *parts]))
    hot = _one_hot(own, enemy, index)
    k = len(index)
    diff = (hot[:, :k] - hot[:, k:]) * champion_scale
    return sparse.hstack([dense, diff, diff.multiply(minute[:, None])], format="csr")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class WinModel:
    """The fitted network with its scaler, calibration, the inputs it reads and (for the
    item- and champion-aware versions) the catalog it reads them with."""

    def __init__(self, scaler, network, calibrator, features=BASE_FEATURES, catalog: Catalog | None = None,
                 identity: bool = False) -> None:
        self.scaler, self.network, self.calibrator = scaler, network, calibrator
        self.features, self.catalog, self.identity = tuple(features), catalog, identity
        self.version = VERSION

    # Models saved before these attributes existed read only the first 29 inputs.
    def _features(self) -> tuple[str, ...]:
        return getattr(self, "features", BASE_FEATURES)

    @property
    def reads_items(self) -> bool:
        return getattr(self, "catalog", None) is not None and "items_own" in self._features()

    def design(self, X: np.ndarray, own: np.ndarray | None = None, enemy: np.ndarray | None = None):
        """Network inputs: the scaled features, then (with champions) one column per champion
        on each side."""
        Z = self.scaler.transform(X)
        if not getattr(self, "identity", False):
            return Z
        from scipy import sparse

        return sparse.hstack([sparse.csr_matrix(Z), _one_hot(own, enemy, self.catalog.index)], format="csr")

    def raw(self, X: np.ndarray, own=None, enemy=None) -> np.ndarray:
        layer = getattr(self, "context", None)
        if layer is None:
            return self.network.predict_proba(self.design(X, own, enemy))[:, 1]
        # The game-state network on its own inputs, then the context layer on top.
        base = [self.features.index(f) for f in BASE_FEATURES]
        p = self.network.predict_proba(self.scaler.transform(X[:, base]))[:, 1]
        Z = context_matrix(X, own, enemy, self.features, self.catalog.index, _logit(p),
                           getattr(self, "champion_scale", 1.0))
        return layer.predict_proba(Z)[:, 1]

    def predict(self, X: np.ndarray, own=None, enemy=None) -> np.ndarray:
        return np.clip(self.calibrator.predict(self.raw(X, own, enemy)), 0.001, 0.999)

    def _arrays(self, states: list[dict]):
        feats = self._features()
        X = np.asarray([[s.get(k, 0.0) for k in feats] for s in states], dtype=np.float32)
        if not (getattr(self, "identity", False) or getattr(self, "context", None) is not None):
            return X, None, None
        own = np.asarray([s.get("_own") or [0] * 5 for s in states], dtype=np.int16)
        enemy = np.asarray([s.get("_enemy") or [0] * 5 for s in states], dtype=np.int16)
        return X, own, enemy

    def blue(self, blue_states: list[dict], red_states: list[dict]) -> np.ndarray:
        """Blue's win probability for paired states (the same moments from each side)."""
        if not blue_states:
            return np.zeros(0)
        return (self.predict(*self._arrays(blue_states)) + 1 - self.predict(*self._arrays(red_states))) / 2


VARIANTS = (("state", BASE_FEATURES, False), ("items", FULL_FEATURES, False), ("champions", FULL_FEATURES, True),
            ("stacked", FULL_FEATURES, True))
VARIANT_NAMES = {"state": "Game state (gold, kills, objectives, who's alive)",
                 "items": "+ items and team composition, in the network",
                 "champions": "+ which champions are playing, in the network",
                 "stacked": "Game state network + a logistic layer for items, composition and champions"}


def train(data: dict[str, np.ndarray], catalog: Catalog | None = None, log=print,
          layers: tuple[int, ...] = LAYERS, variants=VARIANTS) -> tuple[WinModel, dict]:
    """Train each version on the same split, keep the best on the calibration games, and
    report every version and the baseline on the test games."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X_all, y, games, minutes = data["X"], data["y"], data["games"], data["minutes"]
    own, enemy = data.get("own"), data.get("enemy")
    full = X_all.shape[1] == len(FULL_FEATURES)
    train_m, cal_m, test_m = _split(games)
    # A tenth of the training games, held out to stop each network and pick its penalty.
    rng = np.random.default_rng(11)
    train_ids = np.unique(games[train_m])
    val_ids = rng.choice(train_ids, max(1, len(train_ids) // 10), replace=False)
    val_m = train_m & np.isin(games, val_ids)
    fit_m = train_m & ~val_m
    yt, mt = y[test_m], minutes[test_m]

    def scores(p, mask=slice(None)):
        return {"log_loss": round(float(log_loss(yt[mask], p[mask], labels=[0, 1])), 4),
                "brier": round(float(brier_score_loss(yt[mask], p[mask])), 4),
                "auc": round(float(roc_auc_score(yt[mask], p[mask])), 4) if len(set(yt[mask])) > 1 else None}

    spans = ((0, 2), (0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 99))

    def span_mask(lo, hi):
        return (mt >= lo) & (mt < hi)

    held_m = cal_m | test_m                 # every game no version trained on
    results, models, held = [], {}, {}
    for name, feats, identity in variants:
        if (feats is not BASE_FEATURES and not full) or (identity and (catalog is None or own is None)):
            continue
        if name == "stacked":
            if "state" not in models:
                continue
            started = time.monotonic()
            model, layer_c, val_loss = _stack(models["state"][0], X_all, own, enemy, y, fit_m, val_m, catalog, log)
            raw_cal = model.raw(X_all[cal_m], own[cal_m], enemy[cal_m])
            model.calibrator = Platt().fit(raw_cal, y[cal_m])
            held[name] = _held_out_losses(model.raw(X_all[held_m], own[held_m], enemy[held_m]), y[held_m], games[held_m])
            p_cal = model.predict(X_all[cal_m], own[cal_m], enemy[cal_m])
            p_test = model.predict(X_all[test_m], own[test_m], enemy[test_m])
            fit_s = time.monotonic() - started
            res = {"name": name, "label": VARIANT_NAMES[name],
                   "inputs": len(BASE_FEATURES) + model.context.coef_.shape[1] - 1,
                   **scores(p_test), "ece": round(_ece(p_test, yt), 4),
                   "calibration_games_log_loss": round(float(log_loss(y[cal_m], p_cal, labels=[0, 1])), 4),
                   "epochs": models["state"][2], "alpha": layer_c, "validation_games_log_loss": round(val_loss, 4),
                   "selection_log_loss": round(float(held[name][0].sum() / held[name][1].sum()), 4),
                   "fit_seconds": round(fit_s, 1),
                   "by_minute": {f"{lo}-{hi if hi < 99 else ''}": scores(p_test, span_mask(lo, hi))["auc"]
                                 for lo, hi in spans if span_mask(lo, hi).sum() > 200}}
            log(f"  {name:<10} log loss {res['log_loss']:.4f}  AUC {res['auc']:.3f}  ({fit_s:.0f}s, {layer_c})")
            results.append(res)
            models[name] = (model, p_test, models["state"][2])
            continue
        X = X_all[:, [FULL_FEATURES.index(f) for f in feats]] if full else X_all
        scaler = StandardScaler().fit(X[fit_m])
        model = WinModel(scaler, None, None, feats, catalog if feats is not BASE_FEATURES else None, identity)

        def sub(m):
            return X[m], (own[m] if own is not None else None), (enemy[m] if enemy is not None else None)
        started = time.monotonic()
        d_fit, d_val = model.design(*sub(fit_m)), model.design(*sub(val_m))
        tried = []
        for alpha in ALPHAS:
            net, val_loss, epochs = _fit(d_fit, y[fit_m], d_val, y[val_m], layers, alpha)
            tried.append((val_loss, alpha, epochs, net))
            log(f"    {name}: penalty {alpha:g} -> {val_loss:.4f} on validation games ({epochs} epochs)")
        val_loss, alpha, epochs, network = min(tried, key=lambda t: t[0])
        model.network = network
        fit_s = time.monotonic() - started
        raw_cal = model.raw(*sub(cal_m))
        model.calibrator = Platt().fit(raw_cal, y[cal_m])
        held[name] = _held_out_losses(model.raw(*sub(held_m)), y[held_m], games[held_m])
        p_cal, p_test = model.predict(*sub(cal_m)), model.predict(*sub(test_m))
        res = {"name": name, "label": VARIANT_NAMES[name],
               "inputs": len(feats) + (2 * len(catalog.index) if identity else 0),
               **scores(p_test), "ece": round(_ece(p_test, yt), 4),
               "calibration_games_log_loss": round(float(log_loss(y[cal_m], p_cal, labels=[0, 1])), 4),
               "epochs": epochs, "alpha": alpha, "validation_games_log_loss": round(val_loss, 4),
               "selection_log_loss": round(float(held[name][0].sum() / held[name][1].sum()), 4),
               "fit_seconds": round(fit_s, 1),
               "by_minute": {f"{lo}-{hi if hi < 99 else ''}": scores(p_test, span_mask(lo, hi))["auc"]
                             for lo, hi in spans if span_mask(lo, hi).sum() > 200}}
        log(f"  {name:<10} log loss {res['log_loss']:.4f}  AUC {res['auc']:.3f}  ({fit_s:.0f}s, penalty {alpha:g})")
        results.append(res)
        models[name] = (model, p_test, epochs)

    # Baseline: only the gold lead and the minute.
    gi, mi = BASE_FEATURES.index("gold_diff"), BASE_FEATURES.index("minute")

    def base_x(A):
        return np.column_stack([A[:, gi], A[:, mi], A[:, gi] * A[:, mi]])
    baseline = LogisticRegression(max_iter=1000).fit(base_x(X_all[fit_m]), y[fit_m])
    p_base = baseline.predict_proba(base_x(X_all[test_m]))[:, 1]

    if "state" in held:
        for r in results:
            if r["name"] != "state":
                r["vs_state"] = _paired(held["state"], held[r["name"]])
    best = min(results, key=lambda r: r["selection_log_loss"])
    model, p_net, _ = models[best["name"]]
    if getattr(model, "context", None) is not None:
        weights = dict(zip(["network logit"] + context_names(catalog.index), model.context.coef_[0], strict=False))
        metrics_context = {k: round(float(v), 4) for k, v in weights.items() if not k.startswith("champion ")}
    else:
        metrics_context = None
    metrics = {
        "version": VERSION, "chosen": best["name"], "features": list(model.features),
        "champion_inputs": 2 * len(catalog.index) if model.identity else 0,
        "items_known": len(catalog.items) if catalog else 0,
        "games": int(len(np.unique(games))), "rows": int(len(y)),
        "test_games": int(len(np.unique(games[test_m]))), "test_rows": int(test_m.sum()),
        "network": {**scores(p_net), "ece": round(_ece(p_net, yt), 4), "epochs": best["epochs"],
                    "fit_seconds": best["fit_seconds"], "layers": list(layers)},
        "baseline": {**scores(p_base), "ece": round(_ece(p_base, yt), 4), "inputs": "gold lead, minute"},
        "variants": results, "context_weights": metrics_context,
        "auc_by_minute": {f"{lo}-{hi if hi < 99 else ''}": {
            "rows": int(span_mask(lo, hi).sum()), "network": scores(p_net, span_mask(lo, hi))["auc"],
            "baseline": scores(p_base, span_mask(lo, hi))["auc"]}
            for lo, hi in spans[1:] if span_mask(lo, hi).sum() > 200},
        "calibration": _calibration(p_net, yt),
        "trained_at": time.strftime("%Y-%m-%d"),
    }
    return model, metrics


def _stack(state: WinModel, X_all, own, enemy, y, fit_m, val_m, catalog: Catalog, log=print):
    """The game-state network with a logistic layer on top for items, composition and
    champions. A linear layer can't memorise games by their line-up the way a network can, so
    it learns only what holds across games; its penalty is chosen on the validation games."""
    import copy

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    base = [FULL_FEATURES.index(f) for f in BASE_FEATURES]

    logits = {}

    def Z(m, scale):
        key = id(m)
        if key not in logits:
            logits[key] = _logit(state.network.predict_proba(state.scaler.transform(X_all[m][:, base]))[:, 1])
        return context_matrix(X_all[m], own[m], enemy[m], FULL_FEATURES, catalog.index, logits[key], scale)
    tried = []
    for scale in CHAMPION_SCALES:
        z_fit, z_val = Z(fit_m, scale), Z(val_m, scale)
        for c in CONTEXT_C:
            layer = LogisticRegression(C=c, max_iter=500, solver="lbfgs").fit(z_fit, y[fit_m])
            loss = float(log_loss(y[val_m], layer.predict_proba(z_val)[:, 1], labels=[0, 1]))
            log(f"    stacked: C {c:g}, champions x{scale:g} -> {loss:.4f} on validation games")
            tried.append((loss, c, scale, layer))
    loss, c, scale, layer = min(tried, key=lambda t: t[0])
    model = copy.copy(state)
    model.features, model.catalog, model.identity, model.context = FULL_FEATURES, catalog, False, layer
    model.champion_scale = scale
    return model, f"C {c:g}, champions x{scale:g}", loss


def save(model: WinModel, metrics: dict, models_dir: Path | str) -> Path:
    import joblib

    out = Path(models_dir) / DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / MODEL_FILE)
    (out / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _loaded.pop(str(out / MODEL_FILE), None)
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
