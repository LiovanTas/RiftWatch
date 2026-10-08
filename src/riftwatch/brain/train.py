"""Training the brain.

For every head, in both feature variants, pooled over roles and for each role with enough
games of its own:

1. **Grouped cross-validation.** Games are split into folds; each fold's games are predicted
   by models that never saw them, so every situation gets an out-of-fold prediction. Moments
   of one game are highly correlated (four readings a second), so splitting by row would
   flatter the models badly. Every game weighs the same, however long its laning was.
2. **Model selection.** Each candidate -- gradient-boosted trees at a few settings, and a
   small neural network (a multi-layer perceptron) -- is cross-validated; the one with the
   lowest held-out loss is kept. Every candidate's score is recorded, so the card shows how
   trees and the network compare. (With a few dozen games trees usually win: they handle
   missing readings and small data well. A sequence model over the raw readings becomes
   worth trying with a few hundred games.)
3. **The bar to clear.** A head is *usable* only if it beats the plain base rate (the
   training folds' average) on held-out games by at least MIN_GAIN, and does so in most
   folds -- not on average thanks to one lucky fold. Unusable heads stay on the card; the
   coach never quotes them.
4. **Calibration.** Probabilities are mapped through a logistic (Platt) fit of what
   actually happened on the out-of-fold predictions, so "30%" means 30%. Candidates are
   compared, and heads judged, on *cross-fitted* calibrated predictions: each fold calibrated
   by a fit on the other folds only. (A network can rank situations well but come out
   over-confident; judged raw it would lose to the base rate for the wrong reason.)
5. **Bagging.** The final model is an ensemble fitted on games resampled with replacement;
   the spread between its members says how sure the brain is about a given situation.
Folds, candidates, bags and learning-curve points are independent fits, so they run side by
side in worker processes that split the machine's cores.

6. **What it learned.** Permutation importance per feature group (how much worse held-out
   predictions get when a group is scrambled), a learning curve (is more video still
   helping?), and patterns in plain words (``explain``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import psycopg

from riftwatch.brain import data as bdata
from riftwatch.brain import explain, features
from riftwatch.brain.heads import HEADS, PRIMARY
from riftwatch.brain.model import Brain, Platt, Unit, estimator, fit, matrix, raw_predict

MIN_GAIN = 0.01           # beat the base rate's held-out loss by at least 1% ...
FOLD_SHARE = 0.6          # ... in at least this share of folds
TREE_GRID = (
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.3},
    {"learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 80, "l2_regularization": 3.0},
)
NEURAL_GRID = (
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-3},
    {"hidden_layer_sizes": (32,), "alpha": 1e-2},
)
CURVE = (0.25, 0.5, 0.75, 1.0)


@dataclass
class TrainConfig:
    folds: int = 5
    bags: int = 5
    search: bool = True              # try every setting (else the first of each family)
    families: tuple[str, ...] = ("trees", "neural")
    max_iter: int = 300              # boosting rounds
    neural_epochs: int = 40
    neural_rows: int = 20000         # a network trains on at most this many situations
    jobs: int = 0                    # models fitted side by side (0: half the cores)
    seed: int = 0
    learning_curve: bool = True
    importance: bool = True
    min_videos: int = 5
    role_min_videos: int = 10
    min_rows: int = 200
    min_class: int = 20              # binary heads: at least this many of each outcome
    champion_min_videos: int = 3
    min_tier: str | None = None

    @classmethod
    def quick(cls, **kw) -> TrainConfig:
        """Few folds and bags, no search: for trying things out."""
        return cls(**{"folds": 3, "bags": 2, "search": False, "max_iter": 80,
                      "neural_epochs": 15, "learning_curve": False, **kw})


# -- metrics ------------------------------------------------------------------------------

def _logloss(y, p, w) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w))


def _rmse(y, p, w) -> float:
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def _loss(kind: str, y, p, w) -> float:
    return _logloss(y, p, w) if kind == "binary" else _rmse(y, p, w)


def _ece(y, p, w, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error, and the reliability table behind it."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total, err, table = w.sum(), 0.0, []
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        pred, seen = np.average(p[m], weights=w[m]), np.average(y[m], weights=w[m])
        err += w[m].sum() / total * abs(pred - seen)
        table.append({"predicted": round(float(pred), 4), "observed": round(float(seen), 4),
                      "share": round(float(w[m].sum() / total), 4)})
    return float(err), table


def _metrics(kind: str, y, p, base, w, fold_of: np.ndarray) -> dict:
    folds = [f for f in np.unique(fold_of) if f >= 0]
    better = sum(_loss(kind, y[fold_of == f], p[fold_of == f], w[fold_of == f])
                 < _loss(kind, y[fold_of == f], base[fold_of == f], w[fold_of == f])
                 for f in folds)
    m: dict = {"folds": len(folds), "folds_better": int(better)}
    if kind == "binary":
        from sklearn.metrics import roc_auc_score

        m["log_loss"] = round(_logloss(y, p, w), 5)
        m["base_log_loss"] = round(_logloss(y, base, w), 5)
        m["brier"] = round(float(np.average((p - y) ** 2, weights=w)), 5)
        m["base_brier"] = round(float(np.average((base - y) ** 2, weights=w)), 5)
        if len(np.unique(y)) > 1:
            m["auc"] = round(float(roc_auc_score(y, p, sample_weight=w)), 4)
        ece, table = _ece(y, p, w)
        m["ece"] = round(ece, 4)
        m["reliability"] = table
        m["rate"] = round(float(np.average(y, weights=w)), 5)
        m["gain"] = round(1 - m["log_loss"] / m["base_log_loss"], 4) if m["base_log_loss"] else 0.0
    else:
        m["rmse"] = round(_rmse(y, p, w), 5)
        m["base_rmse"] = round(_rmse(y, base, w), 5)
        m["mae"] = round(float(np.average(np.abs(y - p), weights=w)), 5)
        m["base_mae"] = round(float(np.average(np.abs(y - base), weights=w)), 5)
        m["r2"] = round(1 - (m["rmse"] / m["base_rmse"]) ** 2, 4) if m["base_rmse"] else 0.0
        m["mean"] = round(float(np.average(y, weights=w)), 5)
        m["gain"] = round(1 - m["rmse"] / m["base_rmse"], 4) if m["base_rmse"] else 0.0
    m["usable"] = bool(m["gain"] >= MIN_GAIN and len(folds)
                       and better >= np.ceil(FOLD_SHARE * len(folds)))
    return m


# -- the pieces ---------------------------------------------------------------------------

def _weights(videos: np.ndarray) -> np.ndarray:
    """Every game weighs the same; the weights average 1."""
    _, inverse, counts = np.unique(videos, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w * len(w) / w.sum()


def _fold_of(videos: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    out = np.full(len(videos), -1)
    for i, f in enumerate(folds):
        out[np.isin(videos, f)] = i
    return out


@dataclass
class _Setup:
    head: str
    kind: str
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    videos: np.ndarray
    fold_of: np.ndarray
    columns: list[str]
    categories: dict[str, list[str]]


def _candidates(cfg: TrainConfig) -> list[tuple[str, dict]]:
    out = []
    for family in cfg.families:
        grid = TREE_GRID if family == "trees" else NEURAL_GRID
        out += [(family, p) for p in (grid if cfg.search else grid[:1])]
    return out


def _spec(family: str, params: dict, s: _Setup, cfg: TrainConfig, seed: int | None = None) -> tuple:
    """Everything ``model.estimator`` needs, as plain values a worker process can receive."""
    params = dict(params)
    if family == "neural":
        params["max_iter"] = cfg.neural_epochs
    return (family, s.kind, params, len(s.categories) if family == "trees" else 0,
            len(s.columns), cfg.seed if seed is None else seed, cfg.max_iter)


def _job(spec: tuple, X, y, w, X_test, threads: int | None):
    """Fit one model (and predict ``X_test``), in a worker process."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads):
        model = fit(estimator(*spec), spec[0], X, y, w)
        pred = None if X_test is None else raw_predict(model, spec[1], X_test)
    return model, pred


def _run(tasks: list[tuple], cfg: TrainConfig) -> list[tuple]:
    """Fit independent models side by side, splitting the cores between them."""
    cpus = os.cpu_count() or 1
    # The same number of workers every time: the pool is reused, not restarted (starting
    # worker processes costs about a second each on Windows).
    jobs = cfg.jobs or max(1, cpus // 2)
    if jobs <= 1 or len(tasks) <= 1:
        return [_job(*t, None) for t in tasks]
    from joblib import Parallel, delayed

    threads = max(1, cpus // jobs)
    return Parallel(n_jobs=jobs)(delayed(_job)(*t, threads) for t in tasks)


def _inputs(family: str, s: _Setup, rows: np.ndarray) -> np.ndarray:
    """The network gets the numeric inputs only (champion codes are labels, not amounts)."""
    X = s.X[rows]
    return X if family == "trees" else X[:, :len(s.columns)]


def _fit_rows(family: str, rows: np.ndarray, cfg: TrainConfig, rng) -> np.ndarray:
    idx = np.flatnonzero(rows)
    if family == "neural" and len(idx) > cfg.neural_rows:
        idx = np.sort(rng.choice(idx, cfg.neural_rows, replace=False))
    return idx


def _base(kind: str, y, w) -> float:
    return float(np.average(y, weights=w))


def _cv(s: _Setup, candidates: list[tuple[str, dict]], cfg: TrainConfig):
    """Every candidate's out-of-fold predictions and fold models, and per-row base
    predictions (the training folds' average)."""
    rng = np.random.default_rng(cfg.seed)
    base = np.full(len(s.y), np.nan)
    folds = []
    for f in np.unique(s.fold_of[s.fold_of >= 0]):
        test = s.fold_of == f
        train = ~test
        if len(np.unique(s.videos[train])) < 2:
            continue
        base[test] = _base(s.kind, s.y[train], s.w[train])
        folds.append((test, train))
    oofs = [np.full(len(s.y), np.nan) for _ in candidates]
    fitted: list[list] = [[] for _ in candidates]
    tasks, slots = [], []
    for c, (family, params) in enumerate(candidates):
        for test, train in folds:
            if s.kind == "binary" and len(np.unique(s.y[train])) < 2:
                oofs[c][test] = base[test]
                continue
            idx = _fit_rows(family, train, cfg, rng)
            tasks.append((_spec(family, params, s, cfg), _inputs(family, s, idx), s.y[idx],
                          s.w[idx], _inputs(family, s, test)))
            slots.append((c, test))
    for (c, test), (model, pred) in zip(slots, _run(tasks, cfg), strict=True):
        oofs[c][test] = pred
        fitted[c].append((test, model))
    return oofs, base, fitted


def _importance(s: _Setup, family: str, fitted, seed: int) -> dict[str, dict]:
    """Held-out loss increase when each feature group is scrambled, averaged over folds."""
    rng = np.random.default_rng(seed)
    names = s.columns + list(s.categories) if family == "trees" else s.columns
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        groups.setdefault(features.group_of(name), []).append(i)
    rise: dict[str, list[tuple[float, float]]] = {g: [] for g in groups}
    for test, model in fitted:
        X = _inputs(family, s, test)
        y, w = s.y[test], s.w[test]
        before = _loss(s.kind, y, raw_predict(model, s.kind, X), w)
        for g, cols in groups.items():
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            after = _loss(s.kind, y, raw_predict(model, s.kind, Xp), w)
            rise[g].append((after - before, w.sum()))
    out = {}
    for g, vals in rise.items():
        if vals:
            d = np.array(vals)
            out[g] = {"loss_increase": round(float(np.average(d[:, 0], weights=d[:, 1])), 6),
                      "folds_hurt": int((d[:, 0] > 0).sum())}
    total = sum(max(0.0, v["loss_increase"]) for v in out.values()) or 1.0
    for v in out.values():
        v["share"] = round(max(0.0, v["loss_increase"]) / total, 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["loss_increase"]))


def _curve(s: _Setup, family: str, params: dict, cfg: TrainConfig) -> dict:
    """Held-out loss when training on a growing share of each fold's training games."""
    rng = np.random.default_rng(cfg.seed + 1)
    tasks, slots = [], []
    for share in CURVE:
        for f in np.unique(s.fold_of[s.fold_of >= 0]):
            test = s.fold_of == f
            pool = np.unique(s.videos[~test])
            if len(pool) < 2:
                continue
            chosen = rng.choice(pool, size=min(max(2, round(share * len(pool))), len(pool)),
                                replace=False)
            train = np.isin(s.videos, chosen) & ~test
            if s.kind == "binary" and len(np.unique(s.y[train])) < 2:
                continue
            idx = _fit_rows(family, train, cfg, rng)
            tasks.append((_spec(family, params, s, cfg), _inputs(family, s, idx), s.y[idx],
                          s.w[idx], _inputs(family, s, test)))
            slots.append((share, test, train, len(chosen)))
    by_share: dict[float, list] = {}
    for (share, test, train, n), (_, p) in zip(slots, _run(tasks, cfg), strict=True):
        base = np.full(test.sum(), _base(s.kind, s.y[train], s.w[train]))
        by_share.setdefault(share, []).append(
            (_loss(s.kind, s.y[test], p, s.w[test]), _loss(s.kind, s.y[test], base, s.w[test]),
             s.w[test].sum(), n))
    points = []
    for share, rows in by_share.items():
        d = np.array(rows)
        points.append({"share": share, "train_games": round(float(d[:, 3].mean()), 1),
                       "loss": round(float(np.average(d[:, 0], weights=d[:, 2])), 5),
                       "base_loss": round(float(np.average(d[:, 1], weights=d[:, 2])), 5)})
    improving = (len(points) >= 2 and points[-1]["loss"] < points[-2]["loss"] * (1 - MIN_GAIN / 2))
    return {"head": s.head, "points": points, "still_improving": bool(improving)}


def _bag(s: _Setup, family: str, params: dict, cfg: TrainConfig) -> list:
    """The final ensemble: models fitted on games resampled with replacement."""
    rng = np.random.default_rng(cfg.seed + 2)
    games = np.unique(s.videos)
    tasks = []
    for b in range(max(1, cfg.bags)):
        w = s.w
        if cfg.bags > 1:
            for _ in range(10):
                draw = rng.choice(games, size=len(games), replace=True)
                counts = dict(zip(*np.unique(draw, return_counts=True), strict=True))
                times = np.array([counts.get(v, 0) for v in s.videos], float)
                if s.kind != "binary" or len(np.unique(s.y[times > 0])) > 1:
                    break
            w = s.w * times
        rows = _fit_rows(family, w > 0, cfg, rng)
        tasks.append((_spec(family, params, s, cfg, seed=cfg.seed + b),
                      _inputs(family, s, rows), s.y[rows], w[rows], None))
    return [model for model, _ in _run(tasks, cfg)]


def _calibrator(y, p, w) -> Platt | None:
    if min((y == 1).sum(), (y == 0).sum()) < 30:
        return None
    return Platt.fit(p, y, w)


def _crossfit(y, p, w, fold_of) -> np.ndarray:
    """Each fold's predictions calibrated by a fit on the other folds' (unchanged where a
    fit isn't possible)."""
    out = p.copy()
    for f in np.unique(fold_of[fold_of >= 0]):
        test, rest = fold_of == f, (fold_of != f) & (fold_of >= 0)
        cal = _calibrator(y[rest], p[rest], w[rest])
        if cal is not None:
            out[test] = cal.predict(p[test])
    return out


def _quantiles(values: np.ndarray, qs=(10, 25, 50, 75, 90, 99)) -> dict[str, float]:
    values = values[~np.isnan(values)]
    if not len(values):
        return {}
    return {f"p{q}": round(float(np.percentile(values, q)), 5) for q in qs}


# -- one unit -----------------------------------------------------------------------------

def _setup(df: pd.DataFrame, head, variant: str, folds: list[np.ndarray],
           cfg: TrainConfig) -> tuple[_Setup | None, str | None]:
    rows = df[df[head.name].notna()]
    videos = rows["video_id"].to_numpy()
    n_videos = len(np.unique(videos))
    if n_videos < cfg.min_videos:
        return None, f"{n_videos} game(s) with this label; needs {cfg.min_videos}"
    if len(rows) < cfg.min_rows:
        return None, f"{len(rows)} situations with this label; needs {cfg.min_rows}"
    y = rows[head.name].to_numpy(float)
    if head.kind == "binary" and min((y == 1).sum(), (y == 0).sum()) < cfg.min_class:
        return None, (f"{int((y == 1).sum())} positive and {int((y == 0).sum())} negative "
                      f"situations; needs {cfg.min_class} of each")
    columns = [c for c in features.VARIANTS[variant] if rows[c].notna().any()
               and rows[c].nunique(dropna=True) > 1]
    categories = {}
    for c in ("champion", "opponent_champion"):
        per = rows.groupby(c)["video_id"].nunique()
        keep = sorted(per[per >= cfg.champion_min_videos].index.tolist())
        if keep:
            categories[c] = keep
    s = _Setup(head.name, head.kind, matrix(rows, columns, categories), y, _weights(videos),
               videos, _fold_of(videos, folds), columns, categories)
    return s, None


def train_unit(df: pd.DataFrame, head, variant: str, segment: str, folds: list[np.ndarray],
               cfg: TrainConfig, curve: bool = False) -> tuple[Unit, dict | None, np.ndarray | None]:
    """Fit and judge one head on ``df``. Returns the unit, its learning curve (if asked) and
    its out-of-fold predictions (for comparing role units with the pooled one)."""
    unit = Unit(head.name, variant, segment)
    s, why = _setup(df, head, variant, folds, cfg)
    if s is None:
        unit.skipped = why
        return unit, None, None
    unit.columns, unit.categories = s.columns, s.categories

    candidates = []
    best = None
    tried = _candidates(cfg)
    oofs, base, fitted_all = _cv(s, tried, cfg)
    for (family, params), raw, fitted in zip(tried, oofs, fitted_all, strict=True):
        ok = ~np.isnan(raw)
        if not ok.any():
            continue
        oof = raw.copy()
        if s.kind == "binary":
            oof[ok] = _crossfit(s.y[ok], raw[ok], s.w[ok], s.fold_of[ok])
        loss = _loss(s.kind, s.y[ok], oof[ok], s.w[ok])
        candidates.append({"family": family, "params": params, "loss": round(loss, 5)})
        if best is None or loss < best[0]:
            best = (loss, family, params, raw, oof, base, fitted)
    if best is None:
        unit.skipped = "no fold could be held out"
        return unit, None, None
    _, unit.family, unit.params, raw, oof, base, fitted = best
    if unit.family == "neural":
        unit.categories = {}                     # the network takes the numeric inputs only
    ok = ~np.isnan(oof)
    fold_of = np.where(ok, s.fold_of, -1)
    metrics = _metrics(s.kind, s.y[ok], oof[ok], base[ok], s.w[ok], fold_of[ok])
    metrics["candidates"] = candidates
    if s.kind == "binary":
        metrics["log_loss_uncalibrated"] = round(_logloss(s.y[ok], raw[ok], s.w[ok]), 5)
    metrics["games"] = int(len(np.unique(s.videos)))
    metrics["situations"] = int(len(s.y))
    if s.kind == "binary":
        unit.calibrator = _calibrator(s.y[ok], raw[ok], s.w[ok])
        cal = unit.calibrator.predict(raw[ok]) if unit.calibrator is not None else raw[ok]
        unit.quantiles = {"all": _quantiles(cal), "positive": _quantiles(cal[s.y[ok] == 1])}
        metrics["calibrated"] = unit.calibrator is not None
    else:
        unit.quantiles = {"all": _quantiles(oof[ok])}
    if cfg.importance:
        metrics["importance"] = _importance(s, unit.family, fitted, cfg.seed)
    unit.usable = metrics.pop("usable")
    unit.metrics = metrics
    unit.models = _bag(s, unit.family, unit.params, cfg)
    unit.medians = {c: float(np.nanmedian(s.X[:, i])) for i, c in enumerate(s.columns)}
    learning = _curve(s, unit.family, unit.params, cfg) if curve else None
    full_oof = np.full(len(df), np.nan)
    full_oof[np.flatnonzero(df[head.name].notna().to_numpy())] = oof
    return unit, learning, full_oof


# -- the whole brain ----------------------------------------------------------------------

def train(conn: psycopg.Connection, cfg: TrainConfig | None = None,
          progress: Callable[[str], None] | None = None, view: str = "spectator") -> Brain:
    """Train every head from the checked videos of ``view``. Raises when there are too few
    games for held-out scores to mean anything."""
    cfg = cfg or TrainConfig()
    say = progress or (lambda _m: None)
    started = time.perf_counter()
    checks = bdata.check(conn, view)
    excluded = [{"video_id": c.video_id, "title": c.title, "problems": c.problems}
                for c in checks if not c.ok]
    df = bdata.load(conn, view=view, min_tier=cfg.min_tier)
    games = sorted(df["video_id"].unique().tolist()) if len(df) else []
    if len(games) < cfg.min_videos:
        raise ValueError(f"{len(games)} usable video(s) with laning situations; need at least "
                         f"{cfg.min_videos} before held-out scores mean anything"
                         + (f" ({len(excluded)} excluded: see riftwatch brain check)"
                            if excluded else ""))
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(np.array(games))
    k = min(cfg.folds, len(games))
    folds = [order[i::k] for i in range(k)]
    say(f"{len(df)} situations from {len(games)} games; {k}-fold cross-validation by game")

    roles = {r: sorted(df.loc[df[f"role_{r}"] == 1, "video_id"].unique().tolist())
             for r in features.ROLES}
    variants = ["full", "basic"]
    if not any(df[c].notna().any() for c in features.VARIANTS["full"]
               if features.BY_NAME[c].needs):
        variants = ["basic"]                     # no video shows the panel: one variant
    units: dict[str, Unit] = {}
    curve = None
    for head in HEADS:
        for variant in variants:
            want_curve = cfg.learning_curve and head.name == PRIMARY and variant == variants[0]
            unit, learning, pooled_oof = train_unit(df, head, variant, "all", folds, cfg,
                                                    curve=want_curve)
            units[Brain.key(head.name, variant, "all")] = unit
            curve = learning or curve
            say(_line(unit))
            if unit.skipped:
                continue
            for role, vids in roles.items():
                if len(vids) < cfg.role_min_videos:
                    continue
                part = df[df[f"role_{role}"] == 1]
                ru, _, role_oof = train_unit(part, head, variant, role, folds, cfg)
                if not ru.skipped and role_oof is not None:
                    labelled = part[head.name].notna().to_numpy()
                    y = part[head.name].to_numpy(float)[labelled]
                    w = _weights(part["video_id"].to_numpy()[labelled])
                    mine = role_oof[labelled]
                    pooled = pooled_oof[part.index.to_numpy()][labelled]
                    ok = ~np.isnan(mine) & ~np.isnan(pooled)
                    if ok.any():
                        a = _loss(head.kind, y[ok], mine[ok], w[ok])
                        b = _loss(head.kind, y[ok], pooled[ok], w[ok])
                        ru.metrics["loss_vs_pooled"] = {"role": round(a, 5), "pooled": round(b, 5)}
                        ru.metrics["beats_pooled"] = bool(a < b)
                units[Brain.key(head.name, variant, role)] = ru
                say(_line(ru))

    brain = Brain(
        version="", created=time.strftime("%Y-%m-%d %H:%M:%S"), config=asdict(cfg),
        data={"view": view, "games": len(games), "situations": int(len(df)),
              "video_ids": [int(v) for v in games],
              "roles": {r: len(v) for r, v in roles.items() if v},
              "champions": int(df["champion"].nunique()),
              "trade_rate": round(float(df["trade"].mean()), 5),
              "analyzer_version": _analyzer_version(),
              "seconds": None},
        units=units, learning_curve=curve, excluded=excluded)
    say("extracting patterns")
    brain.patterns = explain.patterns(brain, df, seed=cfg.seed)
    brain.data["seconds"] = round(time.perf_counter() - started, 1)
    return brain


def _analyzer_version() -> int:
    from riftwatch.vision.library import ANALYZER_VERSION

    return ANALYZER_VERSION


def _line(u: Unit) -> str:
    name = f"{u.head}/{u.variant}/{u.segment}"
    if u.skipped:
        return f"  {name:<28} skipped: {u.skipped}"
    m = u.metrics
    score = (f"log loss {m['log_loss']:.4f} vs base {m['base_log_loss']:.4f}"
             + (f", AUC {m['auc']:.3f}" if "auc" in m else "")
             if "log_loss" in m else f"RMSE {m['rmse']:.4f} vs base {m['base_rmse']:.4f}")
    return (f"  {name:<28} {u.family:<6} {score}; better in {m['folds_better']}/{m['folds']} "
            f"folds -> {'usable' if u.usable else 'not usable'}")
