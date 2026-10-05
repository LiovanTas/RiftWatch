"""Train and evaluate the high-elo models, one set per role.

* **Decision model** -- given a situation, how likely a high-elo player in that role is to
  make each decision (a jungler ganking top, a mid laner roaming bot, a support warding...).
  Gradient-boosted trees on the situation features plus the champion.
* **Outcome models** -- given a situation *and* a decision, what tends to follow within
  three minutes: the team taking an objective, the player dying, the team's gold swing, and
  for laners how their CS gap to the lane opponent moves. Asking these "what if" for each
  decision is how the coach compares options.

Evaluation splits by game, never by row: minutes of one game are highly correlated, and a
model that trained on minute 6 of a game would look better than it is on minute 7. The
decision model is compared with two baselines -- always guessing the most common decision,
and guessing the most common decision for that minute -- since accuracy alone means little
when farming is half of all minutes.

The outcome models learn associations in high-elo play, not causes: a jungler who ganks bot
in a winning situation was already likely to profit. Their answers are phrased that way.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DECISION = "decision"
OUTCOMES = {
    "team_objective": "classifier",
    "player_died": "classifier",
    "gold_swing": "regressor",
    "lane_cs_swing": "regressor",     # laners only; skipped where the column is absent
}


@dataclass
class Trained:
    role: str
    decision_model: Any
    outcome_models: dict[str, Any]
    features: list[str]
    decisions: list[str]
    champions: list[int]
    metrics: dict[str, Any] = field(default_factory=dict)


def _features(df, exclude: tuple[str, ...] = ()) -> list[str]:
    """Situation feature columns, minus any whose name (after ``f_``) starts with one of
    ``exclude`` -- used to measure what a group of features is worth."""
    return [c for c in df.columns
            if c.startswith("f_") and not any(c[2:].startswith(x) for x in exclude)]


def _matrix(df, features: list[str], champions: list[int], decisions: list[str] | None = None):
    """Feature matrix: situation features, champion as a category index, and (for the
    outcome models) the decision as a category index."""
    champ_index = {c: i for i, c in enumerate(champions)}
    cols = [df[features].to_numpy(dtype=float),
            df["champion_id"].map(lambda c: champ_index.get(c, -1)).to_numpy(dtype=float)[:, None]]
    if decisions is not None:
        dec_index = {d: i for i, d in enumerate(decisions)}
        cols.append(df[DECISION].map(dec_index).to_numpy(dtype=float)[:, None])
    x = np.hstack(cols)
    x[x == -1] = np.nan     # unseen champion: let the trees route it as missing
    return x


def _categorical_mask(n_features: int, extra: int) -> list[bool]:
    return [False] * n_features + [True] * extra


def train(dataset: Path, *, test_fraction: float = 0.2, seed: int = 7,
          exclude: tuple[str, ...] = ()) -> Trained:
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_parquet(dataset)
    role = str(df["role"].iloc[0]) if "role" in df.columns and len(df) else dataset.stem
    features = _features(df, exclude)
    decisions = sorted(df[DECISION].unique())
    champions = sorted(df["champion_id"].unique().tolist())
    if len(champions) > 250:
        # HistGradientBoosting caps categories at 255; keep the most common champions.
        champions = df["champion_id"].value_counts().index[:250].tolist()

    split = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(split.split(df, groups=df["match_id"]))
    tr, te = df.iloc[train_idx], df.iloc[test_idx]

    # -- decision model ------------------------------------------------------------------
    x_tr, x_te = _matrix(tr, features, champions), _matrix(te, features, champions)
    y_tr, y_te = tr[DECISION].to_numpy(), te[DECISION].to_numpy()
    started = time.perf_counter()
    decision_model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
        categorical_features=_categorical_mask(len(features), 1),
        early_stopping=True, validation_fraction=0.1, random_state=seed,
    ).fit(x_tr, y_tr)
    fit_s = time.perf_counter() - started

    proba = decision_model.predict_proba(x_te)
    classes = list(decision_model.classes_)
    top = np.argsort(-proba, axis=1)
    y_idx = np.array([classes.index(y) for y in y_te])
    accuracy = float(np.mean(top[:, 0] == y_idx))
    top2 = float(np.mean((top[:, 0] == y_idx) | (top[:, 1] == y_idx)))

    majority = tr[DECISION].value_counts().idxmax()
    by_minute = tr.groupby("f_minute")[DECISION].agg(lambda s: s.value_counts().idxmax())
    minute_guess = te["f_minute"].map(by_minute).fillna(majority).to_numpy()
    per_class = {}
    for i, c in enumerate(classes):
        pred_c = top[:, 0] == i
        true_c = y_idx == i
        per_class[c] = {
            "share": round(float(true_c.mean()), 3),
            "precision": round(float((pred_c & true_c).sum() / max(pred_c.sum(), 1)), 3),
            "recall": round(float((pred_c & true_c).sum() / max(true_c.sum(), 1)), 3),
        }
    metrics: dict[str, Any] = {
        "games": int(df["match_id"].nunique()), "rows": int(len(df)),
        "train_games": int(tr["match_id"].nunique()), "test_games": int(te["match_id"].nunique()),
        "decision": {
            "accuracy": round(accuracy, 3), "top2_accuracy": round(top2, 3),
            "log_loss": round(float(log_loss(y_te, proba, labels=classes)), 3),
            "baseline_majority_accuracy": round(float(np.mean(y_te == majority)), 3),
            "baseline_by_minute_accuracy": round(float(np.mean(y_te == minute_guess)), 3),
            "per_class": per_class, "fit_seconds": round(fit_s, 1),
        },
        "outcomes": {},
    }

    # -- outcome models ------------------------------------------------------------------
    xo_tr = _matrix(tr, features, champions, decisions)
    xo_te = _matrix(te, features, champions, decisions)
    outcome_models = {}
    for name, kind in OUTCOMES.items():
        if f"o_{name}" not in df.columns:
            continue
        target_tr, target_te = tr[f"o_{name}"].to_numpy(), te[f"o_{name}"].to_numpy()
        common = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                      l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                      categorical_features=_categorical_mask(len(features), 2),
                      random_state=seed)
        if kind == "classifier":
            model = HistGradientBoostingClassifier(**common).fit(xo_tr, target_tr.astype(int))
            p = model.predict_proba(xo_te)[:, 1]
            base_rate = float(target_tr.mean())
            metrics["outcomes"][name] = {
                # AUC needs both outcomes in the test set; a tiny dataset may not have them.
                "auc": (round(float(roc_auc_score(target_te.astype(int), p)), 3)
                        if len(set(target_te.astype(int))) > 1 else None),
                "log_loss": round(float(log_loss(target_te.astype(int), p, labels=[0, 1])), 4),
                "baseline_log_loss": round(float(log_loss(
                    target_te.astype(int), np.full_like(p, base_rate), labels=[0, 1])), 4),
                "base_rate": round(base_rate, 3),
            }
        else:
            model = HistGradientBoostingRegressor(**common).fit(xo_tr, target_tr)
            pred = model.predict(xo_te)
            metrics["outcomes"][name] = {
                "mae": round(float(mean_absolute_error(target_te, pred)), 1),
                "baseline_mae": round(float(mean_absolute_error(
                    target_te, np.full_like(pred, np.median(target_tr)))), 1),
            }
        outcome_models[name] = model

    metrics["role"] = role
    return Trained(role, decision_model, outcome_models, features, decisions, champions, metrics)


def save(trained: Trained, directory: Path) -> Path:
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trained.role}.joblib"
    joblib.dump(trained, path)
    (directory / f"{trained.role}.metrics.json").write_text(json.dumps(trained.metrics, indent=2))
    return path


def load(directory: Path, role: str) -> Trained:
    import joblib

    return joblib.load(directory / f"{role}.joblib")
