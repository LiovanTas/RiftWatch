"""A trained brain: one fitted unit per head, feature variant and segment (all roles pooled,
or one role), and how to ask it about situations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from riftwatch.brain import features
from riftwatch.brain.heads import BY_NAME as HEADS

FAMILIES = ("trees", "neural")


def matrix(df: pd.DataFrame, columns: list[str], categories: dict[str, list[str]]) -> np.ndarray:
    """Numeric columns, then each categorical column as a code (NaN: not a known category)."""
    parts = [df[columns].to_numpy(float)] if columns else []
    for name, vocab in categories.items():
        index = {v: i for i, v in enumerate(vocab)}
        parts.append(df[name].map(index).to_numpy(float)[:, None])
    return np.hstack(parts) if parts else np.empty((len(df), 0))


def estimator(family: str, kind: str, params: dict, n_categorical: int, n_columns: int,
              seed: int, max_iter: int):
    if family == "trees":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingClassifier if kind == "binary" else HistGradientBoostingRegressor
        mask = [False] * n_columns + [True] * n_categorical
        return cls(max_iter=max_iter, early_stopping=False, random_state=seed,
                   categorical_features=mask if n_categorical else None, **params)
    if family == "neural":
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        cls = MLPClassifier if kind == "binary" else MLPRegressor
        return Pipeline([
            # Unknown readings become the median plus a "this was unknown" input.
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("net", cls(random_state=seed, learning_rate_init=1e-3, batch_size=256, **params)),
        ])
    raise ValueError(f"unknown model family {family!r}")


def fit(model, family: str, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "trees":
            X = X.copy()
            X[:, np.isnan(X).all(axis=0)] = 0.0     # a column never seen: constant, unused
            model.fit(X, y, sample_weight=w)
        else:
            model.fit(X, y, net__sample_weight=w)
    return model


def raw_predict(model, kind: str, X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "binary":
            return model.predict_proba(X)[:, 1]
        return model.predict(X)


class Platt:
    """Calibration: a logistic fit of what happened on the logit of the raw prediction.
    Two numbers, so it can't overfit the few situations at the extremes the way a free-form
    (isotonic) mapping does with a small library."""

    def __init__(self, slope: float, intercept: float) -> None:
        self.slope, self.intercept = slope, intercept

    @classmethod
    def fit(cls, p: np.ndarray, y: np.ndarray, w: np.ndarray) -> Platt:
        from sklearn.linear_model import LogisticRegression

        z = _logit(p)[:, None]
        lr = LogisticRegression(C=1e4).fit(z, y.astype(int), sample_weight=w)
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def predict(self, p: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.slope * _logit(p) + self.intercept)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class Unit:
    head: str
    variant: str                       # "full" | "basic"
    segment: str                       # "all" or a role
    family: str | None = None
    params: dict = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)          # numeric inputs
    categories: dict[str, list[str]] = field(default_factory=dict)
    models: list = field(default_factory=list)                # the bagged ensemble
    calibrator: Any = None
    metrics: dict = field(default_factory=dict)
    usable: bool = False
    skipped: str | None = None
    medians: dict[str, float] = field(default_factory=dict)   # a typical situation
    quantiles: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return HEADS[self.head].kind

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """(bags, rows): every bagged model's prediction before calibration."""
        X = matrix(df, self.columns, self.categories)
        return np.array([raw_predict(m, self.kind, X) for m in self.models])

    def predict_each(self, df: pd.DataFrame) -> np.ndarray:
        """(bags, rows): every bagged model's (calibrated) prediction."""
        out = self.predict_raw(df)
        if self.calibrator is not None:
            out = np.array([self.calibrator.predict(p) for p in out])
        return out

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """The ensemble's mean prediction for each row, and the spread between its models."""
        each = self.predict_each(df)
        return each.mean(axis=0), each.std(axis=0)

    def summary(self) -> dict:
        return {"head": self.head, "variant": self.variant, "segment": self.segment,
                "family": self.family, "params": self.params, "usable": self.usable,
                "skipped": self.skipped, "inputs": len(self.columns) + len(self.categories),
                "metrics": self.metrics}


@dataclass
class Brain:
    version: str
    created: str
    config: dict
    data: dict                                   # what it learned from
    units: dict[str, Unit]                       # key: head/variant/segment
    patterns: list[dict] = field(default_factory=list)
    learning_curve: dict | None = None
    excluded: list[dict] = field(default_factory=list)

    @staticmethod
    def key(head: str, variant: str, segment: str) -> str:
        return f"{head}/{variant}/{segment}"

    def unit(self, head: str, variant: str = "full", role: str | None = None) -> Unit | None:
        """The unit to ask: the role's own when it beat the pooled one, else the pooled one;
        the basic variant when the full one wasn't trained."""
        for v in dict.fromkeys((variant, "basic")):
            if role:
                u = self.units.get(self.key(head, v, role))
                if u is not None and not u.skipped and u.metrics.get("beats_pooled"):
                    return u
            u = self.units.get(self.key(head, v, "all"))
            if u is not None and not u.skipped:
                return u
        return None

    def usable(self, head: str, variant: str = "full", role: str | None = None) -> bool:
        u = self.unit(head, variant, role)
        return u is not None and u.usable

    @staticmethod
    def variant_for(df: pd.DataFrame) -> str:
        """``full`` when the situations show the HUD panel (replays), else ``basic``."""
        return "full" if len(df) and df["mana"].notna().mean() > 0.5 else "basic"

    def predict(self, df: pd.DataFrame, variant: str | None = None) -> pd.DataFrame:
        """Every head's prediction (and the ensemble's spread, ``<head>_sd``) per situation,
        by each row's role. Heads without a unit are left out."""
        variant = variant or self.variant_for(df)
        out = pd.DataFrame(index=df.index)
        roles = _row_roles(df)
        for head in HEADS:
            for role in pd.unique(roles):
                unit = self.unit(head, variant, role)
                if unit is None:
                    continue
                rows = roles == role
                mean, sd = unit.predict(df[rows])
                out.loc[rows, head] = mean
                out.loc[rows, f"{head}_sd"] = sd
        return out

    def card(self) -> dict:
        return {"version": self.version, "created": self.created, "config": self.config,
                "data": self.data, "excluded": self.excluded,
                "units": [u.summary() for u in self.units.values()],
                "patterns": self.patterns, "learning_curve": self.learning_curve,
                "features": [{"name": f.name, "group": f.group, "about": f.about,
                              "needs": f.needs} for f in features.CATALOGUE + features.CATEGORICAL]}


def _row_roles(df: pd.DataFrame) -> np.ndarray:
    roles = np.full(len(df), None, dtype=object)
    for r in features.ROLES:
        col = f"role_{r}"
        if col in df:
            roles[df[col].to_numpy(float) == 1] = r
    return roles
