"""Every trained brain, kept with its model card.

Brains live under ``<models dir>/brain/<version>/`` (the fitted models, and ``card.json`` --
what it learned from, every head's held-out scores, importances, patterns, learning curve).
``CURRENT`` names the one the coach uses: the newest, unless training was told not to switch,
and any earlier version can be switched back to.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from riftwatch.brain.model import Brain

DIRNAME = "brain"


def root(models_dir: Path | str) -> Path:
    return Path(models_dir) / DIRNAME


def save(brain: Brain, models_dir: Path | str, promote: bool = True) -> str:
    import joblib

    base = root(models_dir)
    base.mkdir(parents=True, exist_ok=True)
    version = time.strftime("%Y%m%d-%H%M%S")
    n = 1
    while (base / version).exists():
        n += 1
        version = f"{time.strftime('%Y%m%d-%H%M%S')}-{n}"
    brain.version = version
    target = base / version
    target.mkdir()
    joblib.dump(brain, target / "brain.joblib")
    (target / "card.json").write_text(json.dumps(brain.card(), indent=2, default=str))
    if promote:
        use(models_dir, version)
    return version


def versions(models_dir: Path | str) -> list[dict]:
    """Every saved brain's card, oldest first, marked ``current``."""
    base = root(models_dir)
    if not base.exists():
        return []
    now = current(models_dir)
    out = []
    for d in sorted(p for p in base.iterdir() if (p / "card.json").exists()):
        card = json.loads((d / "card.json").read_text())
        card["current"] = d.name == now
        out.append(card)
    return out


def current(models_dir: Path | str) -> str | None:
    marker = root(models_dir) / "CURRENT"
    if not marker.exists():
        return None
    name = marker.read_text().strip()
    return name if (root(models_dir) / name / "brain.joblib").exists() else None


def use(models_dir: Path | str, version: str) -> None:
    if not (root(models_dir) / version / "brain.joblib").exists():
        raise ValueError(f"no brain version {version!r} under {root(models_dir)}")
    (root(models_dir) / "CURRENT").write_text(version)


def load(models_dir: Path | str, version: str | None = None) -> Brain | None:
    """The given version, or the current one; None when none has been trained."""
    version = version or current(models_dir)
    if version is None:
        return None
    path = root(models_dir) / version / "brain.joblib"
    if not path.exists():
        raise ValueError(f"no brain version {version!r} under {root(models_dir)}")
    import joblib

    return joblib.load(path)
