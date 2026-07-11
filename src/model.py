"""Train and evaluate a next-day market-direction classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import RETURN_TARGET_COLUMN, TARGET_COLUMN, feature_columns

# Default decision threshold on P(up); tuned per-model by ``tune_threshold``.
DEFAULT_THRESHOLD = 0.5


@dataclass
class EvaluationResult:
    """Cross-validated performance metrics."""

    accuracy: float
    f1: float
    n_splits: int
    baseline_accuracy: float
    threshold: float = DEFAULT_THRESHOLD


def build_model(model_name: str = "gboost", random_state: int = 42) -> Pipeline:
    """Create a scaling + classification pipeline for the requested model.

    Args:
        model_name: One of ``"gboost"``, ``"forest"`` or ``"logistic"``.
        random_state: Seed for reproducibility.
    """
    if model_name == "gboost":
        clf = GradientBoostingClassifier(random_state=random_state)
    elif model_name == "forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "logistic":
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    return Pipeline(steps=[("scaler", StandardScaler()), ("clf", clf)])


def _xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = feature_columns(frame)
    return frame[features].to_numpy(), frame[TARGET_COLUMN].to_numpy()


def tune_threshold(
    frame: pd.DataFrame, model_name: str = "gboost", n_splits: int = 5
) -> float:
    """Pick the P(up) threshold that maximises out-of-fold accuracy.

    Using a tuned threshold (rather than a fixed 0.5) counteracts the upward
    drift bias in a long-only index series, where a naive classifier tends to
    always predict "up".
    """
    x, y = _xy(frame)
    n_splits = max(2, min(n_splits, len(frame) - 1))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    oof_proba: list[float] = []
    oof_true: list[int] = []
    for train_idx, test_idx in splitter.split(x):
        model = build_model(model_name)
        model.fit(x[train_idx], y[train_idx])
        oof_proba.extend(model.predict_proba(x[test_idx])[:, 1].tolist())
        oof_true.extend(y[test_idx].tolist())

    proba = np.asarray(oof_proba)
    true = np.asarray(oof_true)
    best_t, best_acc = DEFAULT_THRESHOLD, -1.0
    for t in np.linspace(0.35, 0.65, 31):
        acc = accuracy_score(true, (proba >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return round(best_t, 3)


def evaluate_model(
    frame: pd.DataFrame,
    n_splits: int = 5,
    model_name: str = "gboost",
    threshold: float | None = None,
) -> EvaluationResult:
    """Evaluate the model with a forward-chaining time-series split.

    Args:
        frame: Training frame produced by ``build_training_frame``.
        n_splits: Number of forward-chaining folds.
        model_name: Which classifier to evaluate.
        threshold: Decision threshold on P(up); defaults to 0.5.

    Returns:
        An :class:`EvaluationResult` with mean accuracy and F1 across folds.
    """
    x, y = _xy(frame)
    thr = DEFAULT_THRESHOLD if threshold is None else threshold

    n_splits = max(2, min(n_splits, len(frame) - 1))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    accuracies: list[float] = []
    f1s: list[float] = []
    baselines: list[float] = []
    for train_idx, test_idx in splitter.split(x):
        model = build_model(model_name)
        model.fit(x[train_idx], y[train_idx])
        proba = model.predict_proba(x[test_idx])[:, 1]
        preds = (proba >= thr).astype(int)
        accuracies.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, zero_division=0))
        # Majority-class baseline learned from the training fold.
        majority = int(round(y[train_idx].mean()))
        baselines.append(accuracy_score(y[test_idx], np.full_like(preds, majority)))

    return EvaluationResult(
        accuracy=float(np.mean(accuracies)),
        f1=float(np.mean(f1s)),
        n_splits=n_splits,
        baseline_accuracy=float(np.mean(baselines)),
        threshold=thr,
    )


def walk_forward_accuracy(
    frame: pd.DataFrame,
    horizon: int,
    model_name: str = "gboost",
    threshold: float | None = None,
) -> float:
    """Expanding-window walk-forward accuracy over the last ``horizon`` rows.

    For each of the final ``horizon`` samples the model is retrained on all
    prior samples and evaluated on the single next day, mimicking live use.
    """
    x, y = _xy(frame)
    thr = DEFAULT_THRESHOLD if threshold is None else threshold
    horizon = max(1, min(horizon, len(frame) - 10))

    correct = 0
    start = len(frame) - horizon
    for k in range(start, len(frame)):
        model = build_model(model_name)
        model.fit(x[:k], y[:k])
        proba = model.predict_proba(x[k].reshape(1, -1))[0, 1]
        pred = int(proba >= thr)
        correct += int(pred == y[k])
    return correct / horizon


def select_model(
    frame: pd.DataFrame,
    n_splits: int = 5,
    candidates: tuple[str, ...] = ("gboost", "forest", "logistic"),
) -> tuple[str, float, EvaluationResult]:
    """Choose the best (model, threshold) by cross-validated accuracy."""
    best: tuple[str, float, EvaluationResult] | None = None
    for name in candidates:
        thr = tune_threshold(frame, model_name=name, n_splits=n_splits)
        result = evaluate_model(
            frame, n_splits=n_splits, model_name=name, threshold=thr
        )
        if best is None or result.accuracy > best[2].accuracy:
            best = (name, thr, result)
    assert best is not None
    return best


def train_model(
    frame: pd.DataFrame, model_name: str = "gboost", random_state: int = 42
) -> Pipeline:
    """Fit the model on the full training frame."""
    x, y = _xy(frame)
    model = build_model(model_name=model_name, random_state=random_state)
    model.fit(x, y)
    return model


def _xy_return(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = feature_columns(frame)
    return frame[features].to_numpy(), frame[RETURN_TARGET_COLUMN].to_numpy()


def build_return_model(random_state: int = 42) -> Pipeline:
    """Create the scaling + gradient-boosting regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(random_state=random_state)),
        ]
    )


def train_return_model(frame: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """Fit the next-day percentage-change regressor on the full frame."""
    x, y = _xy_return(frame)
    model = build_return_model(random_state=random_state)
    model.fit(x, y)
    return model
