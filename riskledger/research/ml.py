"""Lookahead-safe walk-forward ML (purge + embargo) over a feature matrix.

This is the honest core of the ML signal family. A naive ``fit(all) -> predict(all)``
would train on the future and is worthless out of sample. Instead we walk the
timeline forward in sequential test folds; for each fold the model is trained
**only on the past**, and we additionally **purge + embargo** (López de Prado,
*Advances in Financial ML* ch.7): any training sample whose triple-barrier label
resolved at or after ``test_start - embargo`` is dropped, so a label formed using
bars near the test period cannot leak the answer into training.

The default estimator is :class:`~sklearn.ensemble.HistGradientBoostingClassifier`
— the gradient-boosted-trees workhorse the best systematic shops reach for first:
fast, strong on tabular features, and natively NaN-tolerant (our feature matrix
has warmup/missing cross-asset NaNs). All ``float`` / numpy / sklearn.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

ModelFactory = Callable[[], ClassifierMixin]


def default_model() -> ClassifierMixin:
    """A sensible, deterministic gradient-boosted-trees classifier.

    Shallow trees + a learning-rate < 1 + early-stoppable depth keep it from
    memorising 5-minute noise; ``random_state`` fixes determinism so a rerun is
    bit-identical (the platform's reproducibility rule).
    """
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200,
        l2_regularization=1.0, min_samples_leaf=200, random_state=0)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward controls: fold count, embargo, and the warmup train floor."""

    n_splits: int = 6      # sequential test folds across the timeline
    embargo: int = 24      # bars purged between the train tail and the test fold
    min_train: int = 1500  # don't emit predictions until this many train samples exist


def purged_train_mask(t1: np.ndarray, valid: np.ndarray, test_start: int,
                      embargo: int) -> np.ndarray:
    """Boolean mask of training rows usable for a fold beginning at ``test_start``.

    A row ``i`` is trainable iff it is valid, lies strictly before the fold, and
    its label fully resolved before ``test_start - embargo`` (the purge+embargo).
    """
    n = len(t1)
    idx = np.arange(n)
    cutoff = test_start - embargo
    return valid & (idx < test_start) & (t1 < cutoff)


def walk_forward_proba(x: np.ndarray, y: np.ndarray, t1: np.ndarray, valid: np.ndarray,
                       *, config: WalkForwardConfig | None = None,
                       model_factory: ModelFactory = default_model) -> np.ndarray:
    """Per-row P(positive class), trained only on the purged past. NaN where untrained.

    ``x`` is the (n, k) feature matrix (NaNs allowed — the default model tolerates
    them); ``y`` the binary label in {0, 1}; ``t1`` the per-row label-resolution
    index and ``valid`` the per-row label validity (both from
    :func:`triple-barrier labelling`). The timeline is cut into
    ``n_splits`` contiguous test folds; each fold is predicted by a model fit on
    the purged+embargoed rows before it. Rows in the first (unbacked) span, or any
    fold without enough training data, stay ``NaN``.
    """
    cfg = config or WalkForwardConfig()
    n = len(y)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or cfg.n_splits < 1:
        return out
    bounds = np.linspace(0, n, cfg.n_splits + 1, dtype=int)
    for f in range(cfg.n_splits):
        test_start, test_end = int(bounds[f]), int(bounds[f + 1])
        if test_end <= test_start:
            continue
        train = purged_train_mask(t1, valid, test_start, cfg.embargo)
        if int(train.sum()) < cfg.min_train:
            continue
        ytr = y[train]
        if len(np.unique(ytr)) < 2:        # a single-class train fold can't classify
            continue
        xtr = x[train]
        # Drop feature columns with NO finite value in THIS fold's training rows.
        # A cross-asset feature can be globally non-empty yet entirely NaN inside an
        # early fold (e.g. a partner whose history only covers the recent half), and
        # the gradient-boosting binner raises on an all-NaN column. Selecting per
        # fold — and predicting on the same columns — keeps the model NaN-tolerant
        # without leaking anything (the selection is on column emptiness, not labels).
        keep = np.isfinite(xtr).any(axis=0)
        if not keep.any():
            continue
        model = model_factory()
        model.fit(xtr[:, keep], ytr)
        proba = model.predict_proba(x[test_start:test_end][:, keep])
        pos = list(model.classes_).index(1) if 1 in model.classes_ else proba.shape[1] - 1
        out[test_start:test_end] = proba[:, pos]
    return out
