"""Overfitting guards for hyperparameter optimisation (López de Prado & Bailey).

When a search (a grid, or a Bayesian optimiser such as Optuna's TPE) reports the
*best* of ``N`` configurations, the headline metric is a MAX over ``N``
selections and is therefore optimistically biased: the more variants you try,
the higher the best in-sample score you expect purely by luck, even with no real
edge. This module collects the small, reusable, framework-free pieces that make
that bias visible:

* :func:`chronological_split` — a time-ordered in-sample / out-of-sample (holdout)
  split. Optimise on IS, *confirm* on OOS; a large IS->OOS drop is the practical,
  López-de-Prado-aligned overfitting signal.
* :func:`probability_of_backtest_overfitting` — PBO via CSCV
  (Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest
  Overfitting*, Journal of Computational Finance, 2016). Given an
  ``(n_trials x n_folds)`` performance matrix, it measures how often the config
  that is best in-sample lands below the median out-of-sample.

Pure Python (stdlib ``itertools``/``statistics`` only): no numpy/scipy, no
strategy or engine dependencies, so the functions are trivially unit-testable
without Optuna installed.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from itertools import combinations


def chronological_split(
    n_samples: int, holdout_frac: float
) -> tuple[range, range]:
    """Split ``range(n_samples)`` into (in-sample, out-of-sample) by time order.

    The first ``1 - holdout_frac`` of the (already chronologically ordered)
    samples is the in-sample (IS) training span; the last ``holdout_frac`` is the
    out-of-sample (OOS) holdout the winning config is *confirmed* on. Because bars
    are time-ordered, this is a true forward holdout — the OOS span is strictly
    later than everything the optimiser saw.

    :param n_samples: total number of samples (``>= 1``).
    :param holdout_frac: fraction reserved for OOS, in ``[0, 1)``. ``0`` disables
        the holdout (OOS is empty, IS is everything) for backward compatibility.
    :raises ValueError: if ``n_samples < 1`` or ``holdout_frac`` is outside
        ``[0, 1)``.
    :returns: ``(is_range, oos_range)`` as two contiguous, non-overlapping
        ``range`` objects covering ``range(n_samples)``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if not (0.0 <= holdout_frac < 1.0):
        raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
    # floor() via int(): the OOS gets the tail, IS keeps at least one bar.
    n_oos = int(n_samples * holdout_frac)
    n_is = n_samples - n_oos
    return range(n_is), range(n_is, n_samples)


def probability_of_backtest_overfitting(
    performance: Sequence[Sequence[float]],
    *,
    n_groups: int = 6,
    n_test_groups: int = 1,
) -> float | None:
    """Probability of Backtest Overfitting (PBO) via CSCV.

    Bailey, Borwein, López de Prado & Zhu (2016), *The Probability of Backtest
    Overfitting*. The CSCV procedure splits the ``n_folds`` time slices into all
    balanced train/test partitions, picks the config that is best on the *train*
    (in-sample) slices, and checks where that same config ranks on the held-out
    *test* (out-of-sample) slices. If the in-sample winner systematically lands in
    the bottom half out-of-sample, the search is overfit.

    Concretely, for each CSCV partition we compute the logit of the relative OOS
    rank ``w`` of the in-sample-best config; PBO is the fraction of partitions
    whose OOS rank is below the median (``logit <= 0``), i.e. the probability that
    the selected strategy underperforms the median out-of-sample.

    :param performance: an ``(n_trials x n_folds)`` matrix — ``performance[i][j]``
        is config ``i``'s score on time slice ``j`` (higher is better). Every row
        must have the same length ``n_folds``.
    :param n_groups: number of contiguous fold-groups to partition into
        (``2..n_folds``; the CSCV ``S``). Clamped to ``n_folds`` if larger.
    :param n_test_groups: groups assigned to the test (OOS) side per partition.
    :returns: PBO in ``[0, 1]``, or ``None`` when the matrix is too small/degenerate
        to form a single valid CSCV partition (fewer than 2 trials, fewer than 2
        usable folds, or no partition leaves a non-empty train and test side).
    """
    n_trials = len(performance)
    if n_trials < 2:
        return None
    n_folds = len(performance[0])
    if any(len(row) != n_folds for row in performance):
        raise ValueError("all rows of `performance` must have the same length")
    if n_folds < 2:
        return None

    groups = min(n_groups, n_folds)
    if groups < 2 or not (1 <= n_test_groups < groups):
        return None
    # A fully constant performance matrix carries no information to judge
    # overfitting — return None rather than a misleading PBO=1.0 (the strict-<
    # rank would make every IS winner appear to lose OOS).
    first = performance[0][0]
    if all(x == first for row in performance for x in row):
        return None

    # Partition the `n_folds` columns into `groups` contiguous fold-groups.
    base, extra = divmod(n_folds, groups)
    fold_groups: list[list[int]] = []
    start = 0
    for g in range(groups):
        size = base + (1 if g < extra else 0)
        fold_groups.append(list(range(start, start + size)))
        start += size

    logits: list[float] = []
    for test_combo in combinations(range(groups), n_test_groups):
        test_folds = [f for g in test_combo for f in fold_groups[g]]
        train_folds = [f for g in range(groups) if g not in test_combo
                       for f in fold_groups[g]]
        if not test_folds or not train_folds:
            continue

        # In-sample (train) score per config; pick the best config.
        is_scores = [statistics.fmean(row[f] for f in train_folds)
                     for row in performance]
        best = max(range(n_trials), key=lambda i: is_scores[i])

        # Out-of-sample (test) score per config; rank of the IS winner.
        oos_scores = [statistics.fmean(row[f] for f in test_folds)
                      for row in performance]
        # Relative rank w in (0, 1): fraction of configs the winner beats OOS.
        beaten = sum(1 for i in range(n_trials)
                     if i != best and oos_scores[i] < oos_scores[best])
        w = (beaten + 0.5) / n_trials  # +0.5 keeps the logit finite at the edges
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        logits.append(math_log_odds(w))

    if not logits:
        return None
    # PBO = P(logit <= 0) = fraction of partitions where the IS winner is below
    # the OOS median.
    below = sum(1 for lo in logits if lo <= 0.0)
    return below / len(logits)


def math_log_odds(w: float) -> float:
    """Logit ``ln(w / (1 - w))`` of a probability ``w`` in ``(0, 1)``.

    Kept as a tiny named helper (rather than inlining ``math.log``) so the PBO
    rank-distribution can be reasoned about and tested in isolation.
    """
    import math

    return math.log(w / (1.0 - w))
