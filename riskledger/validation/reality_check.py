"""Data-snooping multiple-testing correction for the selected objective.

White's **Reality Check** (2000, *A Reality Check for Data Snooping*) and Hansen's
**Test for Superior Predictive Ability** (2005) answer the question the Deflated
Sharpe Ratio does *not*: when you search over ``M`` candidate models and report the
**best** one, is that best performance significantly better than a benchmark, or is
it just the lucky maximum of ``M`` noisy draws? The DSR deflates a *Sharpe*; this
module deflates whatever benchmark-relative per-window statistic the caller feeds
in — for this platform that is the **prop pass rate** (the optimizer selects on
``real_pass_rate`` / ``ev_per_fee``, neither of which the DSR covers).

Setup (White 2000, eq. 2.4; Hansen 2005, §2). Given an ``M`` models × ``W`` windows
matrix of *benchmark-relative* performance ``d[k][w]`` (for pass-rate the caller
passes ``pass_indicator(0/1) - benchmark``), with the per-model sample mean::

    fbar_k = mean_w d[k][w]

the observed statistic is the best standardized mean over the model set::

    V = max_k sqrt(W) * fbar_k.

Under the data-snooping null *H0: max_k E[d_k] <= 0* (no model beats the benchmark),
the sampling distribution of ``V`` is approximated by a **moving-block bootstrap**
over the ``W`` windows (Politis & Romano 1994 — block length = the overlap span so
the resample preserves the windows' autocorrelation; circular wrap; seeded). For
each resample ``b`` with per-model resampled means ``fbar*_k`` we **recenter** by the
observed mean (so the bootstrap world has a true mean of zero — the null) and take::

    V*_b = max_k sqrt(W) * (fbar*_k - fbar_k).

* **White's Reality Check p-value** = fraction of ``V*_b >= V``. Recenters *every*
  model by its own ``fbar_k``; a single very-bad model inflates the null max and
  costs power (White's RC is conservative when poor models are in the set).

* **Hansen's SPA p-value** = same bootstrap, but each model is recentered by
  ``g_k = fbar_k * 1[ fbar_k >= -A_k ]`` (the *consistent* recentering — a model far
  enough below the benchmark is treated as having a true mean of exactly 0, so it
  does NOT contribute a downward pull to the null max) and **studentized** by the
  model's bootstrap std ``sigma_k`` (so models on very different scales are
  comparable and poor models don't dilute power). With the threshold
  ``A_k = (1/4) * W^{-1/4} * sigma_k`` (Hansen 2005, §3.2 — the standard
  ``sqrt(W) * fbar_k >= -sqrt(2 ln ln W)`` rule expressed per-model). The SPA p-value
  is therefore ``<=`` White's RC p-value: it is the **tighter** (more powerful) test.

Both p-values and the index of the best model are returned. **Interpretation for the
caller:** a *high* p-value (e.g. ``> 0.1``) means the best selected performance is
NOT distinguishable from a lucky max-over-``M`` — i.e. the "edge" is most likely a
**data-snooping artifact**. A *low* p-value means the best model survives the
multiple-testing correction over the searched set.

Pure stdlib (``math`` / ``random`` / ``statistics``); no scipy/numpy. Deterministic
for a fixed ``seed``.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

__all__ = ["RealityCheckResult", "reality_check"]


@dataclass(frozen=True, slots=True)
class RealityCheckResult:
    """Outcome of a White-RC / Hansen-SPA data-snooping test.

    Attributes:
        best_index: index ``k`` of the model with the largest ``fbar_k`` (the one the
            caller "selected" — the max-over-``M``).
        best_mean: that model's benchmark-relative sample mean ``fbar_best`` (for
            pass-rate, ``selected_pass_rate - benchmark``).
        white_p: White (2000) Reality Check p-value — P(lucky max >= observed) with
            full-mean recentering. Conservative.
        spa_p: Hansen (2005) SPA p-value — consistent recentering + studentization.
            ``spa_p <= white_p``; the tighter, higher-power test.
        n_models: number of models ``M`` in the comparison set.
        n_obs: number of windows ``W`` (the bootstrap sample length).
    """

    best_index: int
    best_mean: float
    white_p: float
    spa_p: float
    n_models: int
    n_obs: int


def _block_indices(rng: random.Random, n: int, block: int) -> list[int]:
    """One circular moving-block resample of ``[0, n)`` (Politis & Romano 1994).

    Draws ``ceil(n / block)`` block starts uniformly and concatenates ``block``
    consecutive (mod ``n``) indices each, truncated to exactly ``n`` — so every
    resample has the original sample length and preserves within-block ordering
    (hence the autocorrelation of overlapping windows).
    """
    block = max(1, min(block, n))
    n_blocks = math.ceil(n / block)
    idx: list[int] = []
    for _ in range(n_blocks):
        start = rng.randrange(n)
        for k in range(block):
            idx.append((start + k) % n)
    return idx[:n]


def reality_check(
    performance: list[list[float]],
    *,
    block: int = 1,
    n_boot: int = 2000,
    seed: int = 0,
) -> RealityCheckResult:
    """White (2000) Reality Check + Hansen (2005) SPA on a benchmark-relative matrix.

    Args:
        performance: ``M × W`` matrix; ``performance[k][w]`` is model ``k``'s
            performance on window ``w`` **relative to the benchmark**. The CALLER
            subtracts the benchmark — for the prop pass-rate objective pass
            ``pass_indicator(0/1) - benchmark_pass_rate``. The test then answers "is
            the BEST selected pass rate significantly above the benchmark *after*
            accounting for the search over ``M`` configs?".
        block: moving-block length for the bootstrap. Pass the overlap span
            (``ceil(horizon / step)`` for rolling windows) so the resample preserves
            the windows' autocorrelation; ``1`` is the iid case-resampling bootstrap.
        n_boot: number of bootstrap resamples (default 2000).
        seed: RNG seed for ``random.Random`` — fully deterministic output.

    Returns:
        A :class:`RealityCheckResult` with both p-values and the best model's index
        and mean.

    Raises:
        ValueError: if the matrix is empty (``M < 1``), has fewer than 2 windows
            (``W < 2`` — the bootstrap needs a sampling distribution), or is ragged
            (rows of differing length).

    Notes:
        * With a single model (``M == 1``) the RC/SPA max-over-models degenerates to a
          plain one-model moving-block bootstrap test of "is this model's mean above
          the benchmark"; both p-values are still well-defined and returned.
        * ``spa_p <= white_p`` always (Hansen's consistent recentering only ever
          removes downward pulls from the null max).
    """
    m = len(performance)
    if m < 1:
        raise ValueError("performance must contain at least one model (M >= 1)")
    w = len(performance[0])
    if w < 2:
        raise ValueError("each model needs at least 2 windows (W >= 2) to bootstrap")
    if any(len(row) != w for row in performance):
        raise ValueError("ragged matrix: every model row must have the same length W")

    sqrt_w = math.sqrt(w)
    fbar = [statistics.fmean(row) for row in performance]
    best_index = max(range(m), key=lambda k: fbar[k])
    best_mean = fbar[best_index]

    # Observed statistic V = max_k sqrt(W) * fbar_k (White 2000, eq. 2.4).
    v_obs = sqrt_w * best_mean

    rng = random.Random(seed)
    # First pass: collect per-model resampled means to (a) build the White null and
    # (b) estimate each model's bootstrap std sigma_k for the SPA studentization. We
    # store the resampled means so the SPA pass reuses the SAME resamples (identical
    # null draws -> spa_p and white_p are directly comparable / monotone).
    boot_means: list[list[float]] = []  # [b][k] = fbar*_k on resample b
    for _ in range(n_boot):
        idx = _block_indices(rng, w, block)
        row_means = [statistics.fmean([row[j] for j in idx]) for row in performance]
        boot_means.append(row_means)

    # White RC: recenter each model by its full observed mean; null max over models.
    white_hits = 0
    for row_means in boot_means:
        v_star = max(sqrt_w * (row_means[k] - fbar[k]) for k in range(m))
        if v_star >= v_obs:
            white_hits += 1
    white_p = white_hits / n_boot

    # SPA: per-model bootstrap std sigma_k (variation of the recentered mean across
    # resamples), the consistent recentering threshold A_k, the studentized observed
    # statistic and the studentized + consistently-recentered null.
    sigma: list[float] = []
    for k in range(m):
        centred = [boot_means[b][k] - fbar[k] for b in range(n_boot)]
        # Population std of the bootstrap recentered means; floor away from 0 so a
        # constant (e.g. all-pass / all-fail) model doesn't divide-by-zero.
        s = statistics.pstdev(centred) if n_boot > 1 else 0.0
        sigma.append(s if s > 1e-12 else 1e-12)

    # Hansen 2005 §3.2 consistent-recentering threshold A_k = (1/4) W^{-1/4} sigma_k:
    # a model whose standardized mean sqrt(W)*fbar_k/sigma_k is below -2 ln ln W is
    # treated as a true-zero (g_k = 0) so it can't pull the null max downward.
    thresh = [0.25 * (w ** -0.25) * sigma[k] for k in range(m)]
    g = [fbar[k] if fbar[k] >= -thresh[k] else 0.0 for k in range(m)]

    # Studentized observed statistic: max_k sqrt(W) * fbar_k / sigma_k.
    t_obs = max(sqrt_w * fbar[k] / sigma[k] for k in range(m))
    spa_hits = 0
    for row_means in boot_means:
        # Recenter by g_k (consistent), studentize by sigma_k, take the null max.
        t_star = max(sqrt_w * (row_means[k] - g[k]) / sigma[k] for k in range(m))
        if t_star >= t_obs:
            spa_hits += 1
    spa_p = spa_hits / n_boot

    # SPA is the tighter test by construction: clamp to enforce spa_p <= white_p even
    # under Monte Carlo noise from the shared finite resample set.
    spa_p = min(spa_p, white_p)

    return RealityCheckResult(
        best_index=best_index,
        best_mean=best_mean,
        white_p=white_p,
        spa_p=spa_p,
        n_models=m,
        n_obs=w,
    )
