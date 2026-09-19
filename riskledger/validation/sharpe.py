"""Anti-overfitting Sharpe statistics (López de Prado).

A high backtested Sharpe ratio is cheap to manufacture: the more strategy
variants you try, the higher the *maximum* Sharpe you expect to observe purely
by luck, even when none of the variants has any real edge. The objective
function ranks strategies by their probability of *passing* a prop evaluation,
so before any strategy is trusted we must discount its Sharpe for (a) the
finite, non-normal sample it was measured on and (b) the number of trials that
were run to find it.

This module implements two López de Prado statistics:

* **Probabilistic Sharpe Ratio (PSR)** — the probability that the *true* Sharpe
  exceeds a benchmark, given the observed Sharpe, sample length, and the
  skewness / kurtosis of the returns. It corrects the naive Sharpe for short,
  fat-tailed, skewed samples.
* **Deflated Sharpe Ratio (DSR)** — PSR with the benchmark set to the *expected
  maximum* Sharpe under the null hypothesis of no skill, derived from the number
  of trials and the variance of those trials' Sharpes. This is the
  multiple-testing deflation: it asks whether the observed Sharpe beats what we
  would expect to see as the best of many random trials.

Pure ``math`` only (stdlib); no numpy/scipy. All inputs and outputs are
``float``: these are statistical estimators, not monetary quantities.
"""

from __future__ import annotations

import math

# Euler-Mascheroni constant, used in the E[max] order-statistic approximation.
_EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function ``Phi(x)``.

    Computed exactly (to machine precision) from the error function:
    ``Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))``.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _inv_norm_cdf(p: float) -> float:
    """Inverse standard normal CDF (quantile function) ``Phi^-1(p)``.

    Acklam / Beasley-Springer-Moro rational approximation, accurate to roughly
    1.15e-9 over the open interval ``(0, 1)``. Returns ``-inf`` / ``+inf`` at the
    boundaries ``0`` / ``1``.

    :param p: probability in ``[0, 1]``.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    # Coefficients for the rational approximation.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    # Split the domain into low, central, and high regions.
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def sample_skew_kurtosis(values: list[float]) -> tuple[float, float]:
    """Sample skewness and (non-excess) kurtosis of ``values``.

    Returns ``(0.0, 3.0)`` (the normal moments) when there are fewer than four
    observations or the variance is zero, so callers can pass the result
    straight into :func:`probabilistic_sharpe_ratio` as a no-op in degenerate
    cases. Per-trade prop PnL is highly non-normal, so feeding the real moments
    is what makes the PSR correction meaningful rather than reducing to
    ``Phi(SR*sqrt(n-1))``.
    """
    n = len(values)
    if n < 4:
        return 0.0, 3.0
    mean = math.fsum(values) / n
    m2 = math.fsum((v - mean) ** 2 for v in values) / n
    if m2 <= 0.0:
        return 0.0, 3.0
    m3 = math.fsum((v - mean) ** 3 for v in values) / n
    m4 = math.fsum((v - mean) ** 4 for v in values) / n
    std = math.sqrt(m2)
    return m3 / std**3, m4 / m2**2


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    *,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR), López de Prado (2012).

    Estimates the probability that the true Sharpe ratio exceeds
    ``benchmark_sharpe``, correcting the observed Sharpe for the sample length
    and the non-normality (skew / kurtosis) of the returns::

        PSR = Phi(
            (SR_obs - SR_bench) * sqrt(n - 1)
            / sqrt(1 - skew * SR_obs + ((kurtosis - 1) / 4) * SR_obs**2)
        )

    The denominator is the standard error of the estimated Sharpe; negative
    skew and excess kurtosis inflate it, lowering the probability — exactly the
    penalty short, fat-tailed track records deserve.

    :param observed_sharpe: the Sharpe ratio measured on the sample.
    :param n_observations: number of return observations (``>= 2`` required).
    :param benchmark_sharpe: Sharpe to test against (default ``0``).
    :param skew: skewness of the returns (``0`` for normal).
    :param kurtosis: (non-excess) kurtosis of the returns (``3`` for normal).
    :returns: a probability in ``[0, 1]``.

    Fail-safe: returns ``0.0`` when ``n_observations < 2`` or the variance term
    under the square root is non-positive.
    """
    if n_observations < 2:
        return 0.0

    variance = (
        1.0
        - skew * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * observed_sharpe * observed_sharpe
    )
    if variance <= 0.0:
        return 0.0

    z = (
        (observed_sharpe - benchmark_sharpe)
        * math.sqrt(float(n_observations - 1))
        / math.sqrt(variance)
    )
    return _norm_cdf(z)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    *,
    n_trials: int,
    variance_of_trial_sharpes: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR), López de Prado & Bailey (2014).

    The DSR is the PSR evaluated against a benchmark equal to the *expected
    maximum* Sharpe under the null hypothesis of no skill. When you run
    ``n_trials`` independent strategy variants whose Sharpes have variance
    ``variance_of_trial_sharpes``, the best of them is expected to score::

        E[max] = sqrt(V) * (
            (1 - gamma) * Phi^-1(1 - 1/N)
            + gamma * Phi^-1(1 - 1/(N * e))
        )

    where ``V`` is the trial variance, ``N`` the number of trials, ``gamma`` the
    Euler-Mascheroni constant, and ``Phi^-1`` the inverse normal CDF. This is the
    multiple-testing deflation: a Sharpe is only credible if it beats the best
    a lucky random search would have produced. The DSR is then::

        DSR = PSR(observed_sharpe, n_observations, benchmark_sharpe=E[max], ...)

    :param observed_sharpe: the Sharpe ratio measured on the selected strategy.
    :param n_observations: number of return observations (``>= 2`` required).
    :param n_trials: number of strategy configurations tried (``N``).
    :param variance_of_trial_sharpes: variance of the trials' Sharpe ratios.
    :param skew: skewness of the selected strategy's returns.
    :param kurtosis: (non-excess) kurtosis of the selected strategy's returns.
    :returns: a probability in ``[0, 1]``.

    Guard: with ``n_trials < 1`` (or non-positive trial variance) there is no
    deflation, so the benchmark falls back to ``0`` and the DSR reduces to the
    plain PSR.
    """
    if n_trials < 1 or variance_of_trial_sharpes <= 0.0:
        expected_max = 0.0
    else:
        std_trials = math.sqrt(variance_of_trial_sharpes)
        n = float(n_trials)
        expected_max = std_trials * (
            (1.0 - _EULER_MASCHERONI) * _inv_norm_cdf(1.0 - 1.0 / n)
            + _EULER_MASCHERONI * _inv_norm_cdf(1.0 - 1.0 / (n * math.e))
        )

    return probabilistic_sharpe_ratio(
        observed_sharpe,
        n_observations,
        benchmark_sharpe=expected_max,
        skew=skew,
        kurtosis=kurtosis,
    )
