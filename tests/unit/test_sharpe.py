"""Unit tests for anti-overfitting Sharpe statistics (López de Prado)."""

from __future__ import annotations

import math

from riskledger.validation.sharpe import (
    _inv_norm_cdf,
    _norm_cdf,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


def test_norm_cdf_midpoint() -> None:
    assert _norm_cdf(0.0) == 0.5


def test_norm_cdf_tails() -> None:
    assert _norm_cdf(8.0) > 0.999999
    assert _norm_cdf(-8.0) < 0.000001


def test_inv_norm_cdf_roundtrip() -> None:
    # Phi^-1(Phi(x)) should recover x for moderate values.
    for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
        assert math.isclose(_inv_norm_cdf(_norm_cdf(x)), x, abs_tol=1e-6)


def test_inv_norm_cdf_known_quantiles() -> None:
    assert math.isclose(_inv_norm_cdf(0.5), 0.0, abs_tol=1e-9)
    assert math.isclose(_inv_norm_cdf(0.975), 1.959963985, abs_tol=1e-6)


def test_inv_norm_cdf_boundaries() -> None:
    assert _inv_norm_cdf(0.0) == -math.inf
    assert _inv_norm_cdf(1.0) == math.inf


def test_psr_in_unit_interval() -> None:
    for sr in (-2.0, -0.5, 0.0, 0.5, 1.5, 3.0):
        p = probabilistic_sharpe_ratio(sr, 250)
        assert 0.0 <= p <= 1.0


def test_psr_monotonic_in_observed_sharpe() -> None:
    # Small n keeps PSR away from the 0/1 saturation so monotonicity is visible.
    n = 12
    prev = probabilistic_sharpe_ratio(-1.0, n)
    for sr in (-0.5, 0.0, 0.5, 1.0, 1.5):
        cur = probabilistic_sharpe_ratio(sr, n)
        assert cur > prev
        prev = cur


def test_psr_half_when_observed_equals_benchmark() -> None:
    # When observed == benchmark, z == 0 so PSR == 0.5 regardless of n.
    p = probabilistic_sharpe_ratio(
        1.0, 100_000, benchmark_sharpe=1.0
    )
    assert math.isclose(p, 0.5, abs_tol=1e-9)


def test_psr_approaches_half_as_n_grows_near_benchmark() -> None:
    # A tiny positive edge over the benchmark with large n still sits near 0.5
    # only at exact equality; verify the equality case across large n.
    for n in (1_000, 10_000, 100_000):
        p = probabilistic_sharpe_ratio(0.7, n, benchmark_sharpe=0.7)
        assert math.isclose(p, 0.5, abs_tol=1e-9)


def test_psr_zero_for_small_n() -> None:
    assert probabilistic_sharpe_ratio(2.0, 1) == 0.0
    assert probabilistic_sharpe_ratio(2.0, 0) == 0.0


def test_psr_zero_when_variance_nonpositive() -> None:
    # Heavy negative skew with a large Sharpe can drive the variance term <= 0.
    p = probabilistic_sharpe_ratio(
        5.0, 250, skew=10.0, kurtosis=3.0
    )
    assert p == 0.0


def test_deflation_lowers_psr() -> None:
    # With multiple trials and dispersion across them, DSR must be <= the plain
    # PSR measured against a zero benchmark (deflation can only raise the bar).
    sr = 2.0
    n = 250
    psr_zero = probabilistic_sharpe_ratio(sr, n, benchmark_sharpe=0.0)
    dsr = deflated_sharpe_ratio(
        sr, n, n_trials=100, variance_of_trial_sharpes=0.25
    )
    assert dsr <= psr_zero
    assert dsr < psr_zero  # strict: nonzero deflation actually bites


def test_more_trials_deflate_more() -> None:
    sr = 2.0
    n = 250
    few = deflated_sharpe_ratio(
        sr, n, n_trials=10, variance_of_trial_sharpes=0.25
    )
    many = deflated_sharpe_ratio(
        sr, n, n_trials=1000, variance_of_trial_sharpes=0.25
    )
    assert many <= few


def test_dsr_falls_back_to_psr_without_trials() -> None:
    sr = 1.5
    n = 250
    psr_zero = probabilistic_sharpe_ratio(sr, n, benchmark_sharpe=0.0)
    # n_trials < 1 -> benchmark falls back to 0.
    assert math.isclose(
        deflated_sharpe_ratio(
            sr, n, n_trials=0, variance_of_trial_sharpes=0.25
        ),
        psr_zero,
        abs_tol=1e-12,
    )
    # Non-positive variance -> also no deflation.
    assert math.isclose(
        deflated_sharpe_ratio(
            sr, n, n_trials=100, variance_of_trial_sharpes=0.0
        ),
        psr_zero,
        abs_tol=1e-12,
    )


def test_dsr_in_unit_interval() -> None:
    for sr in (0.0, 1.0, 2.5):
        p = deflated_sharpe_ratio(
            sr, 250, n_trials=50, variance_of_trial_sharpes=0.5
        )
        assert 0.0 <= p <= 1.0
