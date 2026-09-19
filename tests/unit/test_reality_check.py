"""Tests for the White (2000) Reality Check / Hansen (2005) SPA data-snooping test.

These run WITHOUT optuna — they exercise the pure-stdlib ``reality_check`` directly
on synthetic benchmark-relative matrices. The contract being pinned:

* a genuinely superior model (beats the benchmark on every window) -> LOW p-values
  (the test correctly rejects the data-snooping null);
* a set of pure-noise models centred on the benchmark -> HIGH p-value (the test
  correctly FAILS to reject — a lucky max is exactly what RC/SPA is designed to
  discount);
* the block length changes the bootstrap (sanity);
* the same seed gives identical output (determinism);
* degenerate input is guarded.
"""

from __future__ import annotations

import random

import pytest

from riskledger.validation.reality_check import RealityCheckResult, reality_check


def test_genuinely_superior_model_rejects_null() -> None:
    """One model beats the benchmark on EVERY window -> both p-values are tiny.

    Benchmark-relative performance of the good model is a strongly positive constant
    (+0.6 every window, e.g. a 60-pp-above-benchmark pass rate); the others sit at /
    below the benchmark. The best mean is far above zero, so almost no recentered
    bootstrap max reaches it.
    """
    good = [0.6] * 40
    mediocre = [0.0] * 40
    bad = [-0.3] * 40
    res = reality_check([good, mediocre, bad], block=4, n_boot=1000, seed=0)

    assert res.best_index == 0
    assert res.best_mean == pytest.approx(0.6)
    assert res.white_p < 0.05
    assert res.spa_p < 0.05
    assert res.spa_p <= res.white_p
    assert res.n_models == 3
    assert res.n_obs == 40


def test_pure_noise_does_not_reject() -> None:
    """Many noise models centred on the benchmark -> high p-value (no false edge).

    This is the headline honesty property: with ``M`` models that are pure noise
    around the benchmark, the max-over-``M`` mean looks "good" but RC/SPA recognise
    it as a lucky maximum and DO NOT reject (p-value well above 0.1). Mean ~ 0 so the
    selected max is a snooping artifact by construction.
    """
    rng = random.Random(123)
    n_models, n_windows = 20, 60
    matrix = [[rng.gauss(0.0, 1.0) for _ in range(n_windows)] for _ in range(n_models)]
    res = reality_check(matrix, block=1, n_boot=2000, seed=0)

    # The selected (max) model exists and has a positive sample mean (lucky), but the
    # snooping-corrected p-values are NOT significant.
    assert res.best_mean > 0.0
    assert res.white_p > 0.1
    assert res.spa_p > 0.05  # SPA is tighter but a pure-noise max is still not signif.
    assert res.spa_p <= res.white_p


def test_block_length_changes_bootstrap() -> None:
    """Different block lengths give different p-values on autocorrelated data (sanity).

    With strongly serially-correlated windows the moving-block length materially
    changes the resampling distribution, so block=1 and block=10 must not produce the
    identical p-value (the dependence-preserving bootstrap is actually engaged).
    """
    # Autocorrelated benchmark-relative series: a zero-drift random walk (an AR(1)
    # with rho near 1) so the observed mean sits inside the bootstrap bulk and the
    # p-value is in a sensitive mid-range where the block length visibly moves it.
    rng = random.Random(7)
    series: list[float] = []
    x = 0.0
    for _ in range(120):
        x = 0.95 * x + rng.gauss(0.0, 0.1)
        series.append(x)
    matrix = [list(series), [v - 0.05 for v in series]]

    p_block1 = reality_check(matrix, block=1, n_boot=1500, seed=0).white_p
    p_block12 = reality_check(matrix, block=12, n_boot=1500, seed=0).white_p
    # The dependence-preserving bootstrap is actually engaged: the two p-values
    # differ (and the mid-range value confirms we're in the sensitive regime).
    assert p_block1 != p_block12


def test_determinism_same_seed() -> None:
    """Two calls with the same seed and inputs return byte-identical results."""
    rng = random.Random(99)
    matrix = [[rng.gauss(0.05, 1.0) for _ in range(50)] for _ in range(8)]
    a = reality_check(matrix, block=3, n_boot=800, seed=42)
    b = reality_check(matrix, block=3, n_boot=800, seed=42)
    assert a == b
    assert isinstance(a, RealityCheckResult)


def test_different_seed_differs() -> None:
    """Different seeds give (in general) different p-values — the RNG is wired in."""
    rng = random.Random(5)
    matrix = [[rng.gauss(0.0, 1.0) for _ in range(50)] for _ in range(8)]
    a = reality_check(matrix, block=2, n_boot=800, seed=1)
    b = reality_check(matrix, block=2, n_boot=800, seed=2)
    assert (a.white_p, a.spa_p) != (b.white_p, b.spa_p)


def test_single_model_is_a_one_model_bootstrap() -> None:
    """M==1 degenerates to a single-model bootstrap test; p-values still defined."""
    res = reality_check([[0.5] * 30], block=3, n_boot=500, seed=0)
    assert res.best_index == 0
    assert res.n_models == 1
    assert 0.0 <= res.white_p <= 1.0
    assert 0.0 <= res.spa_p <= 1.0


def test_guard_empty_matrix() -> None:
    """M < 1 raises ValueError."""
    with pytest.raises(ValueError, match="at least one model"):
        reality_check([], n_boot=10)


def test_guard_single_window() -> None:
    """W < 2 raises ValueError (no sampling distribution to bootstrap)."""
    with pytest.raises(ValueError, match="at least 2 windows"):
        reality_check([[0.5]], n_boot=10)


def test_guard_ragged_rows() -> None:
    """Rows of differing length raise ValueError."""
    with pytest.raises(ValueError, match="ragged"):
        reality_check([[0.1, 0.2, 0.3], [0.1, 0.2]], n_boot=10)


def test_pvalues_in_unit_interval() -> None:
    """Both p-values are always probabilities in [0, 1] across a random matrix."""
    rng = random.Random(2024)
    matrix = [[rng.gauss(0.1, 0.5) for _ in range(45)] for _ in range(6)]
    res = reality_check(matrix, block=5, n_boot=600, seed=11)
    assert 0.0 <= res.white_p <= 1.0
    assert 0.0 <= res.spa_p <= 1.0
    assert res.spa_p <= res.white_p
