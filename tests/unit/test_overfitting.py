"""Unit tests for the overfitting guards (pure helpers, no Optuna needed)."""

from __future__ import annotations

import math

import pytest

from riskledger.validation.overfitting import (
    chronological_split,
    math_log_odds,
    probability_of_backtest_overfitting,
)


class TestChronologicalSplit:
    def test_default_split_is_first_oos_is_tail(self) -> None:
        is_r, oos_r = chronological_split(100, 0.3)
        assert list(is_r) == list(range(70))
        assert list(oos_r) == list(range(70, 100))
        # Contiguous, non-overlapping, covering everything.
        assert is_r.stop == oos_r.start
        assert oos_r.stop == 100

    def test_zero_frac_disables_holdout(self) -> None:
        is_r, oos_r = chronological_split(50, 0.0)
        assert list(is_r) == list(range(50))
        assert len(oos_r) == 0

    def test_oos_is_strictly_later_than_is(self) -> None:
        is_r, oos_r = chronological_split(10, 0.5)
        assert max(is_r) < min(oos_r)

    def test_floor_keeps_is_nonempty_on_tiny_sets(self) -> None:
        # int() floors the OOS count, so IS always retains at least one sample.
        is_r, oos_r = chronological_split(3, 0.3)  # floor(0.9) == 0 OOS
        assert len(is_r) == 3
        assert len(oos_r) == 0

    @pytest.mark.parametrize("frac", [-0.1, 1.0, 1.5])
    def test_rejects_out_of_range_frac(self, frac: float) -> None:
        with pytest.raises(ValueError):
            chronological_split(10, frac)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            chronological_split(0, 0.3)


class TestLogOdds:
    def test_symmetry(self) -> None:
        assert math_log_odds(0.5) == pytest.approx(0.0)
        assert math_log_odds(0.75) == pytest.approx(-math_log_odds(0.25))

    def test_sign(self) -> None:
        assert math_log_odds(0.9) > 0
        assert math_log_odds(0.1) < 0


class TestPBO:
    def test_no_overfit_consistent_winner(self) -> None:
        # Config 0 is uniformly the best across every fold: an honest, robust edge.
        # The IS-best is always the OOS-best too, so PBO must be ~0.
        n_folds = 8
        matrix = [
            [10.0] * n_folds,  # always best
            [5.0] * n_folds,
            [4.0] * n_folds,
            [3.0] * n_folds,
        ]
        pbo = probability_of_backtest_overfitting(matrix, n_groups=8, n_test_groups=1)
        assert pbo is not None
        assert pbo == pytest.approx(0.0)

    def test_full_overfit_anti_correlated(self) -> None:
        # Each config wins in exactly one fold and is worst everywhere else: the
        # in-sample winner is the leftover, never the OOS winner -> high PBO.
        rng_folds = 6
        n_trials = 6
        matrix: list[list[float]] = []
        for i in range(n_trials):
            row = [0.0] * rng_folds
            row[i] = 100.0  # config i only shines in fold i
            matrix.append(row)
        pbo = probability_of_backtest_overfitting(matrix, n_groups=6, n_test_groups=1)
        assert pbo is not None
        # Highly overfit: IS winner systematically below the OOS median.
        assert pbo > 0.5

    def test_returns_none_when_too_small(self) -> None:
        assert probability_of_backtest_overfitting([[1.0, 2.0]]) is None  # 1 trial
        assert probability_of_backtest_overfitting([[1.0], [2.0]]) is None  # 1 fold

    def test_rejects_ragged_matrix(self) -> None:
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting([[1.0, 2.0], [3.0]])

    def test_pbo_is_a_probability(self) -> None:
        # Random-ish but deterministic matrix: PBO must stay in [0, 1].
        matrix = [
            [math.sin(i + j) for j in range(8)] for i in range(10)
        ]
        pbo = probability_of_backtest_overfitting(matrix, n_groups=8, n_test_groups=1)
        assert pbo is not None
        assert 0.0 <= pbo <= 1.0
