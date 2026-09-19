"""The named performance conventions must stay distinct and correctly defined."""

from __future__ import annotations

import math

import pytest

from riskledger.validation.conventions import (
    active_period_sharpe,
    annualised_sharpe,
    max_drawdown_absolute,
    max_drawdown_fraction,
    per_trade_sharpe,
)


def test_the_two_sharpe_conventions_are_not_the_same_number() -> None:
    """The point of naming them: the same series gives two very different figures.

    Seven sites in this repo reported "a Sharpe ratio" under three conventions, so
    figures quoted across the conclusion documents were not comparable unless the
    reader knew which script produced each.
    """
    series = [0.01, -0.004, 0.012, -0.003, 0.008, -0.006, 0.011, -0.002]
    per_trade = per_trade_sharpe(series)
    annualised = annualised_sharpe(series)
    assert annualised == pytest.approx(per_trade * math.sqrt(252))
    assert annualised > per_trade * 10                      # ~15.9x apart


def test_annualising_a_sparse_book_inflates_it_and_active_period_corrects() -> None:
    """A book flat most of the year must not be annualised as if it traded daily.

    This is the gap_fade_fx deflation: ~82 active days a year reported as sqrt(252),
    turning a real ~3.5 into a headline 6.1.
    """
    active = [0.02, -0.01, 0.03, -0.008, 0.025, -0.012]
    sparse = active + [0.0] * 60                            # flat most of the time

    naive = annualised_sharpe(sparse)
    honest = active_period_sharpe(sparse)
    assert honest < naive, "the sparse series was annualised as if fully invested"
    assert honest == pytest.approx(annualised_sharpe(active,
                                                     max(1, round(252 * len(active) / len(sparse)))))


def test_drawdown_absolute_and_fraction_measure_different_things() -> None:
    equity = [100.0, 150.0, 75.0, 120.0]
    assert max_drawdown_absolute(equity) == pytest.approx(75.0)
    assert max_drawdown_fraction(equity) == pytest.approx(0.5)


def test_drawdown_fraction_refuses_a_pnl_series_instead_of_lying() -> None:
    """A fraction against a near-zero peak reads as a percentage and is not one."""
    cumulative_pnl_from_zero = [0.0, -5.0, -3.0, 2.0, -1.0]
    with pytest.raises(ValueError, match="cumulative-PnL"):
        max_drawdown_fraction(cumulative_pnl_from_zero)
    # The absolute form is well defined for the same series: the worst fall is the
    # first one, peak 0 -> trough -5 (the later 2 -> -1 is only 3).
    assert max_drawdown_absolute(cumulative_pnl_from_zero) == pytest.approx(5.0)


@pytest.mark.parametrize("fn", [per_trade_sharpe, annualised_sharpe, active_period_sharpe])
def test_degenerate_inputs_return_zero_not_an_exception(fn: object) -> None:
    assert fn([]) == 0.0                                    # type: ignore[operator]
    assert fn([0.01]) == 0.0                                # type: ignore[operator]
    assert fn([0.01, 0.01]) == 0.0                          # type: ignore[operator]
