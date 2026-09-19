"""The named performance conventions, defined once.

Seven sites in this repo computed a "Sharpe ratio" under three incompatible
conventions, and four computed a "drawdown" under three definitions. The numbers in
the conclusion documents were therefore **not comparable with each other** unless the
reader knew which script produced each one -- a Sharpe of 0.6 from the per-trade
convention and a Sharpe of 0.6 from the annualised-daily convention say entirely
different things about the same strategy.

The fix is not to collapse them: they measure genuinely different quantities and each
has a legitimate use. The fix is to NAME them, implement each exactly once, and make a
caller state which one it means.

Conventions in use here:

``per_trade_sharpe``
    Mean / standard deviation across *trade* PnLs, **not annualised**. A unitless
    measure of how consistent individual trades are. Used by the event-driven
    backtest, which produces independent trades with no shared timeline and so has
    no periods-per-year to annualise by.

``annualised_sharpe``
    Mean / standard deviation across *periodic returns*, scaled by
    ``sqrt(periods_per_year)``. The portfolio convention, comparable to published
    fund figures. Used by the equity-curve metrics and the research layer.

    A caveat worth stating where it is used: annualising by ``sqrt(252)`` assumes the
    strategy is exposed on all 252 days. A book that trades on 82 days a year and is
    flat the rest inflates its annualised Sharpe by roughly ``sqrt(252/82)`` -- one
    headline in the research this was extracted from read 6.10 and was really about
    3.48 for this reason. Pass ``periods_per_year`` honestly, or use
    :func:`active_period_sharpe`.

``max_drawdown_absolute`` / ``max_drawdown_fraction``
    The largest peak-to-trough fall of an equity curve, in currency units and as a
    fraction of the running peak respectively. The fraction is the one comparable
    across accounts of different size; the absolute is the one a prop drawdown rule
    is written in.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

TRADING_DAYS = 252


def per_trade_sharpe(pnls: Sequence[float]) -> float:
    """Mean / stdev of per-trade PnL. NOT annualised. 0.0 if undefined."""
    if len(pnls) < 2:
        return 0.0
    stdev = statistics.stdev(pnls)
    if stdev == 0.0:
        return 0.0
    return statistics.fmean(pnls) / stdev


def annualised_sharpe(returns: Sequence[float],
                      periods_per_year: int = TRADING_DAYS) -> float:
    """Mean / stdev of periodic returns, scaled by ``sqrt(periods_per_year)``.

    ``periods_per_year`` must describe the series actually passed: daily returns of a
    strategy that is only in the market some days are still daily observations, but
    see the module docstring on what annualising a sparse series does to the number.
    """
    if len(returns) < 2:
        return 0.0
    stdev = statistics.stdev(returns)
    if stdev == 0.0:
        return 0.0
    return statistics.fmean(returns) / stdev * math.sqrt(periods_per_year)


def active_period_sharpe(returns: Sequence[float],
                         periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe counting only the periods with a non-zero return.

    For a book that is flat most of the time, this annualises by the number of days it
    was actually exposed rather than by the whole calendar, which is the honest figure
    when comparing a sparse strategy against a continuously-invested one.
    """
    active = [r for r in returns if r != 0.0]
    if len(active) < 2 or not returns:
        return 0.0
    fraction_active = len(active) / len(returns)
    return annualised_sharpe(active, max(1, round(periods_per_year * fraction_active)))


def max_drawdown_absolute(equity: Sequence[float]) -> float:
    """Largest peak-to-trough fall of ``equity``, in the curve's own units (>= 0)."""
    peak = -math.inf
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst


def max_drawdown_fraction(equity: Sequence[float]) -> float:
    """Largest peak-to-trough fall as a fraction of the running peak.

    Requires a real ACCOUNT EQUITY curve — one that starts from a positive balance.
    A cumulative-PnL series starting at zero is rejected rather than served, because
    dividing by a near-zero peak produces a number that reads like a percentage and
    is not one: ``[0, -5, -3, 2, -1]`` would otherwise report a "150% drawdown".
    Use :func:`max_drawdown_absolute` for those, or rebase onto a balance first.

    A result above 1.0 is possible only if equity went negative, which is itself the
    finding and is deliberately not clamped away.
    """
    if not equity:
        return 0.0
    if equity[0] <= 0:
        raise ValueError(
            "max_drawdown_fraction needs an account equity curve starting from a "
            "positive balance; this looks like a cumulative-PnL series "
            f"(first value {equity[0]}). Use max_drawdown_absolute instead."
        )
    peak = -math.inf
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    return worst
