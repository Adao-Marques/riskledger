"""Pure drawdown math — the shared source of truth for risk decisions.

Both the PolicyEngine (rejects signals that would breach a limit) and the
DrawdownGuard (polls equity, fires alert/block/kill levels) build on these
functions, so the limit semantics are defined in exactly one place.

All values are :class:`decimal.Decimal`. Functions are pure and side-effect free.
"""

from __future__ import annotations

from decimal import Decimal

from ..config.models import PropFirmRules
from ..core.enums import DrawdownType
from ..core.types import Account

ZERO = Decimal("0")
ONE = Decimal("1")


def total_loss_floor(rules: PropFirmRules, account: Account) -> Decimal:
    """Equity level at/below which the *total* drawdown rule is breached.

    * ``static``:   ``initial_balance - max_total_drawdown`` (fixed).
    * ``trailing``/``eod``: ``high_watermark - max_total_drawdown``. When
      ``trailing_locks_at_initial`` is set, the floor never rises above the
      initial balance (it locks there once the account is far enough in profit).

    Note: for ``eod`` the caller is responsible for only ratcheting
    ``account.high_watermark`` at end-of-day; this function just reads it.
    """
    md = rules.max_total_drawdown
    if rules.drawdown_type is DrawdownType.STATIC:
        return account.initial_balance - md
    floor = account.high_watermark - md
    if rules.trailing_locks_at_initial:
        floor = min(floor, account.initial_balance)
    return floor


def daily_loss_floor(rules: PropFirmRules, day_start_equity: Decimal) -> Decimal | None:
    """Equity floor for the daily-loss rule, or None if the firm has no daily rule."""
    if rules.max_daily_loss is None:
        return None
    return day_start_equity - rules.max_daily_loss


def consumed_fraction(current_equity: Decimal, floor: Decimal, buffer: Decimal) -> Decimal:
    """Fraction of ``buffer`` consumed: 0 = healthy, 1 = at the floor, >1 = breached.

    ``buffer`` is the full allowed loss distance (e.g. ``max_total_drawdown``).
    Negative results (equity above the reference, i.e. in profit beyond the
    buffer) are clamped to 0.
    """
    if buffer <= ZERO:
        return ONE
    remaining = current_equity - floor
    consumed = ONE - (remaining / buffer)
    return consumed if consumed > ZERO else ZERO


def total_consumed_fraction(rules: PropFirmRules, account: Account) -> Decimal:
    """Fraction of the total-drawdown buffer consumed at the current equity."""
    floor = total_loss_floor(rules, account)
    return consumed_fraction(account.current_equity, floor, rules.max_total_drawdown)


def daily_consumed_fraction(rules: PropFirmRules, current_equity: Decimal,
                            day_start_equity: Decimal) -> Decimal:
    """Fraction of the daily-loss buffer consumed (0 if the firm has no daily rule)."""
    if rules.max_daily_loss is None:
        return ZERO
    floor = day_start_equity - rules.max_daily_loss
    return consumed_fraction(current_equity, floor, rules.max_daily_loss)
