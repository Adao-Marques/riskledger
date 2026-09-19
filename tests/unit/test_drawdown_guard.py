"""Unit tests for the DrawdownGuard (equity -> discrete risk levels)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from riskledger.config.models import PropFirmRules
from riskledger.core.enums import DrawdownType
from riskledger.core.types import Account
from riskledger.risk.drawdown_guard import DrawdownGuard, GuardLevel


@pytest.fixture
def static_rules() -> PropFirmRules:
    return PropFirmRules(drawdown_type=DrawdownType.STATIC, max_total_drawdown=Decimal("2000"))


def _acc() -> Account:
    return Account.open("a", "f", Decimal("50000"))


def test_levels_across_a_decline(static_rules: PropFirmRules) -> None:
    guard = DrawdownGuard(static_rules)
    acc = _acc()
    # floor 48000, buffer 2000. consumed = (50000-eq)/2000
    assert guard.on_tick(acc, Decimal("50000"), day_start_equity=Decimal("50000")).level is GuardLevel.OK
    assert guard.on_tick(acc, Decimal("48500"), day_start_equity=Decimal("50000")).level is GuardLevel.ALERT  # 0.75
    assert guard.on_tick(acc, Decimal("48300"), day_start_equity=Decimal("50000")).level is GuardLevel.BLOCK  # 0.85
    assert guard.on_tick(acc, Decimal("48100"), day_start_equity=Decimal("50000")).level is GuardLevel.KILL   # 0.95


def test_sizing_multiplier(static_rules: PropFirmRules) -> None:
    guard = DrawdownGuard(static_rules)
    acc = _acc()
    assert guard.on_tick(acc, Decimal("50000"), day_start_equity=Decimal("50000")).sizing_multiplier == Decimal("1")
    assert guard.on_tick(acc, Decimal("48500"), day_start_equity=Decimal("50000")).sizing_multiplier == Decimal("0.5")
    assert guard.on_tick(acc, Decimal("48100"), day_start_equity=Decimal("50000")).sizing_multiplier == Decimal("0")


def test_trailing_ratchets_hwm_on_tick() -> None:
    rules = PropFirmRules(drawdown_type=DrawdownType.TRAILING, max_total_drawdown=Decimal("2000"))
    guard = DrawdownGuard(rules)
    acc = _acc()
    guard.on_tick(acc, Decimal("51000"), day_start_equity=Decimal("50000"))
    assert acc.high_watermark == Decimal("51000")
    guard.on_tick(acc, Decimal("50500"), day_start_equity=Decimal("50000"))
    assert acc.high_watermark == Decimal("51000")  # does not drop


def test_eod_does_not_ratchet_intraday_but_does_at_close() -> None:
    rules = PropFirmRules(drawdown_type=DrawdownType.EOD, max_total_drawdown=Decimal("2000"))
    guard = DrawdownGuard(rules)
    acc = _acc()
    guard.on_tick(acc, Decimal("51000"), day_start_equity=Decimal("50000"))
    assert acc.high_watermark == Decimal("50000")  # no intraday ratchet
    guard.end_of_day(acc)
    assert acc.high_watermark == Decimal("51000")


def test_on_bar_trailing_ratchets_to_peak_not_close() -> None:
    # The favourable intra-bar extreme (peak) should ratchet the HWM, not the
    # close — otherwise the trailing floor lags the real high-water mark.
    rules = PropFirmRules(drawdown_type=DrawdownType.TRAILING, max_total_drawdown=Decimal("2000"))
    guard = DrawdownGuard(rules)
    acc = _acc()
    guard.on_bar(acc, peak_equity=Decimal("51000"), trough_equity=Decimal("50200"),
                 day_start_equity=Decimal("50000"))
    assert acc.high_watermark == Decimal("51000")  # peak, not the 50200 "close"


def test_on_bar_trough_breach_kills_even_if_close_recovers() -> None:
    # Floor = HWM(50000) - 2000 = 48000. The bar dips to 47900 (breach) but
    # "closes" at 49000. The intra-bar trough must still fire a KILL.
    rules = PropFirmRules(drawdown_type=DrawdownType.TRAILING, max_total_drawdown=Decimal("2000"))
    guard = DrawdownGuard(rules)
    acc = _acc()
    a = guard.on_bar(acc, peak_equity=Decimal("49000"), trough_equity=Decimal("47900"),
                     day_start_equity=Decimal("50000"))
    assert a.level is GuardLevel.KILL
    assert a.triggering_vector == "total"


def test_on_bar_static_does_not_ratchet(static_rules: PropFirmRules) -> None:
    # Static drawdown: a favourable peak must NOT move the high-watermark.
    guard = DrawdownGuard(static_rules)
    acc = _acc()
    guard.on_bar(acc, peak_equity=Decimal("51000"), trough_equity=Decimal("50500"),
                 day_start_equity=Decimal("50000"))
    assert acc.high_watermark == Decimal("50000")  # unchanged
    assert acc.current_equity == Decimal("50500")  # assessed at the trough


def test_worst_vector_wins_daily(static_rules: PropFirmRules) -> None:
    rules = PropFirmRules(
        drawdown_type=DrawdownType.STATIC,
        max_total_drawdown=Decimal("10000"),  # total stays healthy
        max_daily_loss=Decimal("1000"),
    )
    guard = DrawdownGuard(rules)
    acc = _acc()
    a = guard.on_tick(acc, Decimal("49050"), day_start_equity=Decimal("50000"))  # daily 95%
    assert a.triggering_vector == "daily"
    assert a.level is GuardLevel.KILL


def test_eod_hwm_ratchets_to_the_close_not_the_intrabar_trough() -> None:
    """An ``eod`` floor must step to the day's CLOSE, not the last bar's low.

    ``on_bar`` deliberately leaves ``current_equity`` at the intra-bar trough so a
    breach is caught even when the close recovers. Ratcheting from that same value
    under-ratchets the high-watermark whenever the day closed above the prior one,
    leaving the floor lower than the firm's and letting the account survive
    drawdowns that would really have killed it.

    Four firms state the rule in writing: the floor steps end-of-day, the breach is
    tested per-tick on open equity.
    """
    rules = PropFirmRules(drawdown_type=DrawdownType.EOD,
                          max_total_drawdown=Decimal("2000"))
    guard = DrawdownGuard(rules)
    account = Account.open("eod-1", "firm", Decimal("50000"))

    # A bar that dips to 50,400 and closes at 51,000.
    guard.on_bar(account, peak_equity=Decimal("51200"), trough_equity=Decimal("50400"),
                 day_start_equity=Decimal("50000"))
    assert account.high_watermark == Decimal("50000")     # no intraday ratchet

    guard.end_of_day(account, Decimal("51000"))
    assert account.high_watermark == Decimal("51000"), "the floor did not step to the close"

    # Without the closing equity it would have ratcheted to the trough instead.
    other = Account.open("eod-2", "firm", Decimal("50000"))
    guard.on_bar(other, peak_equity=Decimal("51200"), trough_equity=Decimal("50400"),
                 day_start_equity=Decimal("50000"))
    guard.end_of_day(other)
    assert other.high_watermark == Decimal("50400")       # the documented fallback
