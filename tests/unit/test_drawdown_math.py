"""Unit tests for the shared pure drawdown math."""

from __future__ import annotations

from decimal import Decimal

import pytest

from riskledger.config.models import PropFirmRules
from riskledger.core.enums import DrawdownType
from riskledger.core.types import Account
from riskledger.risk.drawdown import (
    daily_loss_floor,
    total_consumed_fraction,
    total_loss_floor,
)


def _acc(equity: str, hwm: str, initial: str = "50000") -> Account:
    a = Account.open("a", "topstep", Decimal(initial))
    a.high_watermark = Decimal(hwm)
    a.current_equity = Decimal(equity)
    return a


def test_static_floor_is_fixed() -> None:
    rules = PropFirmRules(drawdown_type=DrawdownType.STATIC, max_total_drawdown=Decimal("2000"))
    acc = _acc("51000", "52000")
    assert total_loss_floor(rules, acc) == Decimal("48000")


def test_trailing_floor_follows_hwm() -> None:
    rules = PropFirmRules(drawdown_type=DrawdownType.TRAILING, max_total_drawdown=Decimal("2000"))
    acc = _acc("50500", "51000")
    assert total_loss_floor(rules, acc) == Decimal("49000")


def test_trailing_locks_at_initial() -> None:
    rules = PropFirmRules(
        drawdown_type=DrawdownType.TRAILING,
        max_total_drawdown=Decimal("2000"),
        trailing_locks_at_initial=True,
    )
    # HWM high enough that HWM - 2000 > initial -> floor locks at 50000
    acc = _acc("54000", "55000")
    assert total_loss_floor(rules, acc) == Decimal("50000")


def test_total_consumed_fraction_endpoints() -> None:
    rules = PropFirmRules(drawdown_type=DrawdownType.STATIC, max_total_drawdown=Decimal("2000"))
    healthy = _acc("50000", "50000")  # floor 48000, remaining 2000 -> 0 consumed
    assert total_consumed_fraction(rules, healthy) == Decimal("0")
    at_floor = _acc("48000", "50000")  # remaining 0 -> 1 consumed
    assert total_consumed_fraction(rules, at_floor) == Decimal("1")
    at_90 = _acc("48200", "50000")  # remaining 200 / 2000 = 0.1 -> 0.9 consumed
    assert total_consumed_fraction(rules, at_90) == Decimal("0.9")


def test_daily_loss_floor_none_when_no_rule() -> None:
    rules = PropFirmRules(drawdown_type=DrawdownType.STATIC, max_total_drawdown=Decimal("2000"))
    assert daily_loss_floor(rules, Decimal("50000")) is None


@pytest.mark.parametrize("equity,expected", [("50000", "48000"), ("49000", "47000")])
def test_daily_loss_floor(equity: str, expected: str) -> None:
    rules = PropFirmRules(
        drawdown_type=DrawdownType.STATIC,
        max_total_drawdown=Decimal("2000"),
        max_daily_loss=Decimal("2000"),
    )
    assert daily_loss_floor(rules, Decimal(equity)) == Decimal(expected)
