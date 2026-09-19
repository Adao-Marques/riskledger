"""Unit tests for the FixedFractionalSizer (Triple Cap)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from riskledger.config.models import PropFirmRules
from riskledger.core.enums import Direction, DrawdownType
from riskledger.core.types import Account, Signal
from riskledger.risk.position_sizer import FixedFractionalSizer

RULES = PropFirmRules(drawdown_type=DrawdownType.STATIC, max_total_drawdown=Decimal("2000"))


def _acc(equity: str = "50000") -> Account:
    a = Account.open("a", "f", Decimal("50000"))
    a.update_equity(Decimal(equity))
    return a


def _sig(entry: str = "100", stop: str = "98") -> Signal:
    return Signal(
        id="s", strategy_id="orm", ts=datetime(2024, 1, 1, tzinfo=UTC),
        symbol="ES", direction=Direction.LONG,
        entry_price=Decimal(entry), stop_price=Decimal(stop),
    )


def test_dd_buffer_cap_binds() -> None:
    # equity 50000, floor 48000, remaining 2000 -> cap_buffer = 0.10*2000 = 200
    # base 250, per_trade 500 -> min 200; risk/contract = 2 -> 100 contracts
    sizer = FixedFractionalSizer(max_contracts=1000)
    assert sizer.size(_sig(), _acc(), RULES) == Decimal("100")


def test_per_trade_cap_binds() -> None:
    # risk_fraction 0.02 -> base 1000 > per_trade 500; big buffer -> per_trade wins
    sizer = FixedFractionalSizer(risk_fraction=Decimal("0.02"), max_contracts=10000)
    rules = PropFirmRules(drawdown_type=DrawdownType.STATIC, max_total_drawdown=Decimal("40000"))
    # equity 50000, floor 10000, remaining 40000 -> cap_buffer 4000; per_trade 500 binds
    assert sizer.size(_sig(), _acc(), rules) == Decimal("250")  # 500 / 2


def test_base_fraction_binds() -> None:
    sizer = FixedFractionalSizer(max_contracts=10000)
    rules = PropFirmRules(drawdown_type=DrawdownType.STATIC, max_total_drawdown=Decimal("40000"))
    # base 250 < per_trade 500 < cap_buffer 4000 -> base binds -> 125 contracts
    assert sizer.size(_sig(), _acc(), rules) == Decimal("125")


def test_absolute_contract_cap_binds() -> None:
    sizer = FixedFractionalSizer(max_contracts=3)
    assert sizer.size(_sig(), _acc(), RULES) == Decimal("3")


def test_zero_risk_per_unit_returns_zero() -> None:
    sizer = FixedFractionalSizer()
    assert sizer.size(_sig(entry="100", stop="100"), _acc(), RULES) == Decimal("0")


def test_exhausted_buffer_returns_zero() -> None:
    sizer = FixedFractionalSizer()
    assert sizer.size(_sig(), _acc("47000"), RULES) == Decimal("0")  # below floor


def test_never_exceeds_one_percent_risk() -> None:
    sizer = FixedFractionalSizer(max_contracts=100000)
    for eq in ("20000", "33333", "50000", "98765"):
        for stop in ("99.5", "97", "90"):
            acc = _acc(eq)
            sig = _sig(stop=stop)
            n = sizer.size(sig, acc, PropFirmRules(
                drawdown_type=DrawdownType.STATIC, max_total_drawdown=Decimal("90000")))
            risk = n * sig.risk_per_unit  # point_value 1
            assert risk <= Decimal("0.01") * acc.current_equity
