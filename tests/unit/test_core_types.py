"""Unit tests for core types, money and clock helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from riskledger.core.clock import ensure_utc
from riskledger.core.enums import Direction, SignalState
from riskledger.core.money import money, to_decimal
from riskledger.core.types import Account, Position, Signal


def _signal(**kw: object) -> Signal:
    base: dict[str, object] = {
        "id": "s1", "strategy_id": "orm", "ts": datetime(2024, 1, 1, tzinfo=UTC),
        "symbol": "ES", "direction": Direction.LONG,
        "entry_price": Decimal("100"), "stop_price": Decimal("98"),
    }
    base.update(kw)
    return Signal(**base)  # type: ignore[arg-type]


def test_to_decimal_rejects_float() -> None:
    with pytest.raises(TypeError):
        to_decimal(1.5)  # type: ignore[arg-type]


def test_to_decimal_rejects_bool() -> None:
    with pytest.raises(TypeError):
        to_decimal(True)  # type: ignore[arg-type]


def test_money_rounds_half_up() -> None:
    assert money("1.005") == Decimal("1.01")


def test_ensure_utc_rejects_naive() -> None:
    with pytest.raises(ValueError):
        ensure_utc(datetime(2024, 1, 1))


def test_signal_state_transitions() -> None:
    s = _signal()
    assert s.state is SignalState.PENDING
    v = s.validated(size=Decimal("2"))
    assert v.state is SignalState.VALIDATED and v.suggested_size == Decimal("2")
    r = s.rejected("daily loss exceeded")
    assert r.state is SignalState.REJECTED and r.rejection_reason == "daily loss exceeded"
    # original is unchanged (frozen)
    assert s.state is SignalState.PENDING


def test_signal_risk_per_unit() -> None:
    assert _signal().risk_per_unit == Decimal("2")


def test_signal_naive_ts_rejected() -> None:
    with pytest.raises(ValueError):
        _signal(ts=datetime(2024, 1, 1))


def test_position_unrealized_pnl_long_and_short() -> None:
    long = Position("ES", Direction.LONG, Decimal("2"), Decimal("100"))
    assert long.unrealized_pnl(Decimal("105")) == Decimal("10")
    short = Position("ES", Direction.SHORT, Decimal("2"), Decimal("100"))
    assert short.unrealized_pnl(Decimal("105")) == Decimal("-10")


def test_account_hwm_ratchets_up_only() -> None:
    acc = Account.open("a", "topstep", Decimal("50000"))
    acc.update_equity(Decimal("50500"))
    assert acc.high_watermark == Decimal("50500")
    acc.update_equity(Decimal("50200"))
    assert acc.high_watermark == Decimal("50500")  # does not drop
    assert acc.current_equity == Decimal("50200")
