"""Unit tests for VolTargetSizer, build_sizer and min_fundable_balance."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from riskledger.config.models import PropFirmRules
from riskledger.core.enums import Direction, DrawdownType
from riskledger.core.types import Account, Signal
from riskledger.risk.position_sizer import (
    FixedFractionalSizer,
    VolTargetSizer,
    build_sizer,
    min_fundable_balance,
)
from riskledger.risk.risk_config import RiskConfig

POINT_VALUE = Decimal("5")  # MES
# Trailing 50k account with a 2500 total-drawdown buffer (Topstep-like).
RULES = PropFirmRules(drawdown_type=DrawdownType.TRAILING, max_total_drawdown=Decimal("2500"))


def _acc() -> Account:
    return Account.open("a", "f", Decimal("50000"))


def _sig(stop_distance: str) -> Signal:
    """A LONG signal whose stop sits ``stop_distance`` price-points below entry."""
    entry = Decimal("5000")
    return Signal(
        id="s", strategy_id="orm", ts=datetime(2024, 1, 1, tzinfo=UTC),
        symbol="ES", direction=Direction.LONG,
        entry_price=entry, stop_price=entry - Decimal(stop_distance),
    )


# -- min_fundable_balance ------------------------------------------------------

def test_min_fundable_basic() -> None:
    # default risk_fraction 0.005 -> balance = rpc / 0.005
    assert min_fundable_balance(Decimal("50")) == Decimal("10000")
    assert min_fundable_balance(Decimal("1000")) == Decimal("200000")


def test_min_fundable_degenerate_returns_zero() -> None:
    assert min_fundable_balance(Decimal("0")) == Decimal("0")
    assert min_fundable_balance(Decimal("-5")) == Decimal("0")


def test_min_fundable_respects_custom_risk_fraction() -> None:
    # 1% risk fraction halves the required balance vs the 0.5% default.
    assert min_fundable_balance(Decimal("50"), risk_fraction=Decimal("0.01")) == Decimal("5000")


# -- build_sizer ---------------------------------------------------------------

def test_build_sizer_none_is_fixed_fractional() -> None:
    sizer = build_sizer(None, max_contracts=10, point_value=POINT_VALUE)
    assert type(sizer) is FixedFractionalSizer


def test_build_sizer_default_config_is_fixed_fractional() -> None:
    sizer = build_sizer(RiskConfig(), max_contracts=10, point_value=POINT_VALUE)
    assert type(sizer) is FixedFractionalSizer


def test_build_sizer_vol_target_is_vol_target_sizer() -> None:
    sizer = build_sizer(
        RiskConfig(vol_target=True), max_contracts=10, point_value=POINT_VALUE
    )
    assert isinstance(sizer, VolTargetSizer)


# -- VolTargetSizer ------------------------------------------------------------

def test_vol_sizer_steady_stream_matches_base() -> None:
    # A steady stream of identical-stop signals -> scalar ~ 1 -> same size as base.
    base = FixedFractionalSizer(max_contracts=1000, point_value=POINT_VALUE)
    vol = VolTargetSizer(max_contracts=1000, point_value=POINT_VALUE)

    sig = _sig("2")  # 2-point stop
    base_size = base.size(sig, _acc(), RULES)
    assert base_size > Decimal("0")  # the stop is fundable on this account

    last = Decimal("0")
    for _ in range(16):
        last = vol.size(_sig("2"), _acc(), RULES)
    assert last == base_size


def test_vol_sizer_sizes_down_on_vol_spike() -> None:
    base = FixedFractionalSizer(max_contracts=1000, point_value=POINT_VALUE)
    vol = VolTargetSizer(max_contracts=1000, point_value=POINT_VALUE)

    # Seed a calm baseline (small 2-point stops).
    for _ in range(16):
        vol.size(_sig("2"), _acc(), RULES)

    # A volatility spike: ~4x the stop distance.
    wide = _sig("8")
    base_size = base.size(wide, _acc(), RULES)
    vol_size = vol.size(wide, _acc(), RULES)

    # The base sizer should still fund several contracts on the wide stop, and the
    # vol sizer must cut that down (it saw a calm baseline, now reads a spike).
    assert base_size >= Decimal("2")
    assert vol_size <= base_size
    assert vol_size < base_size
