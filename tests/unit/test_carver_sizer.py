"""Unit tests for the Carver-style RealizedVolSizer and build_sizer dispatch."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from riskledger.config.models import PropFirmRules
from riskledger.core.enums import Direction, DrawdownType
from riskledger.core.types import Account, Signal
from riskledger.risk.position_sizer import (
    FixedFractionalSizer,
    RealizedVolSizer,
    VolTargetSizer,
    build_sizer,
)
from riskledger.risk.risk_config import RiskConfig

POINT_VALUE = Decimal("5")  # MES
# Trailing 50k account with a 2500 total-drawdown buffer (Topstep-like).
RULES = PropFirmRules(drawdown_type=DrawdownType.TRAILING, max_total_drawdown=Decimal("2500"))


def _acc() -> Account:
    return Account.open("a", "f", Decimal("50000"))


def _sig(entry: str, *, stop_distance: str = "2", meta: dict[str, object] | None = None) -> Signal:
    """A LONG signal at ``entry`` with a fixed-distance stop below it."""
    e = Decimal(entry)
    return Signal(
        id="s", strategy_id="orm", ts=datetime(2024, 1, 1, tzinfo=UTC),
        symbol="ES", direction=Direction.LONG,
        entry_price=e, stop_price=e - Decimal(stop_distance),
        metadata=meta or {},
    )


# -- build_sizer dispatch ------------------------------------------------------

def test_build_sizer_default_is_fixed_fractional() -> None:
    sizer = build_sizer(RiskConfig(), max_contracts=10, point_value=POINT_VALUE)
    assert type(sizer) is FixedFractionalSizer


def test_build_sizer_vol_stop_is_vol_target() -> None:
    sizer = build_sizer(
        RiskConfig(sizer="vol_stop"), max_contracts=10, point_value=POINT_VALUE
    )
    assert isinstance(sizer, VolTargetSizer)
    assert not isinstance(sizer, RealizedVolSizer)


def test_build_sizer_legacy_vol_target_still_vol_target() -> None:
    sizer = build_sizer(
        RiskConfig(vol_target=True), max_contracts=10, point_value=POINT_VALUE
    )
    assert isinstance(sizer, VolTargetSizer)
    assert not isinstance(sizer, RealizedVolSizer)


def test_build_sizer_vol_realized_is_realized_vol_sizer() -> None:
    sizer = build_sizer(
        RiskConfig(sizer="vol_realized"), max_contracts=10, point_value=POINT_VALUE
    )
    assert isinstance(sizer, RealizedVolSizer)


def test_any_active_reflects_sizer() -> None:
    assert not RiskConfig().any_active
    assert RiskConfig(sizer="vol_stop").any_active
    assert RiskConfig(sizer="vol_realized").any_active
    assert RiskConfig(vol_target=True).any_active


# -- RealizedVolSizer ----------------------------------------------------------

def test_constant_return_vol_matches_base() -> None:
    """A constant per-period return -> scalar ~ 1 -> same size as base."""
    base = FixedFractionalSizer(max_contracts=1000, point_value=POINT_VALUE)
    vol = RealizedVolSizer(max_contracts=1000, point_value=POINT_VALUE)

    # Constant +0.1% step each period (no rounding -> ret is *exactly* identical,
    # so sigma_baseline == sigma_current and the scalar is exactly 1).
    entry = Decimal("5000")
    last = Decimal("0")
    base_size = Decimal("0")
    for _ in range(20):
        entry = entry * Decimal("1.001")
        sig = _sig(str(entry))
        base_size = base.size(sig, _acc(), RULES)
        last = vol.size(sig, _acc(), RULES)
    assert base_size > Decimal("0")
    assert last == base_size


def test_vol_spike_sizes_down() -> None:
    """A big jump in entry price after a calm series cuts the size."""
    vol = RealizedVolSizer(max_contracts=1000, point_value=POINT_VALUE)

    # Calm series: tiny +0.05% steps.
    entry = Decimal("5000")
    calm_size = Decimal("0")
    for _ in range(20):
        entry = (entry * Decimal("1.0005")).quantize(Decimal("0.01"))
        calm_size = vol.size(_sig(str(entry)), _acc(), RULES)
    assert calm_size > Decimal("0")

    # A volatility spike: a large +5% jump in entry price.
    spike_entry = (entry * Decimal("1.05")).quantize(Decimal("0.01"))
    spike_size = vol.size(_sig(str(spike_entry)), _acc(), RULES)
    assert spike_size < calm_size


def test_first_signal_falls_back_to_base() -> None:
    """<2 observations -> base fraction (same as FixedFractional on signal 1)."""
    base = FixedFractionalSizer(max_contracts=1000, point_value=POINT_VALUE)
    vol = RealizedVolSizer(max_contracts=1000, point_value=POINT_VALUE)

    sig = _sig("5000")
    assert vol.size(sig, _acc(), RULES) == base.size(sig, _acc(), RULES)
    # Second signal seeds the EWMA (one squared return) and still falls back.
    sig2 = _sig("5005")
    assert vol.size(sig2, _acc(), RULES) == base.size(sig2, _acc(), RULES)


def test_metadata_realized_vol_override_honoured() -> None:
    """A metadata realized_vol override drives the scalar instead of the self-estimate."""
    base = FixedFractionalSizer(max_contracts=1000, point_value=POINT_VALUE)
    vol = RealizedVolSizer(max_contracts=1000, point_value=POINT_VALUE)

    # Warm a baseline from a calm self-estimated stream.
    entry = Decimal("5000")
    for _ in range(20):
        entry = (entry * Decimal("1.0005")).quantize(Decimal("0.01"))
        vol.size(_sig(str(entry)), _acc(), RULES)

    # Now feed a huge override vol: must size down vs base on the same signal.
    sig = _sig(str(entry), meta={"realized_vol": Decimal("0.5")})
    base_size = base.size(sig, _acc(), RULES)
    over_size = vol.size(sig, _acc(), RULES)
    assert base_size > Decimal("0")
    assert over_size < base_size


def test_metadata_atr_key_also_supported() -> None:
    vol = RealizedVolSizer(max_contracts=1000, point_value=POINT_VALUE)
    # First call seeds prev_entry; provide override directly.
    sig = _sig("5000", meta={"atr": Decimal("0.5")})
    # Baseline is None on the very first override -> baseline == override -> scalar 1.
    base = FixedFractionalSizer(max_contracts=1000, point_value=POINT_VALUE)
    assert vol.size(sig, _acc(), RULES) == base.size(sig, _acc(), RULES)
