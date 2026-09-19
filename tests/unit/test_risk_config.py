"""Unit tests for the RiskConfig knob-bag (vol-target + profit-lock overrides)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from riskledger.risk.risk_config import RiskConfig


def test_defaults_are_a_noop() -> None:
    cfg = RiskConfig()
    assert cfg.vol_target is False
    assert cfg.profit_lock is False
    assert cfg.any_active is False
    # default tunables
    assert cfg.vol_ewma_span == 20
    assert cfg.vol_scalar_floor == Decimal("0.5")
    assert cfg.vol_scalar_ceil == Decimal("1.5")
    assert cfg.profit_lock_at == Decimal("1.0")


def test_vol_target_flips_any_active() -> None:
    assert RiskConfig(vol_target=True).any_active is True


def test_profit_lock_flips_any_active() -> None:
    assert RiskConfig(profit_lock=True).any_active is True


def test_both_active() -> None:
    assert RiskConfig(vol_target=True, profit_lock=True).any_active is True


def test_frozen_assignment_raises() -> None:
    cfg = RiskConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.vol_target = True  # type: ignore[misc]
