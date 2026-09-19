"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import time
from decimal import Decimal

import pytest

from riskledger.config.models import AccountConfig, PropFirmRules
from riskledger.core.enums import AccountPhase, DrawdownType
from riskledger.core.types import Account


@pytest.fixture
def topstep_rules() -> PropFirmRules:
    """Topstep-style trailing-drawdown rules for a 50k account.

    $2,000 trailing drawdown, locks at the initial balance, $1,000 daily loss,
    $3,000 profit target, flat at 20:55 UTC.
    """
    return PropFirmRules(
        drawdown_type=DrawdownType.TRAILING,
        max_total_drawdown=Decimal("2000"),
        max_daily_loss=Decimal("1000"),
        profit_target=Decimal("3000"),
        min_trading_days=2,
        trailing_locks_at_initial=True,
        mandatory_flat_utc=time(20, 55),
    )


@pytest.fixture
def account() -> Account:
    """A fresh 50k Topstep evaluation account."""
    return Account.open(
        id="acc-1",
        prop_firm="topstep",
        initial_balance=Decimal("50000"),
        phase=AccountPhase.EVAL_F1,
    )


@pytest.fixture
def account_config(topstep_rules: PropFirmRules) -> AccountConfig:
    """A complete account configuration."""
    return AccountConfig(
        id="acc-1",
        prop_firm="topstep",
        phase=AccountPhase.EVAL_F1,
        initial_balance=Decimal("50000"),
        rules=topstep_rules,
    )
