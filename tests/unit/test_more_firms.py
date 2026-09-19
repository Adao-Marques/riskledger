"""Unit tests for the additional prop firms (FTMO, Apex) loaded as data."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from riskledger.core.enums import AccountPhase, DrawdownType
from riskledger.firms.firm_registry import FirmRegistry

CONFIGS = Path(__file__).resolve().parents[2] / "riskledger" / "firms" / "configs"


def test_registry_lists_all_three_firms() -> None:
    """The configs directory now yields topstep, ftmo and apex."""
    reg = FirmRegistry.from_dir(CONFIGS)
    firms = reg.list_firms()
    assert "topstep" in firms
    assert "ftmo" in firms
    assert "apex" in firms


def test_ftmo_eval_is_static_with_10k_drawdown() -> None:
    """FTMO's evaluation phase is a static 10k drawdown."""
    reg = FirmRegistry.from_dir(CONFIGS)
    firm = reg.firm("ftmo")
    assert firm.evaluation_type == "two_phase"
    assert firm.fee == Decimal("540")
    assert firm.profit_split == Decimal("0.8")
    rules = reg.rules_for("ftmo", AccountPhase.EVAL_F1)
    assert rules.drawdown_type is DrawdownType.STATIC
    assert rules.max_total_drawdown == Decimal("10000")
    assert rules.max_daily_loss == Decimal("5000")
    assert rules.profit_target == Decimal("10000")


def test_apex_eval_is_trailing() -> None:
    """Apex's evaluation phase uses a trailing drawdown that does not lock."""
    reg = FirmRegistry.from_dir(CONFIGS)
    firm = reg.firm("apex")
    assert firm.evaluation_type == "one_phase"
    assert firm.fee == Decimal("167")
    rules = reg.rules_for("apex", AccountPhase.EVAL_F1)
    assert rules.drawdown_type is DrawdownType.TRAILING
    assert rules.max_total_drawdown == Decimal("2500")
    assert rules.trailing_locks_at_initial is False


def test_both_firms_have_payout_funded_phase() -> None:
    """Both new firms expose a FUNDED phase with a payout objective."""
    reg = FirmRegistry.from_dir(CONFIGS)
    for firm_id in ("ftmo", "apex"):
        funded = reg.firm(firm_id).phase(AccountPhase.FUNDED)
        assert funded.objective == "payout"
        assert funded.rules.profit_target is None
