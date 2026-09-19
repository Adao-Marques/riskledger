"""Unit tests for the funded-lifecycle terms on FirmConfig.

These cover the payout terms modelled as firm data, so an expected-value model can
count payout cycles from a firm's cadence instead of assuming a number.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from riskledger.core.enums import AccountPhase
from riskledger.firms.firm_registry import FirmConfig, FirmRegistry

CONFIGS = Path(__file__).resolve().parents[2] / "riskledger" / "firms" / "configs"


def test_all_firms_parse_lifecycle_fields() -> None:
    """Each shipped firm parses the three funded-lifecycle fields."""
    reg = FirmRegistry.from_dir(CONFIGS)
    for firm_id in ("topstep", "ftmo", "apex"):
        firm = reg.firm(firm_id)
        assert firm.payout_per_cycle is not None
        assert firm.payout_per_cycle >= Decimal("0")
        assert firm.days_between_payouts is not None
        assert firm.days_between_payouts > 0
        assert firm.funded_horizon_days > 0


def test_ftmo_lifecycle_cycles() -> None:
    """FTMO: 180-day horizon at a 14-day cadence yields 12 cycles."""
    reg = FirmRegistry.from_dir(CONFIGS)
    firm = reg.firm("ftmo")
    payout, cycles = firm.funded_lifecycle()
    assert payout == Decimal("5000")
    assert cycles == 180 // 14  # 12


def test_topstep_lifecycle_cycles() -> None:
    """Topstep: 180-day horizon at a 30-day cadence yields 6 cycles."""
    reg = FirmRegistry.from_dir(CONFIGS)
    firm = reg.firm("topstep")
    payout, cycles = firm.funded_lifecycle()
    assert payout == Decimal("2500")
    assert cycles == 6


def test_explicit_horizon_overrides_default() -> None:
    """An explicit horizon overrides the configured funded_horizon_days."""
    reg = FirmRegistry.from_dir(CONFIGS)
    firm = reg.firm("ftmo")
    payout, cycles = firm.funded_lifecycle(horizon_days=28)
    assert payout == Decimal("5000")
    assert cycles == 2


def test_firm_without_terms_yields_zero_payouts() -> None:
    """A firm with no funded terms returns (Decimal('0'), 0)."""
    firm = FirmConfig.model_validate(
        {
            "id": "noterms",
            "name": "No Terms 50K",
            "evaluation_type": "one_phase",
            "fee": "100",
            "profit_split": "0.9",
            "payout_reliability": "0.5",
            "phases": [
                {
                    "phase": AccountPhase.FUNDED,
                    "objective": "payout",
                    "rules": {
                        "drawdown_type": "trailing",
                        "max_total_drawdown": "2000",
                        "profit_target": None,
                    },
                }
            ],
        }
    )
    assert firm.payout_per_cycle is None
    assert firm.days_between_payouts is None
    payout, cycles = firm.funded_lifecycle()
    assert payout == Decimal("0")
    assert cycles == 0
    # An explicit horizon still yields zero when terms are missing.
    assert firm.funded_lifecycle(horizon_days=365) == (Decimal("0"), 0)
