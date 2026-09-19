"""Unit tests for the FirmRegistry (firms modelled as data)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from riskledger.core.enums import AccountPhase, DrawdownType
from riskledger.firms.firm_registry import FirmRegistry

CONFIGS = Path(__file__).resolve().parents[2] / "riskledger" / "firms" / "configs"


def test_loads_topstep_from_dir() -> None:
    reg = FirmRegistry.from_dir(CONFIGS)
    assert "topstep" in reg.list_firms()
    firm = reg.firm("topstep")
    assert firm.evaluation_type == "one_phase"
    assert firm.fee == Decimal("165")
    assert firm.profit_split == Decimal("0.9")


def test_rules_for_eval_phase() -> None:
    reg = FirmRegistry.from_dir(CONFIGS)
    rules = reg.rules_for("topstep", AccountPhase.EVAL_F1)
    assert rules.drawdown_type is DrawdownType.TRAILING
    assert rules.max_total_drawdown == Decimal("2000")
    assert rules.profit_target == Decimal("3000")
    assert rules.trailing_locks_at_initial is True


def test_funded_phase_objective_is_payout() -> None:
    reg = FirmRegistry.from_dir(CONFIGS)
    funded = reg.firm("topstep").phase(AccountPhase.FUNDED)
    assert funded.objective == "payout"
    assert funded.rules.consistency_max_day_pct == Decimal("0.5")


def test_unknown_firm_and_phase_raise() -> None:
    reg = FirmRegistry.from_dir(CONFIGS)
    with pytest.raises(KeyError):
        reg.firm("nonexistent")
    with pytest.raises(KeyError):
        reg.rules_for("topstep", AccountPhase.SCALING)


def test_duplicate_firm_id_raises(tmp_path: Path) -> None:
    doc = (CONFIGS / "topstep.yaml").read_text(encoding="utf-8")
    (tmp_path / "a.yaml").write_text(doc, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(doc, encoding="utf-8")
    with pytest.raises(ValueError):
        FirmRegistry.from_dir(tmp_path)
