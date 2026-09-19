"""Phase-dependent risk posture: aggressive to pass, conservative to survive."""

from __future__ import annotations

from pathlib import Path

from riskledger.firms.firm_registry import FirmRegistry
from riskledger.risk.risk_config import PhaseRiskPolicy, RiskConfig

_CONFIGS = Path(__file__).resolve().parents[2] / "riskledger" / "firms" / "configs"


def test_default_policy_is_aggressive_eval_conservative_funded() -> None:
    p = PhaseRiskPolicy.default()
    # Evaluation: bold (target-seeking) + bank the pass.
    assert p.evaluation.sizer == "bold"
    assert p.evaluation.profit_lock is True
    # Funded: conservative fixed sizer + de-risk/bank payouts.
    assert p.funded.sizer == "fixed"
    assert p.funded.profit_lock is True
    # The two postures are genuinely different.
    assert p.evaluation.sizer != p.funded.sizer


def test_policy_pairs_two_independent_configs() -> None:
    p = PhaseRiskPolicy(evaluation=RiskConfig(sizer="bold"),
                        funded=RiskConfig(sizer="fixed", profit_lock=True))
    assert p.evaluation.any_active        # bold is an active override
    assert p.funded.profit_lock


def test_for_drawdown_matches_eval_sizer_to_structure() -> None:
    # Bold play (Dubins & Savage) is the barrier-maximising posture against a floor
    # that ratchets; against a fixed floor, staking a fraction of the distance to it
    # over-risks instead. The posture must be matched to the drawdown structure.
    trailing = PhaseRiskPolicy.for_drawdown("trailing")
    static = PhaseRiskPolicy.for_drawdown("static")
    assert trailing.evaluation.sizer == "bold"
    assert static.evaluation.sizer == "fixed"
    # Both keep a conservative funded leg; default() is the trailing-matched posture.
    assert trailing.funded.sizer == static.funded.sizer == "fixed"
    assert PhaseRiskPolicy.default() == trailing


def test_for_firm_reads_the_firms_drawdown_type() -> None:
    registry = FirmRegistry.from_dir(_CONFIGS)
    # apex is trailing -> bold eval; ftmo is static -> fixed eval.
    assert PhaseRiskPolicy.for_firm(registry.firm("apex")).evaluation.sizer == "bold"
    assert PhaseRiskPolicy.for_firm(registry.firm("ftmo")).evaluation.sizer == "fixed"
