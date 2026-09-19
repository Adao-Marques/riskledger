"""Unit tests for the PolicyEngine (rules-driven signal validation)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

from riskledger.audit.compliance_log import ComplianceLog
from riskledger.config.models import PropFirmRules
from riskledger.core.enums import AccountStatus, Direction, DrawdownType, SignalState
from riskledger.core.types import Account, Signal
from riskledger.policy.policy_engine import PolicyEngine

UTC_FROM = datetime.fromisoformat


def _signal(ts: str = "2024-01-02T14:00:00+00:00", size: str | None = None, **kw: object) -> Signal:
    base: dict[str, object] = {
        "id": "s1", "strategy_id": "orm", "ts": UTC_FROM(ts),
        "symbol": "ES", "direction": Direction.LONG,
        "entry_price": Decimal("100"), "stop_price": Decimal("98"),
        "suggested_size": Decimal(size) if size is not None else None,
    }
    base.update(kw)
    return Signal(**base)  # type: ignore[arg-type]


def test_validates_clean_signal(account: Account, topstep_rules: PropFirmRules) -> None:
    out = PolicyEngine().validate(_signal(), account, topstep_rules)
    assert out.state is SignalState.VALIDATED


def test_rejects_inactive_account(account: Account, topstep_rules: PropFirmRules) -> None:
    account.status = AccountStatus.SUSPENDED
    out = PolicyEngine().validate(_signal(), account, topstep_rules)
    assert out.state is SignalState.REJECTED
    assert "not active" in (out.rejection_reason or "")


def test_rejects_when_total_drawdown_at_block(account: Account, topstep_rules: PropFirmRules) -> None:
    # floor = 50000 - 2000 = 48000; block at 80% consumed -> equity 48400
    account.update_equity(Decimal("48300"))
    out = PolicyEngine().validate(_signal(), account, topstep_rules)
    assert out.state is SignalState.REJECTED
    assert "total drawdown" in (out.rejection_reason or "")


def test_rejects_worst_case_breaching_floor(account: Account, topstep_rules: PropFirmRules) -> None:
    # equity 48500, floor 48000; size huge so worst-case loss crosses the floor
    account.update_equity(Decimal("48500"))
    out = PolicyEngine().validate(_signal(size="1000"), account, topstep_rules)
    assert out.state is SignalState.REJECTED
    assert "worst-case" in (out.rejection_reason or "")


def test_rejects_after_mandatory_flat(account: Account, topstep_rules: PropFirmRules) -> None:
    out = PolicyEngine().validate(_signal(ts="2024-01-02T21:00:00+00:00"), account, topstep_rules)
    assert out.state is SignalState.REJECTED
    assert "mandatory flat" in (out.rejection_reason or "")


def test_rejects_daily_loss(account: Account) -> None:
    rules = PropFirmRules(
        drawdown_type=DrawdownType.STATIC,
        max_total_drawdown=Decimal("5000"),
        max_daily_loss=Decimal("1000"),
    )
    account.update_equity(Decimal("49100"))  # lost 900 of 1000 daily -> 90% > block
    out = PolicyEngine().validate(_signal(), account, rules, day_start_equity=Decimal("50000"))
    assert out.state is SignalState.REJECTED
    assert "daily loss" in (out.rejection_reason or "")


def test_rejects_news_blackout(account: Account) -> None:
    rules = PropFirmRules(
        drawdown_type=DrawdownType.STATIC,
        max_total_drawdown=Decimal("5000"),
        news_blackout=True,
    )
    out = PolicyEngine().validate(_signal(metadata={"news_blackout": True}), account, rules)
    assert out.state is SignalState.REJECTED
    assert "news" in (out.rejection_reason or "")


def test_records_decisions_to_compliance_log(
    account: Account, topstep_rules: PropFirmRules, tmp_path: Path
) -> None:
    log = ComplianceLog(tmp_path / "audit.jsonl")
    engine = PolicyEngine(compliance_log=log)
    engine.validate(_signal(), account, topstep_rules)
    account.status = AccountStatus.SUSPENDED
    engine.validate(_signal(), account, topstep_rules)
    events = [e.event_type for e in log.entries()]
    assert events == ["SIGNAL_VALIDATED", "SIGNAL_REJECTED"]
    assert log.verify() is True
