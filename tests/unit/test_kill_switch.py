"""Unit tests for the KillSwitch (atomic, idempotent, fail-safe)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from riskledger.audit.compliance_log import ComplianceLog
from riskledger.core.clock import utcnow
from riskledger.core.enums import AccountStatus, Direction
from riskledger.core.types import Account, Fill, Order, Position
from riskledger.risk.kill_switch import KillSwitch


class FakeBroker:
    """Minimal in-test BrokerAdapter; configurable failure modes."""

    def __init__(self, positions: list[Position], *, leave_open: bool = False,
                 raise_on_flatten: bool = False) -> None:
        self._positions = list(positions)
        self._leave_open = leave_open
        self._raise = raise_on_flatten
        self.flatten_calls = 0
        self.cancel_calls = 0

    def submit(self, order: Order) -> Fill:  # pragma: no cover - unused here
        raise NotImplementedError

    def cancel_all(self) -> int:
        self.cancel_calls += 1
        return 2

    def flatten(self) -> list[Fill]:
        self.flatten_calls += 1
        if self._raise:
            raise RuntimeError("broker down")
        if not self._leave_open:
            self._positions = []
        return [Fill("o1", "ES", Direction.SHORT, Decimal("1"), Decimal("100"), utcnow())]

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def get_equity(self) -> Decimal:  # pragma: no cover - unused here
        return Decimal("50000")

    def set_mark(self, symbol: str, price: Decimal) -> None:  # pragma: no cover
        pass


def _account() -> Account:
    return Account.open("acc-1", "topstep", Decimal("50000"))


def _pos() -> list[Position]:
    return [Position("ES", Direction.LONG, Decimal("2"), Decimal("100"))]


def test_trigger_flattens_verifies_and_suspends(tmp_path: Path) -> None:
    broker = FakeBroker(_pos())
    acc = _account()
    log = ComplianceLog(tmp_path / "audit.jsonl")
    ks = KillSwitch(broker, acc, log)
    res = ks.trigger("kill at 90%")
    assert res.verified is True
    assert res.positions_remaining == 0
    assert res.cancelled == 2
    assert acc.status is AccountStatus.KILL_SWITCH_TRIGGERED
    assert log.verify() is True


def test_idempotent(tmp_path: Path) -> None:
    broker = FakeBroker(_pos())
    ks = KillSwitch(broker, _account())
    first = ks.trigger("x")
    second = ks.trigger("y")
    assert first is second
    assert broker.flatten_calls == 1  # not re-flattened


def test_positions_left_open_reported_unverified() -> None:
    broker = FakeBroker(_pos(), leave_open=True)
    acc = _account()
    ks = KillSwitch(broker, acc, max_verify_attempts=2)
    res = ks.trigger("x")
    assert res.verified is False
    assert res.positions_remaining > 0
    assert acc.status is AccountStatus.KILL_SWITCH_TRIGGERED  # still suspended


def test_broker_failure_still_suspends(tmp_path: Path) -> None:
    broker = FakeBroker(_pos(), raise_on_flatten=True)
    acc = _account()
    log = ComplianceLog(tmp_path / "audit.jsonl")
    ks = KillSwitch(broker, acc, log)
    res = ks.trigger("x")
    assert res.verified is False
    assert acc.status is AccountStatus.KILL_SWITCH_TRIGGERED
    assert [e.event_type for e in log.entries()] == ["KILL_90"]


def test_alert_failure_does_not_block_suspension() -> None:
    def boom(_: dict) -> None:
        raise RuntimeError("alert channel down")

    broker = FakeBroker(_pos())
    acc = _account()
    ks = KillSwitch(broker, acc, alert=boom)
    ks.trigger("x")
    assert acc.status is AccountStatus.KILL_SWITCH_TRIGGERED
