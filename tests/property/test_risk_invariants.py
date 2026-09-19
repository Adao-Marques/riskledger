"""Property-based tests for the risk invariants where a bug loses the account.

These use Hypothesis to fuzz the inputs and assert the invariants hold for ALL
of them, not just hand-picked cases:

* Triple Cap: a sized trade never risks more than 1% of equity.
* Trailing high-watermark: never decreases, for any equity sequence.
* Kill switch: idempotent and fail-safe — the account always ends suspended,
  whatever the broker does; ``verified`` implies the book is flat.
* PolicyEngine: never validates a signal that is at/over the block level or whose
  worst case would breach the drawdown floor (fail-safe: when in doubt, reject).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from riskledger.config.models import PropFirmRules
from riskledger.core.enums import AccountStatus, Direction, DrawdownType, SignalState
from riskledger.core.types import Account, Fill, Order, Position, Signal
from riskledger.policy.policy_engine import PolicyEngine
from riskledger.risk.drawdown import total_consumed_fraction, total_loss_floor
from riskledger.risk.drawdown_guard import DrawdownGuard
from riskledger.risk.kill_switch import KillSwitch
from riskledger.risk.position_sizer import FixedFractionalSizer

EQUITY = st.decimals(min_value=Decimal("1000"), max_value=Decimal("1000000"),
                     places=2, allow_nan=False, allow_infinity=False)
RPU = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("500"),
                  places=2, allow_nan=False, allow_infinity=False)
PV = st.decimals(min_value=Decimal("0.1"), max_value=Decimal("100"),
                 places=2, allow_nan=False, allow_infinity=False)
MAXDD = st.decimals(min_value=Decimal("100"), max_value=Decimal("50000"),
                    places=2, allow_nan=False, allow_infinity=False)


def _static_rules(max_dd: Decimal) -> PropFirmRules:
    return PropFirmRules(drawdown_type=DrawdownType.STATIC, max_total_drawdown=max_dd)


def _signal(rpu: Decimal, size: Decimal | None = None) -> Signal:
    return Signal(
        id="s", strategy_id="t", ts=datetime(2024, 1, 2, 14, 0, tzinfo=UTC),
        symbol="ES", direction=Direction.LONG,
        entry_price=Decimal("1000") + rpu, stop_price=Decimal("1000"),
        suggested_size=size,
    )


@given(equity=EQUITY, rpu=RPU, pv=PV, max_dd=MAXDD)
def test_triple_cap_never_risks_more_than_one_percent(
    equity: Decimal, rpu: Decimal, pv: Decimal, max_dd: Decimal
) -> None:
    account = Account.open("a", "f", equity)
    rules = _static_rules(max_dd)
    sizer = FixedFractionalSizer(max_contracts=10_000_000, point_value=pv)
    size = sizer.size(_signal(rpu), account, rules)
    assert size >= 0
    risk = size * rpu * pv
    assert risk <= Decimal("0.01") * equity


@given(equities=st.lists(EQUITY, min_size=1, max_size=50))
def test_trailing_hwm_never_decreases(equities: list[Decimal]) -> None:
    account = Account.open("a", "f", equities[0])
    guard = DrawdownGuard(PropFirmRules(
        drawdown_type=DrawdownType.TRAILING, max_total_drawdown=Decimal("2000")))
    prev = account.high_watermark
    for eq in equities:
        guard.on_tick(account, eq, day_start_equity=equities[0])
        assert account.high_watermark >= prev
        prev = account.high_watermark


class _FuzzBroker:
    """A broker whose calls fail / leave positions according to the fuzzed flags."""

    def __init__(self, *, fail_cancel: bool, fail_flatten: bool,
                 fail_getpos: bool, n_remaining: int) -> None:
        self._fail_cancel = fail_cancel
        self._fail_flatten = fail_flatten
        self._fail_getpos = fail_getpos
        self._positions = [
            Position(f"P{i}", Direction.LONG, Decimal("1"), Decimal("100"))
            for i in range(n_remaining)
        ]

    def submit(self, order: Order) -> Fill:  # pragma: no cover - unused
        raise NotImplementedError

    def cancel_all(self) -> int:
        if self._fail_cancel:
            raise RuntimeError("cancel failed")
        return 0

    def flatten(self) -> list[Fill]:
        if self._fail_flatten:
            raise RuntimeError("flatten failed")
        return []

    def get_positions(self) -> list[Position]:
        if self._fail_getpos:
            raise RuntimeError("get_positions failed")
        return list(self._positions)

    def get_equity(self) -> Decimal:  # pragma: no cover - unused
        return Decimal("0")

    def set_mark(self, symbol: str, price: Decimal) -> None:  # pragma: no cover
        pass


@settings(max_examples=80)
@given(
    fail_cancel=st.booleans(), fail_flatten=st.booleans(),
    fail_getpos=st.booleans(), n_remaining=st.integers(min_value=0, max_value=3),
)
def test_kill_switch_always_suspends_and_is_idempotent(
    fail_cancel: bool, fail_flatten: bool, fail_getpos: bool, n_remaining: int
) -> None:
    account = Account.open("a", "topstep", Decimal("50000"))
    broker = _FuzzBroker(fail_cancel=fail_cancel, fail_flatten=fail_flatten,
                         fail_getpos=fail_getpos, n_remaining=n_remaining)
    ks = KillSwitch(broker, account, max_verify_attempts=2)  # type: ignore[arg-type]
    first = ks.trigger("fuzz")
    second = ks.trigger("again")
    # Fail-safe: always suspended, no matter what the broker did.
    assert account.status is AccountStatus.KILL_SWITCH_TRIGGERED
    # Idempotent: the second trigger returns the very same result.
    assert first is second
    # Never claim verified unless the book is actually flat.
    if first.verified:
        assert first.positions_remaining == 0


@given(equity=EQUITY, max_dd=MAXDD, pv=PV, size=st.decimals(
    min_value=Decimal("0"), max_value=Decimal("1000"), places=0,
    allow_nan=False, allow_infinity=False))
def test_policy_validates_only_when_safe(
    equity: Decimal, max_dd: Decimal, pv: Decimal, size: Decimal
) -> None:
    account = Account.open("a", "f", max(equity, Decimal("1000")))
    account.current_equity = equity
    rules = _static_rules(max_dd)
    signal = _signal(Decimal("2"), size=size if size > 0 else None)
    result = PolicyEngine(point_value=pv).validate(
        signal, account, rules, day_start_equity=account.initial_balance
    )
    if result.state is SignalState.VALIDATED:
        # Validation implies we were below the block level...
        assert total_consumed_fraction(rules, account) < rules.block_threshold
        # ...and the dollar worst case (incl. point_value) did not breach the
        # floor. Without point_value here the assertion would be a tautology that
        # never exercises the real check.
        if signal.suggested_size is not None:
            worst = account.current_equity - signal.risk_per_unit * signal.suggested_size * pv
            assert worst >= total_loss_floor(rules, account)
