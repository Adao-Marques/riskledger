"""Property tests for the co-pilot's core promise.

The product claim is "the floor we show you is the floor the engine enforces". These
fuzz that claim over the whole input space rather than a handful of cases:

* the reported ``total_floor`` is *identical* to :func:`total_loss_floor`, the shared
  function the live ``DrawdownGuard`` / ``PolicyEngine`` build on, for every drawdown
  type, balance, equity and high-watermark;
* the reported guard level and consumed fraction are exactly what ``DrawdownGuard``
  would return for the same account;
* an account at or below its floor is never told anything but STOP, and is never
  offered a position size.
"""

from __future__ import annotations

from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from riskledger.business.copilot import AccountSnapshot, assess_account
from riskledger.config.models import PropFirmRules
from riskledger.core.enums import DrawdownType
from riskledger.core.types import Account
from riskledger.risk.drawdown import total_loss_floor
from riskledger.risk.drawdown_guard import DrawdownGuard

_money = st.decimals(min_value=Decimal("1000"), max_value=Decimal("500000"),
                     places=2, allow_nan=False, allow_infinity=False)
_buffer = st.decimals(min_value=Decimal("100"), max_value=Decimal("25000"),
                      places=2, allow_nan=False, allow_infinity=False)
_delta = st.decimals(min_value=Decimal("-30000"), max_value=Decimal("30000"),
                     places=2, allow_nan=False, allow_infinity=False)


def _rules(dd: DrawdownType, buffer: Decimal, locks: bool,
           daily: Decimal | None) -> PropFirmRules:
    return PropFirmRules(drawdown_type=dd, max_total_drawdown=buffer,
                         max_daily_loss=daily, trailing_locks_at_initial=locks,
                         profit_target=Decimal("3000"))


def _account(snap: AccountSnapshot) -> Account:
    acct = Account.open("p", snap.firm_id, snap.initial_balance)
    acct.current_equity = snap.current_equity
    acct.high_watermark = max(snap.high_watermark, snap.initial_balance)
    return acct


@settings(max_examples=250, deadline=None)
@given(dd=st.sampled_from(list(DrawdownType)), balance=_money, buffer=_buffer,
       equity_delta=_delta, peak_delta=_delta, locks=st.booleans(),
       daily=st.one_of(st.none(), _buffer))
def test_reported_floor_is_exactly_what_the_engine_enforces(
        dd: DrawdownType, balance: Decimal, buffer: Decimal, equity_delta: Decimal,
        peak_delta: Decimal, locks: bool, daily: Decimal | None) -> None:
    equity = balance + equity_delta
    snap = AccountSnapshot(
        firm_id="f", initial_balance=balance, current_equity=equity,
        high_watermark=max(equity, equity + abs(peak_delta)),
        day_start_equity=equity, point_value=Decimal("5"))
    rules = _rules(dd, buffer, locks, daily)
    report = assess_account(snap, rules)

    acct = _account(snap)
    assert report.total_floor == total_loss_floor(rules, acct)
    assert report.distance_to_floor == equity - report.total_floor

    guard = DrawdownGuard(rules).assess(acct, day_start_equity=snap.day_start_equity)
    assert report.guard_level == guard.level.value
    assert report.buffer_consumed_pct == guard.worst_fraction
    assert report.sizing_multiplier == guard.sizing_multiplier
    assert report.triggering_vector == guard.triggering_vector


@settings(max_examples=200, deadline=None)
@given(dd=st.sampled_from(list(DrawdownType)), balance=_money, buffer=_buffer,
       below=st.decimals(min_value=Decimal("0"), max_value=Decimal("5000"), places=2),
       stop=st.decimals(min_value=Decimal("0.25"), max_value=Decimal("100"), places=2))
def test_an_account_at_or_below_its_floor_is_always_stopped_and_sized_to_zero(
        dd: DrawdownType, balance: Decimal, buffer: Decimal, below: Decimal,
        stop: Decimal) -> None:
    # Build an account sitting exactly `below` under its own floor.
    probe = AccountSnapshot(firm_id="f", initial_balance=balance,
                            current_equity=balance, high_watermark=balance,
                            day_start_equity=balance, point_value=Decimal("5"))
    rules = _rules(dd, buffer, locks=False, daily=None)
    floor = total_loss_floor(rules, _account(probe))
    equity = floor - below
    snap = AccountSnapshot(firm_id="f", initial_balance=balance, current_equity=equity,
                           high_watermark=balance, day_start_equity=balance,
                           point_value=Decimal("5"), stop_distance=stop)
    report = assess_account(snap, rules)

    assert report.guard_level == "KILL"
    assert report.verdict == "STOP"
    assert report.prudent_contracts == 0
    assert report.max_safe_contracts == 0
    assert report.can_pass_now is False


@settings(max_examples=150, deadline=None)
@given(balance=_money, buffer=_buffer,
       gain=st.decimals(min_value=Decimal("0"), max_value=Decimal("20000"), places=2),
       stop=st.decimals(min_value=Decimal("0.25"), max_value=Decimal("50"), places=2),
       point_value=st.sampled_from([Decimal("1"), Decimal("5"), Decimal("50")]))
def test_the_worst_case_of_the_max_safe_size_never_crosses_the_floor(
        balance: Decimal, buffer: Decimal, gain: Decimal, stop: Decimal,
        point_value: Decimal) -> None:
    equity = balance + gain
    snap = AccountSnapshot(firm_id="f", initial_balance=balance, current_equity=equity,
                           high_watermark=equity, day_start_equity=equity,
                           point_value=point_value, stop_distance=stop)
    rules = _rules(DrawdownType.STATIC, buffer, locks=False, daily=None)
    report = assess_account(snap, rules)
    worst_case_loss = Decimal(report.max_safe_contracts) * stop * point_value
    assert equity - worst_case_loss >= report.total_floor
    assert report.prudent_contracts <= report.max_safe_contracts
