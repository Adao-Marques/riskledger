"""Core domain types.

Signals, orders, fills and trades are immutable (frozen) — once created they are
a historical fact. ``Account`` is mutable because equity, high-watermark and
status evolve over the life of the account; use the provided helpers so updates
stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from typing import Any

from .clock import ensure_utc
from .enums import AccountPhase, AccountStatus, Direction, OrderType, SignalState


@dataclass(frozen=True, slots=True)
class Signal:
    """A raw or validated trading intention produced by a strategy."""

    id: str
    strategy_id: str
    ts: datetime
    symbol: str
    direction: Direction
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal | None = None
    suggested_size: Decimal | None = None
    state: SignalState = SignalState.PENDING
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts))

    def validated(self, size: Decimal | None = None) -> Signal:
        """Return a copy marked VALIDATED (optionally with a final size)."""
        return replace(
            self,
            state=SignalState.VALIDATED,
            suggested_size=size if size is not None else self.suggested_size,
            rejection_reason=None,
        )

    def rejected(self, reason: str) -> Signal:
        """Return a copy marked REJECTED with a reason."""
        return replace(self, state=SignalState.REJECTED, rejection_reason=reason)

    @property
    def risk_per_unit(self) -> Decimal:
        """Absolute price distance between entry and stop (per unit/contract)."""
        return abs(self.entry_price - self.stop_price)


@dataclass(frozen=True, slots=True)
class Order:
    """An order to be sent to a broker."""

    id: str
    symbol: str
    direction: Direction
    size: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    signal_id: str | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    """A confirmed execution returned by a broker."""

    order_id: str
    symbol: str
    direction: Direction
    size: Decimal
    price: Decimal
    ts: datetime
    slippage: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts))


@dataclass(frozen=True, slots=True)
class Position:
    """An open position held at a broker."""

    symbol: str
    direction: Direction
    size: Decimal
    avg_price: Decimal

    def unrealized_pnl(self, mark_price: Decimal) -> Decimal:
        """PnL per the current mark price (price units × size, no multiplier)."""
        return (mark_price - self.avg_price) * Decimal(self.direction.sign) * self.size


@dataclass(frozen=True, slots=True)
class Trade:
    """A closed (or closing) round-trip trade for the audit/PnL record."""

    id: str
    account_id: str
    signal_id: str | None
    symbol: str
    direction: Direction
    ts_entry: datetime
    entry_price: Decimal
    size: Decimal
    ts_exit: datetime | None = None
    exit_price: Decimal | None = None
    pnl: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    costs: Decimal = Decimal("0")


@dataclass
class Account:
    """Live state of a prop firm account.

    ``current_equity`` and ``high_watermark`` evolve tick-by-tick. The
    high-watermark only ratchets up; use :meth:`update_equity`.
    """

    id: str
    prop_firm: str
    phase: AccountPhase
    initial_balance: Decimal
    current_equity: Decimal
    high_watermark: Decimal
    status: AccountStatus = AccountStatus.ACTIVE

    @classmethod
    def open(cls, id: str, prop_firm: str, initial_balance: Decimal,
             phase: AccountPhase = AccountPhase.EVAL_F1) -> Account:
        """Create a fresh account with equity and HWM at the initial balance."""
        return cls(
            id=id,
            prop_firm=prop_firm,
            phase=phase,
            initial_balance=initial_balance,
            current_equity=initial_balance,
            high_watermark=initial_balance,
        )

    def update_equity(self, equity: Decimal) -> None:
        """Set current equity and ratchet the high-watermark upward only."""
        self.current_equity = equity
        if equity > self.high_watermark:
            self.high_watermark = equity
