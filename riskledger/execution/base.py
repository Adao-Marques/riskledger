"""Broker adapter extension interface.

All execution goes through a ``BrokerAdapter``: a paper simulator and real broker
adapters implement the same interface, so risk logic never knows which it is talking
to.

The interface is deliberately small and synchronous-friendly; the Kill Switch's
safety path depends only on ``cancel_all`` + ``flatten`` + ``get_positions`` +
``get_equity`` working, so those must never depend on the wider stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from ..core.types import Fill, Order, Position


class BrokerAdapter(ABC):
    """Base class for broker adapters."""

    @abstractmethod
    def submit(self, order: Order) -> Fill:
        """Submit an order and return the confirmed Fill. Raise on rejection."""
        raise NotImplementedError

    @abstractmethod
    def cancel_all(self) -> int:
        """Cancel all working orders; return the number cancelled."""
        raise NotImplementedError

    @abstractmethod
    def flatten(self) -> list[Fill]:
        """Close all open positions at market; return the resulting fills."""
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return the currently open positions (broker truth)."""
        raise NotImplementedError

    @abstractmethod
    def get_equity(self) -> Decimal:
        """Return current account equity (broker truth)."""
        raise NotImplementedError

    @abstractmethod
    def set_mark(self, symbol: str, price: Decimal) -> None:
        """Update the current mark price used for equity/PnL (sim convenience)."""
        raise NotImplementedError
