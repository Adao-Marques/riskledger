"""Risk-sizing extension interface.

A ``Sizer`` computes the position size for a validated signal. The MVP ships
Fixed Fractional + Triple Cap; Kelly/CVaR can replace it behind this interface
without touching the rest of the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from ..config.models import PropFirmRules
from ..core.types import Account, Signal


class Sizer(ABC):
    """Base class for position sizers."""

    @abstractmethod
    def size(self, signal: Signal, account: Account, rules: PropFirmRules) -> Decimal:
        """Return the position size (units/contracts) for ``signal``.

        Implementations MUST be fail-safe: when inputs are degenerate (zero risk
        per unit, exhausted drawdown buffer, etc.) they return ``Decimal("0")``
        rather than raising, so the system never over-sizes by accident.
        """
        raise NotImplementedError
