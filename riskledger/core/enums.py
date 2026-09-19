"""Core state enums shared across the platform.

All enums are ``str``-based so they serialize cleanly to YAML/JSON and to the
append-only audit log.
"""

from __future__ import annotations

from enum import StrEnum


class Direction(StrEnum):
    """Trade/position direction."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        """+1 for LONG, -1 for SHORT (useful for PnL math)."""
        return 1 if self is Direction.LONG else -1


class SignalState(StrEnum):
    """Lifecycle of a trading signal."""

    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class AccountStatus(StrEnum):
    """Operational status of an account."""

    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"


class DrawdownType(StrEnum):
    """How a prop firm measures the drawdown limit."""

    STATIC = "static"
    TRAILING = "trailing"
    EOD = "eod"


class AccountPhase(StrEnum):
    """Phase of a prop firm account/challenge."""

    EVAL_F1 = "EVAL_F1"
    EVAL_F2 = "EVAL_F2"
    FUNDED = "FUNDED"
    SCALING = "SCALING"
    FAILED = "FAILED"


class RiskEventType(StrEnum):
    """Types of risk events recorded in the audit trail."""

    ALERT_70 = "ALERT_70"
    BLOCK_80 = "BLOCK_80"
    KILL_90 = "KILL_90"
    VELOCITY = "VELOCITY"
    CONSEC_LOSS = "CONSEC_LOSS"


class OrderType(StrEnum):
    """Supported order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
