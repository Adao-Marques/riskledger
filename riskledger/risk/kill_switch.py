"""The Kill Switch — the atomic, idempotent last line of defence.

When triggered it runs an irrevocable sequence:

    Detect -> Lock -> Cancel -> Flatten -> Verify -> Log -> Alert -> Suspend

It never assumes a close without broker confirmation, and it is **fail-safe**:
if any broker call raises, the switch still suspends the account and records the
event, reporting ``verified=False``. The account always ends suspended.

**Atomicity.** The latch is a real mutex, not a check-then-set on a bool. The
design shares exactly ONE switch per account across every caller that can decide to
kill it, so if two callers — two threads, an
engine and a worker, a watchdog and a guard — reach :meth:`KillSwitch.trigger`
at once, exactly one of them may run the cancel/flatten/verify sequence. The
others block on the winner and receive the winner's :class:`KillResult`. The
only machinery used is :mod:`threading` from the stdlib: the safety path must
still work with the DB, Redis and the network down.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..audit.compliance_log import ComplianceLog
from ..core.clock import utcnow
from ..core.enums import AccountStatus, RiskEventType
from ..core.types import Account, Fill
from ..execution.base import BrokerAdapter

AlertFn = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class KillResult:
    """Outcome of a kill-switch activation."""

    triggered_at: datetime
    reason: str
    cancelled: int
    fills: list[Fill] = field(default_factory=list)
    verified: bool = False
    positions_remaining: int = 0


class KillSwitch:
    """Atomic, idempotent kill switch bound to one account + broker."""

    def __init__(
        self,
        broker: BrokerAdapter,
        account: Account,
        audit: ComplianceLog | None = None,
        *,
        alert: AlertFn | None = None,
        max_verify_attempts: int = 3,
    ) -> None:
        self._broker = broker
        self._account = account
        self._audit = audit
        self._alert = alert
        self._max_verify_attempts = max(1, max_verify_attempts)
        # ``_latch`` guards the check-and-set of ``_firing``/``_result`` ONLY; the
        # broker sequence runs outside it so a hung broker cannot wedge every
        # caller inside the mutex. ``_done`` is what losers wait on instead, and
        # the winner always sets it in a ``finally``.
        self._latch = threading.Lock()
        self._done = threading.Event()
        self._firing = False
        self._owner: int | None = None
        self._result: KillResult | None = None

    @property
    def triggered(self) -> bool:
        """True once the switch has fired (or is mid-firing, and latches)."""
        return self._firing or self._result is not None

    def trigger(self, reason: str) -> KillResult:
        """Fire the switch. Idempotent: a second call returns the first result.

        Never raises: broker failures degrade to ``verified=False`` but the
        account is always suspended and the event always logged.
        """
        # Claim the latch under a mutex BEFORE any side effect, so that of N
        # concurrent callers exactly one runs the cancel/flatten sequence. A
        # check-then-set on a plain bool is not enough: two threads can both read
        # ``_firing is False`` and both flatten, i.e. double orders on a killed
        # account — the exact disaster the single-latch design exists to prevent.
        me = threading.get_ident()
        with self._latch:
            if self._result is not None:  # already fired — do not act again.
                return self._result
            if self._firing and self._owner == me:
                # Re-entrant call from the firing thread itself (e.g. a broker
                # callback). It cannot wait on itself, so report "in flight".
                return KillResult(triggered_at=utcnow(), reason=reason, cancelled=0)
            claimed = not self._firing
            if claimed:
                self._firing = True
                self._owner = me
        if not claimed:
            # Another caller owns the fire: block until it has finished the whole
            # sequence (result latched, audited, account suspended) and hand back
            # ITS result, so every caller sees the same single kill.
            self._done.wait()
            done = self._result
            return (done if done is not None
                    else KillResult(triggered_at=utcnow(), reason=reason, cancelled=0))

        triggered_at = utcnow()
        cancelled = 0
        fills: list[Fill] = []
        verified = False
        remaining = -1
        try:
            cancelled = self._broker.cancel_all()
            fills.extend(self._broker.flatten())
            verified, remaining = self._verify_flat(fills)
        except Exception:  # noqa: BLE001 — fail-safe: suspend regardless of broker errors.
            verified = False

        result = KillResult(
            triggered_at=triggered_at,
            reason=reason,
            cancelled=cancelled,
            fills=fills,
            verified=verified,
            positions_remaining=max(remaining, 0),
        )
        try:
            self._result = result  # latch before side effects so it stays idempotent
            # An audit-sink failure (full disk, unwritable path) must not abort the
            # suspension either: the promise is that the account ALWAYS ends
            # suspended and trigger() never raises.
            with contextlib.suppress(Exception):
                self._log(result)
            self._notify(result)
            self._account.status = AccountStatus.KILL_SWITCH_TRIGGERED  # Suspend
        finally:
            # Release every waiter only once the account is actually suspended, so
            # no caller can observe a latched result on a live account.
            self._done.set()
        return result

    def _verify_flat(self, fills: list[Fill]) -> tuple[bool, int]:
        """Confirm positions are closed; re-flatten up to N attempts. No sleeping."""
        remaining = len(self._broker.get_positions())
        attempts = 0
        while remaining > 0 and attempts < self._max_verify_attempts:
            fills.extend(self._broker.flatten())
            remaining = len(self._broker.get_positions())
            attempts += 1
        return remaining == 0, remaining

    def _log(self, result: KillResult) -> None:
        if self._audit is None:
            return
        self._audit.append(
            RiskEventType.KILL_90,
            {
                "account_id": self._account.id,
                "reason": result.reason,
                "cancelled": result.cancelled,
                "fills": len(result.fills),
                "verified": result.verified,
                "positions_remaining": result.positions_remaining,
            },
            ts=result.triggered_at,
        )

    def _notify(self, result: KillResult) -> None:
        if self._alert is None:
            return
        # An alert-channel failure must never block suspension.
        with contextlib.suppress(Exception):
            self._alert(
                {
                    "event": "KILL_SWITCH",
                    "account_id": self._account.id,
                    "reason": result.reason,
                    "verified": result.verified,
                    "positions_remaining": result.positions_remaining,
                }
            )
