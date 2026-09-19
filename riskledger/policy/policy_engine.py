"""The PolicyEngine — the last barrier before an order reaches the broker.

Given a signal, the live account state and the prop firm's rules, the engine
decides whether the order may proceed. It is **100% rules-driven**: no firm is
hard-coded, every limit comes from :class:`PropFirmRules` and the shared pure
drawdown math, so the same engine governs Topstep, FTMO or any future firm.

Design tenets:

* **Risk before return** — when in doubt, reject. The engine never raises on
  degenerate input; a malformed signal is simply turned away.
* **Decimal everywhere** for equity/PnL/drawdown; **UTC** for every timestamp.
* Every decision is optionally recorded to an append-only, hash-chained
  :class:`~riskledger.audit.compliance_log.ComplianceLog`.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from ..audit.compliance_log import ComplianceLog
from ..config.models import PropFirmRules
from ..core.enums import AccountStatus, SignalState
from ..core.types import Account, Signal
from ..risk.drawdown import (
    daily_consumed_fraction,
    total_consumed_fraction,
    total_loss_floor,
)


class PolicyEngine:
    """Validates or rejects signals against an account's prop firm rules.

    The engine is stateless across calls; all state arrives through the
    arguments to :meth:`validate`. An optional :class:`ComplianceLog` makes the
    decision auditable.
    """

    def __init__(
        self,
        compliance_log: ComplianceLog | None = None,
        *,
        point_value: Decimal = Decimal("1"),
    ) -> None:
        """Create an engine, optionally wired to an audit log.

        :param compliance_log: when supplied, every validation and rejection is
            appended to the hash-chained log.
        :param point_value: dollar value of one full point of price movement for
            the traded instrument (e.g. 50 for ES). Used to convert a signal's
            per-unit price risk into a dollar figure for the worst-case-breach
            check; defaults to 1 (price units == dollars).
        """
        self._log = compliance_log
        self._point_value = point_value

    def validate(
        self,
        signal: Signal,
        account: Account,
        rules: PropFirmRules,
        *,
        day_start_equity: Decimal | None = None,
        now: datetime | None = None,
    ) -> Signal:
        """Return the signal VALIDATED, or REJECTED with a specific reason.

        Checks run in a fixed, fail-safe order; the first failing rule wins. The
        engine never raises on degenerate input — any unexpected condition
        resolves to a rejection rather than an exception.

        :param signal: the raw (PENDING) signal proposed by a strategy.
        :param account: the live account state (status, equity, high-watermark).
        :param rules: the prop firm rules for the account's current phase.
        :param day_start_equity: equity at the start of the trading day; required
            to evaluate the daily-loss rule. When absent, the daily rule is
            skipped (the total-drawdown rule still applies).
        :param now: unused reference time, accepted for interface symmetry with
            time-aware callers; the engine keys time checks off ``signal.ts``.
        :returns: a new :class:`Signal` in state ``VALIDATED`` or ``REJECTED``.
        """
        del now  # time checks key off the signal's own (UTC) timestamp

        reason = self._first_failure(
            signal, account, rules, day_start_equity=day_start_equity
        )
        if reason is not None:
            return self._reject(signal, account, rules, reason)
        return self._accept(signal, account, rules)

    def _first_failure(
        self,
        signal: Signal,
        account: Account,
        rules: PropFirmRules,
        *,
        day_start_equity: Decimal | None,
    ) -> str | None:
        """Return the first violated rule's reason, or None if all pass."""
        # 1. The account must be tradeable at all.
        if account.status is not AccountStatus.ACTIVE:
            return f"account not active: {account.status}"

        # 2. Total drawdown already at/over the block level.
        if total_consumed_fraction(rules, account) >= rules.block_threshold:
            return "total drawdown limit reached"

        # 3. This trade's worst case would breach the total floor. Risk is in
        #    dollars: price distance × size × point_value (e.g. ~50× for ES).
        if signal.suggested_size is not None:
            worst_loss = signal.risk_per_unit * signal.suggested_size * self._point_value
            if account.current_equity - worst_loss < total_loss_floor(rules, account):
                return "trade worst-case breaches total drawdown floor"

        # 4. Daily loss already at/over the block level (only if a daily rule
        #    exists and we know where the day started).
        if rules.max_daily_loss is not None and day_start_equity is not None:
            consumed = daily_consumed_fraction(
                rules, account.current_equity, day_start_equity
            )
            if consumed >= rules.block_threshold:
                return "daily loss limit reached"

        # 5. Past the firm's mandatory flat time (UTC time-of-day).
        if rules.mandatory_flat_utc is not None and signal.ts.timetz() >= (
            rules.mandatory_flat_utc.replace(tzinfo=signal.ts.tzinfo)
        ):
            return "after mandatory flat time"

        # 6. Trading is blacked out around scheduled news.
        if rules.news_blackout and signal.metadata.get("news_blackout") is True:
            return "news blackout"

        return None

    def _accept(self, signal: Signal, account: Account, rules: PropFirmRules) -> Signal:
        """Mark the signal VALIDATED and record the decision."""
        validated = signal.validated()
        self._record(SignalState.VALIDATED, validated, account, rules)
        return validated

    def _reject(
        self, signal: Signal, account: Account, rules: PropFirmRules, reason: str
    ) -> Signal:
        """Mark the signal REJECTED and record the decision with its reason."""
        rejected = signal.rejected(reason)
        self._record(SignalState.REJECTED, rejected, account, rules)
        return rejected

    def _record(
        self,
        state: SignalState,
        signal: Signal,
        account: Account,
        rules: PropFirmRules,
    ) -> None:
        """Append the decision to the compliance log, if one is configured."""
        if self._log is None:
            return
        event_type = (
            "SIGNAL_VALIDATED" if state is SignalState.VALIDATED else "SIGNAL_REJECTED"
        )
        payload = {
            "signal_id": signal.id,
            "strategy_id": signal.strategy_id,
            "account_id": account.id,
            "prop_firm": account.prop_firm,
            "symbol": signal.symbol,
            "direction": str(signal.direction),
            "state": str(signal.state),
            "rejection_reason": signal.rejection_reason,
            "suggested_size": str(signal.suggested_size)
            if signal.suggested_size is not None
            else None,
            "drawdown_type": str(rules.drawdown_type),
        }
        self._log.append(event_type, payload, ts=signal.ts)
