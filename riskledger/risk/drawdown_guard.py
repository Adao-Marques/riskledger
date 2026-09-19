"""Polling drawdown guard — turns equity into discrete risk levels.

The :class:`DrawdownGuard` is the *online* counterpart to the pure math in
:mod:`riskledger.risk.drawdown`. It is polled with the account's live equity and
returns a :class:`GuardAssessment` describing how close the account is to a
breach across three independent vectors (total / daily / session), how badly the
worst one is consumed, and what position-sizing multiplier should apply.

Design rules baked in:

* **Decimal everywhere.** No equity or fraction is ever a ``float``.
* **Worst vector wins.** The assessment reflects the single most-consumed limit,
  so one healthy vector can never mask a dangerous one.
* **High-watermark semantics depend on the drawdown type.** Trailing ratchets the
  HWM on every tick; static never trails; ``eod`` only ratchets at end of day.
* **Fail-safe.** :meth:`DrawdownGuard.assess` never raises; any unexpected error
  collapses to ``KILL`` (the safe direction), and a degenerate buffer yields a
  fully-consumed fraction via the underlying pure functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from ..config.models import PropFirmRules
from ..core.enums import DrawdownType
from ..core.types import Account
from .drawdown import (
    consumed_fraction,
    daily_consumed_fraction,
    total_consumed_fraction,
)

ZERO = Decimal("0")
HALF = Decimal("0.5")
ONE = Decimal("1")


class GuardLevel(StrEnum):
    """Discrete risk levels emitted by the guard, in increasing severity."""

    OK = "OK"
    ALERT = "ALERT"
    BLOCK = "BLOCK"
    KILL = "KILL"


@dataclass(frozen=True, slots=True)
class GuardAssessment:
    """The outcome of a single drawdown assessment.

    Attributes:
        level: The discrete guard level (worst vector wins).
        worst_fraction: The largest consumed fraction across all vectors
            (0 = healthy, 1 = at the limit floor, >1 = breached).
        triggering_vector: Name of the vector that produced ``worst_fraction``
            (``"total"``, ``"daily"`` or ``"session"``).
        sizing_multiplier: Factor to apply to position size: 1 at OK, 0.5 at
            ALERT (cut sizing in half), 0 at BLOCK/KILL (no new risk).
    """

    level: GuardLevel
    worst_fraction: Decimal
    triggering_vector: str
    sizing_multiplier: Decimal


def _sizing_for(level: GuardLevel) -> Decimal:
    """Map a guard level to its position-sizing multiplier."""
    if level is GuardLevel.OK:
        return ONE
    if level is GuardLevel.ALERT:
        return HALF
    return ZERO


class DrawdownGuard:
    """Polls live equity and classifies drawdown risk into discrete levels.

    The guard is stateless beyond the :class:`PropFirmRules` it is built with;
    all evolving state (equity, high-watermark) lives on the
    :class:`~riskledger.core.types.Account` passed to each call.
    """

    def __init__(self, rules: PropFirmRules) -> None:
        """Build a guard for one prop firm phase's rule set."""
        self.rules = rules

    def _level_for(self, fraction: Decimal) -> GuardLevel:
        """Map a consumed fraction to a guard level using the rule thresholds."""
        if fraction >= self.rules.kill_switch_threshold:
            return GuardLevel.KILL
        if fraction >= self.rules.block_threshold:
            return GuardLevel.BLOCK
        if fraction >= self.rules.alert_threshold:
            return GuardLevel.ALERT
        return GuardLevel.OK

    def assess(
        self,
        account: Account,
        *,
        day_start_equity: Decimal,
        session_start_equity: Decimal | None = None,
    ) -> GuardAssessment:
        """Assess the account against all drawdown vectors and pick the worst.

        Three consumed fractions are computed:

        * ``total``   — :func:`total_consumed_fraction` (always evaluated).
        * ``daily``   — :func:`daily_consumed_fraction` (0 if no daily rule).
        * ``session`` — :func:`consumed_fraction` against a per-session floor of
          ``session_start_equity - max_daily_loss``; only evaluated when both a
          ``session_start_equity`` and a daily rule are present, else 0.

        The vector with the largest consumed fraction determines the returned
        level and sizing multiplier.

        Fail-safe: this method never raises. Any unexpected error collapses to a
        ``KILL`` assessment (the conservative direction).
        """
        try:
            equity = account.current_equity

            vectors: list[tuple[str, Decimal]] = [
                ("total", total_consumed_fraction(self.rules, account)),
            ]

            if self.rules.max_daily_loss is not None:
                vectors.append(
                    ("daily", daily_consumed_fraction(self.rules, equity, day_start_equity))
                )
                if session_start_equity is not None:
                    session_floor = session_start_equity - self.rules.max_daily_loss
                    vectors.append(
                        (
                            "session",
                            consumed_fraction(
                                equity, session_floor, self.rules.max_daily_loss
                            ),
                        )
                    )

            triggering_vector, worst_fraction = max(vectors, key=lambda item: item[1])
            level = self._level_for(worst_fraction)
            return GuardAssessment(
                level=level,
                worst_fraction=worst_fraction,
                triggering_vector=triggering_vector,
                sizing_multiplier=_sizing_for(level),
            )
        except Exception:  # noqa: BLE001 — fail-safe: any error means KILL.
            return GuardAssessment(
                level=GuardLevel.KILL,
                worst_fraction=ONE,
                triggering_vector="error",
                sizing_multiplier=ZERO,
            )

    def on_tick(
        self,
        account: Account,
        equity: Decimal,
        *,
        day_start_equity: Decimal,
        session_start_equity: Decimal | None = None,
    ) -> GuardAssessment:
        """Update the account with a new equity reading, then assess.

        High-watermark handling per drawdown type:

        * ``trailing`` — :meth:`Account.update_equity` ratchets the HWM up on
          every tick (intraday trailing).
        * ``static`` / ``eod`` — only ``current_equity`` is set; the HWM is left
          untouched intraday. For ``eod`` the HWM is ratcheted later via
          :meth:`end_of_day`.

        Fail-safe: never raises (delegates to :meth:`assess`).
        """
        if self.rules.drawdown_type is DrawdownType.TRAILING:
            account.update_equity(equity)
        else:
            account.current_equity = equity
        return self.assess(
            account,
            day_start_equity=day_start_equity,
            session_start_equity=session_start_equity,
        )

    def on_bar(
        self,
        account: Account,
        *,
        peak_equity: Decimal,
        trough_equity: Decimal,
        day_start_equity: Decimal,
        session_start_equity: Decimal | None = None,
    ) -> GuardAssessment:
        """Assess one bar against the intra-bar equity extremes.

        Bar data hides the intra-bar path, so a close-only reading both
        under-ratchets the trailing high-watermark (which really follows the
        intra-bar *peak* equity) and can miss a breach that happened at the
        intra-bar *trough* even though price recovered by the close. This method
        reconstructs both faithfully:

        * **HWM ratchet** — for ``trailing`` the HWM is ratcheted UP to
          ``peak_equity`` (the favourable intra-bar extreme), so the trailing
          floor tightens to where the account actually peaked during the bar.
          ``static`` never ratchets; ``eod`` is left to :meth:`end_of_day`.
        * **Breach assessment** — the drawdown is then assessed at
          ``trough_equity`` (the adverse intra-bar extreme): ``current_equity``
          is set to the trough and the existing :meth:`assess` is run. A kill
          therefore fires if the intra-bar low breached the floor even when the
          close recovers above it.

        For a flat account ``peak_equity == trough_equity == close equity`` so
        the behaviour collapses to the close-only path. Fail-safe: never raises
        (delegates to :meth:`assess`).
        """
        if self.rules.drawdown_type is DrawdownType.TRAILING:
            account.update_equity(peak_equity)
        account.current_equity = trough_equity
        return self.assess(
            account,
            day_start_equity=day_start_equity,
            session_start_equity=session_start_equity,
        )

    def end_of_day(self, account: Account,
                   closing_equity: Decimal | None = None) -> None:
        """Ratchet the high-watermark at the close for ``eod`` accounts.

        For ``eod`` drawdown the HWM moves once per day, at the close, to the day's
        CLOSING equity (still up-only via :meth:`Account.update_equity`). No-op for
        ``static`` and ``trailing``.

        ``closing_equity`` must be passed by any caller driving bars through
        :meth:`on_bar`. That method leaves ``account.current_equity`` at the bar's
        intra-bar TROUGH -- correct for testing a breach, wrong for ratcheting --
        so falling back to it under-ratchets the HWM by the last bar's
        close-minus-trough whenever the day closed above the prior watermark. The
        floor then sits lower than the firm's would, which is optimistic: the
        account survives drawdowns that would really have killed it.
        """
        if self.rules.drawdown_type is DrawdownType.EOD:
            account.update_equity(
                account.current_equity if closing_equity is None else closing_equity
            )
