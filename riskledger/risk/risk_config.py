"""Tunable risk overrides: vol-targeting sizing + a profit-lock exit.

A single immutable knob-bag threaded through an evaluation engine, so the same
measurement machinery can be run *with* or *without* two risk overrides:

* **vol-targeting** (:class:`~riskledger.risk.position_sizer.VolTargetSizer`) —
  scale per-trade size inversely to the trade's own volatility (its stop
  distance) relative to a rolling baseline, so the fleet de-risks into volatile
  regimes where the ~$2.5k give-back excursions cluster.
* **profit-lock** — once an evaluation account reaches (a fraction of) its profit
  target, bank the pass: flatten and stop opening new trades for the rest of the
  window, so a peaked-then-gave-back window converts from a kill into a pass. This
  package carries the flag; acting on it is the execution loop's job.

The default (both ``False``) reproduces the prior behaviour exactly, so passing
``RiskConfig()`` (or ``None``) anywhere is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..firms.firm_registry import FirmConfig


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Optional vol-targeting + profit-lock overrides (defaults = prior behaviour)."""

    # --- sizer selection ---
    sizer: str = "fixed"
    """Which sizer to build: ``"fixed"`` (default), ``"vol_stop"`` (scale by the
    strategy's stop distance, :class:`VolTargetSizer`), ``"vol_realized"`` (Carver
    realized-return-vol targeting, :class:`RealizedVolSizer`) or ``"bold"``
    (Dubins-Savage barrier-proportional bold play, :class:`BoldPlaySizer`)."""
    boldness: Decimal = Decimal("0.25")
    """For ``sizer="bold"``: the fraction of the distance-to-drawdown-floor risked
    per trade. Higher = bolder = closer to the gambler's-ruin pass-rate ceiling but
    more variance; clamped to (0, 1) so one stop-out can't reach the floor."""
    vol_target: bool = False
    """Legacy boolean toggle, kept for backward compat: ``True`` selects the
    stop-distance :class:`VolTargetSizer` (equivalent to ``sizer="vol_stop"``)."""
    vol_ewma_span: int = 20
    """EWMA span (in signals) for the rolling baseline risk/return measure."""
    vol_scalar_floor: Decimal = Decimal("0.5")
    vol_scalar_ceil: Decimal = Decimal("1.5")
    """Clamp on the ``baseline / current`` size scalar so the lever stays modest."""

    # --- profit-lock (exit path) ---
    profit_lock: bool = False
    profit_lock_at: Decimal = Decimal("1.0")
    """Arm the lock once profit reaches this fraction of the phase's profit target."""

    @property
    def any_active(self) -> bool:
        """True iff some override changes behaviour (else this is a no-op)."""
        return self.sizer != "fixed" or self.vol_target or self.profit_lock


@dataclass(frozen=True, slots=True)
class PhaseRiskPolicy:
    """Phase-dependent risk posture: aggressive to PASS, conservative to SURVIVE.

    The operating model differs by phase. In an **evaluation** the goal is to *beat
    the profit target* — push for it (a more aggressive, target-seeking sizer), and
    bank the pass the moment the target is hit (profit-lock). Once **funded** there
    is no target, only the account to keep alive — the goal becomes *consistency*:
    a conservative sizer that protects the account and extracts payouts steadily.

    This pairs the two :class:`RiskConfig` postures so the engine can apply the
    right one per phase instead of a single uniform config.
    """

    evaluation: RiskConfig
    """Posture while passing an eval phase (EVAL_F1/F2): target-focused."""
    funded: RiskConfig
    """Posture once funded: conservative / consistency."""

    @classmethod
    def for_drawdown(cls, drawdown_type: str) -> PhaseRiskPolicy:
        """Drawdown-type-MATCHED posture: the evaluation sizer chosen per DD structure.

        The rationale is theoretical, not empirical. Reaching a profit target before a
        loss limit is a barrier problem, and for a bettor with roughly zero edge
        Dubins & Savage show that *bold* play -- staking a fraction of the distance to
        the barrier -- maximises the probability of reaching the target first, where
        timid many-small-bets play minimises it.

        * **trailing** DD -- bold play at a moderate ``boldness`` (0.25). The floor
          ratchets up as equity rises, which caps how much boldness can help.
        * **static** / **eod** DD -- ``fixed`` sizing. Against a floor that does not
          move, staking a fraction of the distance to it over-risks rather than
          exploiting the barrier.

        The funded leg is conservative (``fixed`` plus profit-lock) regardless of
        structure: its objective is survival, not reaching a barrier.

        Honesty note: an empirical comparison of these postures was run in the
        original research platform, but on an engine later found to charge no costs
        and not to enforce overnight-holding rules. Its numbers are therefore not
        reproduced here, and the choice should be re-measured before it is relied on.
        """
        evaluation = (RiskConfig(sizer="bold", boldness=Decimal("0.25"),
                                 profit_lock=True)
                      if drawdown_type == "trailing"
                      else RiskConfig(sizer="fixed", profit_lock=True))
        return cls(evaluation=evaluation,
                   funded=RiskConfig(sizer="fixed", profit_lock=True))

    @classmethod
    def for_firm(cls, firm: FirmConfig) -> PhaseRiskPolicy:
        """DD-matched posture for ``firm`` (reads its entry phase's drawdown type)."""
        entry_phase = firm.phases[0].phase
        return cls.for_drawdown(firm.phase(entry_phase).rules.drawdown_type)

    @classmethod
    def default(cls) -> PhaseRiskPolicy:
        """The trailing-drawdown posture: bold evaluation, conservative funded.

        Equal to :meth:`for_drawdown` ``("trailing")``. For a static or end-of-day
        firm use :meth:`for_firm`, which selects ``fixed`` sizing instead.
        """
        return cls.for_drawdown("trailing")
