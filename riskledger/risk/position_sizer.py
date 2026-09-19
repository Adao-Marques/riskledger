"""Position sizing — Fixed Fractional with a cascading Triple Cap.

The most restrictive cap always wins, so a returned size can never risk more
than ``per_trade_cap`` of equity. Fail-safe: any degenerate input returns zero.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from ..config.models import PropFirmRules
from ..core.types import Account, Signal
from .base import Sizer
from .drawdown import total_loss_floor
from .risk_config import RiskConfig

ZERO = Decimal("0")
ONE = Decimal("1")


class FixedFractionalSizer(Sizer):
    """Risk a fixed fraction of equity per trade, bounded by a Triple Cap.

    The three caps (in account currency), most restrictive wins:

    1. ``per_trade_cap`` × equity — hard 1% risk-per-trade ceiling.
    2. ``dd_buffer_cap_fraction`` × remaining total-drawdown buffer — never risk
       more than 10% of what's left before the drawdown floor.
    3. ``max_contracts`` — an absolute contract ceiling.

    The base target risk is ``risk_fraction`` × equity (0.5%); the per-trade cap
    (1%) is a strictly larger hard ceiling, guaranteeing no trade exceeds 1%.
    """

    def __init__(
        self,
        *,
        risk_fraction: Decimal = Decimal("0.005"),
        per_trade_cap: Decimal = Decimal("0.01"),
        dd_buffer_cap_fraction: Decimal = Decimal("0.10"),
        max_contracts: int = 10,
        point_value: Decimal = Decimal("1"),
    ) -> None:
        self.risk_fraction = risk_fraction
        self.per_trade_cap = per_trade_cap
        self.dd_buffer_cap_fraction = dd_buffer_cap_fraction
        self.max_contracts = Decimal(max_contracts)
        self.point_value = point_value

    def size(self, signal: Signal, account: Account, rules: PropFirmRules) -> Decimal:
        """Return whole contracts to trade (Decimal), never exceeding 1% risk."""
        equity = account.current_equity
        risk_per_unit = signal.risk_per_unit
        risk_per_contract = risk_per_unit * self.point_value

        # Fail-safe guards: any degenerate input -> no position.
        if risk_per_contract <= ZERO or equity <= ZERO or self.point_value <= ZERO:
            return ZERO

        remaining_buffer = account.current_equity - total_loss_floor(rules, account)
        if remaining_buffer <= ZERO:
            return ZERO

        base_budget = self._target_fraction(signal) * equity
        cap_per_trade = self.per_trade_cap * equity
        cap_buffer = self.dd_buffer_cap_fraction * remaining_buffer

        budget = min(base_budget, cap_per_trade, cap_buffer)
        contracts = (budget / risk_per_contract).to_integral_value(rounding=ROUND_DOWN)
        contracts = min(contracts, self.max_contracts)
        return contracts if contracts > ZERO else ZERO

    def _target_fraction(self, signal: Signal) -> Decimal:
        """Risk fraction of equity to target for ``signal`` (the base 0.5%).

        A hook so vol-targeting can scale the base risk per-signal without
        duplicating the Triple Cap. The base sizer ignores the signal.
        """
        return self.risk_fraction


class VolTargetSizer(FixedFractionalSizer):
    """Fixed-Fractional + a volatility scalar: size down in volatile regimes.

    Each signal's *stop distance* (``risk_per_unit``) is a direct, per-trade
    volatility reading. This sizer keeps an EWMA baseline of the per-contract
    risk seen so far and scales the base risk fraction by ``baseline / current``
    (clamped). So when this trade's volatility is above the recent norm the size
    is cut; when below, it is raised — bounded — toward constant *targeted*
    excursion rather than constant risk-to-stop. With steady volatility the
    scalar is ~1 and the behaviour is identical to the base sizer.

    The baseline is per-instance, so each fresh evaluation window (a fresh stack)
    starts its own baseline — windows stay independent.
    """

    def __init__(
        self,
        *,
        vol_ewma_span: int = 20,
        vol_scalar_floor: Decimal = Decimal("0.5"),
        vol_scalar_ceil: Decimal = Decimal("1.5"),
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._alpha = Decimal(2) / Decimal(vol_ewma_span + 1)
        self._floor = vol_scalar_floor
        self._ceil = vol_scalar_ceil
        self._baseline: Decimal | None = None  # EWMA of risk-per-contract

    def _target_fraction(self, signal: Signal) -> Decimal:
        rpc = signal.risk_per_unit * self.point_value
        if rpc <= ZERO:
            return self.risk_fraction
        if self._baseline is None:
            # First signal: seed the baseline, no scaling (nothing to compare to).
            self._baseline = rpc
            return self.risk_fraction
        baseline = self._baseline
        self._baseline = self._alpha * rpc + (ONE - self._alpha) * baseline
        scalar = baseline / rpc
        if scalar < self._floor:
            scalar = self._floor
        elif scalar > self._ceil:
            scalar = self._ceil
        return self.risk_fraction * scalar


class RealizedVolSizer(FixedFractionalSizer):
    """Realized-return-vol **regime scalar** for the position size (Decimal).

    Where :class:`VolTargetSizer` scales by each signal's *stop distance* (the
    strategy's chosen risk-per-unit), this sizer scales by a measure of the
    instrument's **realized return volatility** relative to its own recent norm,
    cutting size when this trade's vol runs above the EWMA baseline and raising it
    (bounded) when below. The intent is Carver-style de-risking into volatile
    regimes (*Systematic Trading* / *Leveraged Trading*).

    HONESTY CAVEAT (2026-06-05 methodology audit): this is a *relative regime
    scalar*, NOT a true Carver volatility target. A true vol target anchors to an
    **absolute** annualized risk level (``target_vol / realized_vol``) so the
    position's P&L vol is held constant; here there is only a bounded ratio to the
    EWMA baseline (clamped ``[floor, ceil]``), so past the clamp risk is NOT held
    constant. Moreover the self-estimate uses the squared **entry-to-entry**
    return as the vol proxy, but opening-range entries are sparse (~1/day) and
    irregularly spaced, so that proxy is noisy and not a proper realized vol. The
    statistically-correct input is a **bar-level ATR / fixed-interval return vol**
    fed via ``signal.metadata["realized_vol"]`` (or ``"atr"``), which overrides
    the self-estimate — wiring strategies to populate it is tracked debt. Treat
    the self-estimate path as a regime heuristic, not a calibrated vol target.

    Sizing: the scalar is ``clamp(sigma_baseline / sigma_current, floor, ceil)``
    where ``sigma_baseline`` is the EWMA-so-far. With **constant** vol the baseline
    equals the current reading, so the scalar is exactly 1 and the size equals the
    base fixed-fractional sizer — no surprise change.

    **Why fixed-fractional, not full-Kelly/Thorp.** Kelly maximises long-run
    growth, which is the wrong objective here: the binding constraint is the prop
    firm's trailing-drawdown floor (a hard kill), not growth-optimality. We
    therefore target a small *fixed* fraction (the base 0.5%, hard-capped at 1%
    by the Triple Cap), i.e. a tiny fraction of even a conservative Kelly bet —
    survival first, growth a distant second.

    Fail-safe: a degenerate vol estimate (``sigma <= 0``) or fewer than two
    observations falls back to the base ``risk_fraction`` (never over-sizes). The
    EWMA is per-instance so each fresh evaluation window stays independent.
    """

    def __init__(
        self,
        *,
        vol_ewma_span: int = 20,
        vol_scalar_floor: Decimal = Decimal("0.5"),
        vol_scalar_ceil: Decimal = Decimal("1.5"),
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._alpha = Decimal(2) / Decimal(vol_ewma_span + 1)
        self._floor = vol_scalar_floor
        self._ceil = vol_scalar_ceil
        self._prev_entry: Decimal | None = None  # last entry price seen
        self._baseline: Decimal | None = None  # EWMA of ret^2 (variance proxy)
        self._n_obs = 0  # number of squared-return observations folded in

    @staticmethod
    def _sigma(variance: Decimal) -> Decimal:
        """Decimal sqrt of a variance, guarding the degenerate/negative case."""
        if variance <= ZERO:
            return ZERO
        return variance.sqrt()

    def _override_vol(self, signal: Signal) -> Decimal | None:
        """A caller-supplied vol estimate from metadata, if present and positive."""
        meta = signal.metadata
        raw = meta.get("realized_vol", meta.get("atr"))
        if raw is None:
            return None
        vol = raw if isinstance(raw, Decimal) else Decimal(str(raw))
        return vol if vol > ZERO else None

    def _target_fraction(self, signal: Signal) -> Decimal:
        # One variance reading per signal, from a caller-supplied vol (preferred,
        # e.g. a bar-level ATR) or, failing that, the squared entry-to-entry
        # return. Both paths fold into the SAME EWMA so the override path actually
        # scales (it previously left the baseline None -> always scalar 1).
        override = self._override_vol(signal)
        if override is not None:
            var = override * override
        else:
            entry = signal.entry_price
            prev = self._prev_entry
            self._prev_entry = entry
            if prev is None or prev <= ZERO:
                return self.risk_fraction  # no prior entry -> no return yet
            ret = (entry - prev) / prev
            var = ret * ret
        if self._baseline is None:
            # Seed the EWMA; a single observation is not a vol estimate.
            self._baseline = var
            self._n_obs = 1
            return self.risk_fraction
        baseline = self._baseline
        self._baseline = self._alpha * var + (ONE - self._alpha) * baseline
        self._n_obs += 1
        if self._n_obs < 2:
            return self.risk_fraction
        sigma_current = self._sigma(var)
        sigma_baseline = self._sigma(baseline)
        if sigma_current <= ZERO or sigma_baseline <= ZERO:
            return self.risk_fraction
        scalar = sigma_baseline / sigma_current
        if scalar < self._floor:
            scalar = self._floor
        elif scalar > self._ceil:
            scalar = self._ceil
        return self.risk_fraction * scalar


class BoldPlaySizer(FixedFractionalSizer):
    """Bet a fraction of the distance to the drawdown floor — Dubins-Savage bold play.

    A prop evaluation is not a forecasting problem, it is a **barrier** problem:
    reach ``+profit_target`` before ``−drawdown``. For a bettor with ~zero edge,
    Dubins & Savage (*How to Gamble If You Must*, 1965) prove that **bold play** —
    few large bets — *maximizes* the probability of reaching a target before ruin,
    while **timid play** (many small bets) *minimizes* it: every extra step pays the
    cost spread again, so a slightly-negative-drift walk of many small trades is
    dragged to ruin. The driftless ceiling is the gambler's-ruin value
    ``D / (T + D)`` (~45% for Apex 50k), versus the ~2-4% the timid 0.5%-of-equity
    sizer actually achieves.

    This sizer therefore risks ``boldness × (equity − drawdown_floor)`` per trade —
    a fraction of the **survival cushion**, not of equity. It is large when there is
    room and shrinks toward zero as the floor nears, so a single stop-out cannot
    cross the floor (``boldness < 1``). Paired with the profit-lock (bank the target,
    gated by ``min_trading_days``) it is the full barrier-optimal play: bet boldly
    toward the target, then stop.

    The firms defend against exactly this with the trailing drawdown + the
    consistency / min-trading-day rules (the engine enforces them as compliance
    gates), so the realized pass rate lands **between** the timid 4% and the
    driftless ``D/(T+D)`` ceiling — the measurement quantifies where. ``max_contracts``
    and the non-positive-buffer guard still apply (survival first).
    """

    def __init__(self, *, boldness: Decimal = Decimal("0.25"), **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        # Clamp to (0, 1): a single trade must never be able to risk the whole
        # cushion (which would put the stop at the floor — one loss = ruin).
        self.boldness = min(max(boldness, Decimal("0.01")), Decimal("0.95"))

    def size(self, signal: Signal, account: Account, rules: PropFirmRules) -> Decimal:
        """Whole contracts risking ``boldness`` × the distance to the drawdown floor."""
        equity = account.current_equity
        risk_per_contract = signal.risk_per_unit * self.point_value
        if risk_per_contract <= ZERO or equity <= ZERO or self.point_value <= ZERO:
            return ZERO
        remaining_buffer = equity - total_loss_floor(rules, account)
        if remaining_buffer <= ZERO:
            return ZERO
        budget = self.boldness * remaining_buffer
        contracts = (budget / risk_per_contract).to_integral_value(rounding=ROUND_DOWN)
        contracts = min(contracts, self.max_contracts)
        return contracts if contracts > ZERO else ZERO


def build_sizer(
    risk_config: RiskConfig | None,
    *,
    max_contracts: int,
    point_value: Decimal,
) -> FixedFractionalSizer:
    """Construct the sizer the ``risk_config`` selects (vol-targeting or base)."""
    if risk_config is not None:
        if risk_config.sizer == "bold":
            return BoldPlaySizer(
                max_contracts=max_contracts, point_value=point_value,
                boldness=risk_config.boldness)
        if risk_config.sizer == "vol_realized":
            return RealizedVolSizer(
                max_contracts=max_contracts,
                point_value=point_value,
                vol_ewma_span=risk_config.vol_ewma_span,
                vol_scalar_floor=risk_config.vol_scalar_floor,
                vol_scalar_ceil=risk_config.vol_scalar_ceil,
            )
        if risk_config.sizer == "vol_stop" or risk_config.vol_target:
            return VolTargetSizer(
                max_contracts=max_contracts,
                point_value=point_value,
                vol_ewma_span=risk_config.vol_ewma_span,
                vol_scalar_floor=risk_config.vol_scalar_floor,
                vol_scalar_ceil=risk_config.vol_scalar_ceil,
            )
    return FixedFractionalSizer(max_contracts=max_contracts, point_value=point_value)


def min_fundable_balance(
    risk_per_contract: Decimal, *, risk_fraction: Decimal = Decimal("0.005")
) -> Decimal:
    """Smallest account balance that can fund >=1 contract at ``risk_fraction``.

    ``risk_per_contract`` is the $ risk to the stop of one contract
    (``stop_distance * point_value``). Below this balance the Triple Cap floors
    the size to 0 contracts and the account never trades — a *fundability* limit,
    not an absence of edge (the 2026-06-05 sizing artifact). Returns 0 for a
    degenerate (non-positive) risk.
    """
    if risk_per_contract <= ZERO:
        return ZERO
    return risk_per_contract / risk_fraction
