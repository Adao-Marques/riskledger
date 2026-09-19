"""Prop-account risk co-pilot — the live risk engine, pointed at a HUMAN's account.

It makes no claim to an edge -- the research this was extracted from found none. What
it does is compute, exactly and auditably, the numbers a firm's published rules imply
for an account. Given the firm rules and a snapshot of the account, it answers the
questions that keep an evaluation alive --

1. **Where is my liquidation line?** the total-drawdown floor + how much buffer is left.
2. **How big can I go right now?** the max contracts whose worst case still respects
   the floor, and a prudent size at the guard's current risk level.
3. **Am I compliant?** trading-day count, winning-day count, the consistency rule
   (no single day too large a share of total profit) and the evaluation deadline.
4. **How close am I to passing?** progress toward the profit target.

Every number is **traceable**: each field of :class:`CopilotReport` carries a
:class:`RuleTrace` naming the rule that produced it, the formula, the firm-config
values that fed it and where they came from. That is the difference between this and
a spreadsheet — you can always answer "why does it say that?".

Pure + ``Decimal``; reuses :class:`~riskledger.risk.drawdown_guard.DrawdownGuard` and the
shared drawdown math so the co-pilot's numbers are exactly what the live engine would
enforce. No I/O — feed it an :class:`AccountSnapshot` (from manual input or a read-only
broker pull) and read the :class:`CopilotReport`; feed it many and read a
:class:`FleetReport`.

**What this is not.** A risk and compliance calculator. It does not predict, promise or
improve trading results, it does not place orders, and it is only as correct as the firm
rules it is given (see ``riskledger/firms/configs/*.yaml``; always verify against your
firm's current rulebook).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from ..config.models import PropFirmRules
from ..core.enums import AccountPhase
from ..core.types import Account
from ..firms.firm_registry import FirmRegistry
from ..risk.drawdown import daily_loss_floor, total_loss_floor
from ..risk.drawdown_guard import DrawdownGuard

_ZERO = Decimal("0")
_ONE = Decimal("1")

VERDICTS = ("GO", "CAUTION", "STOP")
_SEVERITY = {"GO": 0, "CAUTION": 1, "STOP": 2}
_GUARD_ORDER = {"OK": 0, "ALERT": 1, "BLOCK": 2, "KILL": 3}


# --------------------------------------------------------------------------- traces


@dataclass(frozen=True, slots=True)
class RuleTrace:
    """Why one reported number has the value it has.

    Attributes:
        field: The :class:`CopilotReport` attribute this explains.
        value: The rendered value, as text (Decimals are not re-formatted here).
        rule: The prop-firm rule that produced it, in plain language.
        formula: The arithmetic actually evaluated.
        inputs: ``(name, value)`` pairs of every input that fed the formula.
        source: Where those inputs came from (firm config + phase, or the snapshot).
    """

    field: str
    value: str
    rule: str
    formula: str
    inputs: tuple[tuple[str, str], ...]
    source: str

    def render(self) -> str:
        """One-line human rendering: ``field = value  <- rule: formula [inputs]``."""
        ins = ", ".join(f"{k}={v}" for k, v in self.inputs)
        return f"{self.field} = {self.value}  <- {self.rule}: {self.formula}  [{ins}]"


@dataclass(frozen=True, slots=True)
class ComplianceCheck:
    """One pass/fail compliance line item, with the rule that defines it."""

    name: str
    ok: bool
    detail: str
    rule: str
    source: str
    blocking: bool = False
    """True when failing this check means the evaluation is already lost (not just
    'not yet cleared') — e.g. the evaluation deadline has passed."""


# ---------------------------------------------------------------------- input state


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """A point-in-time view of a prop account (what the trader/broker can observe).

    Only the first seven fields are required; the rest describe the account's identity,
    its compliance history and how the next trade should be sized.
    """

    firm_id: str
    initial_balance: Decimal
    current_equity: Decimal
    high_watermark: Decimal           # peak equity (drives trailing/eod floor)
    day_start_equity: Decimal
    point_value: Decimal              # $ per point of the instrument being sized
    days_traded: int = 0
    daily_pnls: tuple[Decimal, ...] = ()   # realised daily PnLs so far (consistency rule)
    # --- identity / fleet ---
    account_id: str = "account"
    label: str = ""                   # display name; defaults to "<firm> <account_id>"
    phase: AccountPhase = AccountPhase.EVAL_F1
    symbol: str = ""                  # instrument being sized (display only)
    # --- sizing defaults (a ``assess_account`` keyword overrides these) ---
    stop_distance: Decimal | None = None   # price distance entry -> stop
    risk_pct: Decimal = Decimal("0.01")    # fraction of equity risked per trade
    # --- evaluation clock ---
    days_elapsed: int = 0             # calendar/trading days since the eval started

    def display_label(self) -> str:
        """Human label for this account (falls back to ``firm_id account_id``)."""
        return self.label or f"{self.firm_id} {self.account_id}"


# --------------------------------------------------------------------------- output


@dataclass(frozen=True, slots=True)
class CopilotReport:
    """Everything the co-pilot derives from a snapshot + the firm rules."""

    # drawdown / liquidation line
    total_floor: Decimal
    distance_to_floor: Decimal
    buffer_consumed_pct: Decimal      # 0..1 (worst vector); >1 = breached
    guard_level: str                  # OK / ALERT / BLOCK / KILL
    sizing_multiplier: Decimal
    daily_floor: Decimal | None
    daily_distance: Decimal | None
    # target progress
    target_equity: Decimal | None
    target_remaining: Decimal | None
    target_pct: Decimal | None        # (equity-initial)/profit_target
    # sizing (for a given stop distance, if supplied)
    max_safe_contracts: int           # worst case still respects the floor
    prudent_contracts: int            # risk-capped + guard-scaled
    # compliance
    days_traded: int
    min_trading_days: int
    days_ok: bool
    largest_day_pct: Decimal | None
    consistency_cap: Decimal | None
    consistency_ok: bool
    can_pass_now: bool
    # verdict
    verdict: str                      # GO / CAUTION / STOP
    reasons: tuple[str, ...]
    # --- identity (defaults keep older positional construction working) ---
    account_id: str = "account"
    label: str = ""
    firm_id: str = ""
    firm_name: str = ""
    phase: str = AccountPhase.EVAL_F1.value
    symbol: str = ""
    equity: Decimal = _ZERO
    initial_balance: Decimal = _ZERO
    # --- extra risk / compliance detail ---
    triggering_vector: str = "total"  # which drawdown vector is worst
    open_pnl: Decimal = _ZERO         # equity - initial_balance
    risk_per_contract: Decimal | None = None
    withdrawable: Decimal | None = None
    winning_days: int = 0
    min_winning_days: int = 0
    winning_days_ok: bool = True
    days_elapsed: int = 0
    max_eval_days: int | None = None
    deadline_ok: bool = True
    checks: tuple[ComplianceCheck, ...] = ()
    traces: tuple[RuleTrace, ...] = ()

    def explain(self, field_name: str) -> RuleTrace | None:
        """Return the :class:`RuleTrace` for ``field_name`` (None if untraced)."""
        for t in self.traces:
            if t.field == field_name:
                return t
        return None

    def display_label(self) -> str:
        """Human label for this account."""
        return self.label or f"{self.firm_id} {self.account_id}".strip()

    def compliance_ok(self) -> bool:
        """True when every compliance check passes."""
        return all(c.ok for c in self.checks)


@dataclass(frozen=True, slots=True)
class FleetReport:
    """N accounts across M firms and phases, assessed together."""

    reports: tuple[CopilotReport, ...] = ()
    errors: tuple[tuple[str, str], ...] = field(default=())
    """``(account label, message)`` for snapshots that could not be assessed
    (unknown firm, or a phase the firm does not define). Never raises on one bad
    account — the rest of the fleet still reports."""

    def __len__(self) -> int:
        return len(self.reports)

    def counts(self) -> dict[str, int]:
        """Number of accounts at each verdict (all three keys always present)."""
        out = dict.fromkeys(VERDICTS, 0)
        for r in self.reports:
            out[r.verdict] = out.get(r.verdict, 0) + 1
        return out

    def worst(self) -> CopilotReport | None:
        """The account most at risk (worst verdict, then most buffer consumed)."""
        if not self.reports:
            return None
        return max(self.reports,
                   key=lambda r: (_SEVERITY.get(r.verdict, 0), r.buffer_consumed_pct))

    def fleet_verdict(self) -> str:
        """The worst verdict across the fleet (GO when empty)."""
        worst = self.worst()
        return worst.verdict if worst is not None else "GO"

    def total_equity(self) -> Decimal:
        """Sum of equity across the fleet."""
        return sum((r.equity for r in self.reports), _ZERO)

    def total_at_risk(self) -> Decimal:
        """Sum of the distance-to-floor across the fleet (capital still riskable)."""
        return sum((r.distance_to_floor for r in self.reports), _ZERO)

    def passable(self) -> tuple[CopilotReport, ...]:
        """Accounts that could be banked as a pass right now."""
        return tuple(r for r in self.reports if r.can_pass_now)


# ------------------------------------------------------------------------ internals


def _account(snap: AccountSnapshot) -> Account:
    acct = Account.open(snap.account_id, snap.firm_id, snap.initial_balance,
                        phase=snap.phase)
    acct.current_equity = snap.current_equity
    acct.high_watermark = max(snap.high_watermark, snap.initial_balance)
    return acct


def _consistency(snap: AccountSnapshot, rules: PropFirmRules) -> tuple[Decimal | None, bool]:
    """Largest winning day as a fraction of total profit, vs the firm's cap.

    Returns ``(largest_day_pct, ok)``. ``ok`` is True when there is no consistency
    rule, not enough data, or no profit yet (nothing to violate). A single day may
    not exceed ``consistency_max_day_pct`` of total profit on most static-DD firms.
    """
    cap = rules.consistency_max_day_pct
    if cap is None or not snap.daily_pnls:
        return None, True
    total = sum((p for p in snap.daily_pnls if p > _ZERO), _ZERO)
    if total <= _ZERO:
        return None, True
    largest = max(snap.daily_pnls)
    pct = largest / total
    return pct, pct <= cap


def _source(snap: AccountSnapshot, *fields: str) -> str:
    """Render where a rule's inputs came from: the firm config + phase, or the snapshot."""
    if not fields:
        return "account snapshot"
    return f"firm config '{snap.firm_id}' / phase {snap.phase.value}: " + ", ".join(
        f"rules.{f}" for f in fields)


def _d(value: Decimal | int | None) -> str:
    return "n/a" if value is None else str(value)


# --------------------------------------------------------------------------- public


def resolve_rules(registry: FirmRegistry, snap: AccountSnapshot) -> PropFirmRules:
    """Look up the rule set for ``snap``'s firm and phase (KeyError if absent)."""
    return registry.firm(snap.firm_id).phase(snap.phase).rules


def assess_account(snap: AccountSnapshot, rules: PropFirmRules, *,
                   stop_distance: Decimal | None = None,
                   risk_pct: Decimal | None = None,
                   firm_name: str = "") -> CopilotReport:
    """Assess ``snap`` against ``rules`` and return the full co-pilot report.

    ``stop_distance`` is the price distance from entry to stop for the trade being
    sized (defaults to ``snap.stop_distance``; sizing is skipped when both are absent).
    ``risk_pct`` is the fraction of equity the prudent size risks per trade (defaults
    to ``snap.risk_pct``, i.e. 1%), additionally capped never to let the worst case
    cross the floor and scaled by the guard's sizing multiplier.

    Pure and total: no I/O, and the guard it delegates to is itself fail-safe.
    """
    stop = snap.stop_distance if stop_distance is None else stop_distance
    risk = snap.risk_pct if risk_pct is None else risk_pct
    acct = _account(snap)
    equity = acct.current_equity
    traces: list[RuleTrace] = []

    # --- drawdown / liquidation line -------------------------------------------
    floor = total_loss_floor(rules, acct)
    distance = equity - floor
    guard = DrawdownGuard(rules).assess(acct, day_start_equity=snap.day_start_equity)

    dd = rules.drawdown_type.value
    formula: str
    ins: tuple[tuple[str, str], ...]
    if dd == "static":
        formula, ins = ("initial_balance - max_total_drawdown",
                        (("initial_balance", _d(snap.initial_balance)),
                         ("max_total_drawdown", _d(rules.max_total_drawdown))))
    else:
        formula = "high_watermark - max_total_drawdown"
        ins = (("high_watermark", _d(acct.high_watermark)),
               ("max_total_drawdown", _d(rules.max_total_drawdown)))
        if rules.trailing_locks_at_initial:
            formula = f"min({formula}, initial_balance)"
            ins = (*ins, ("initial_balance", _d(snap.initial_balance)))
    traces.append(RuleTrace("total_floor", _d(floor), f"total drawdown ({dd})", formula,
                            ins, _source(snap, "drawdown_type", "max_total_drawdown")))
    traces.append(RuleTrace("distance_to_floor", _d(distance),
                            "buffer left before liquidation",
                            "current_equity - total_floor",
                            (("current_equity", _d(equity)), ("total_floor", _d(floor))),
                            "account snapshot + total_floor above"))
    traces.append(RuleTrace(
        "buffer_consumed_pct", f"{guard.worst_fraction}",
        f"worst drawdown vector ({guard.triggering_vector})",
        "1 - (equity - floor) / max_total_drawdown",
        (("max_total_drawdown", _d(rules.max_total_drawdown)),
         ("triggering_vector", guard.triggering_vector)),
        _source(snap, "max_total_drawdown", "max_daily_loss")))
    traces.append(RuleTrace(
        "guard_level", guard.level.value, "DrawdownGuard thresholds",
        "OK < alert <= ALERT < block <= BLOCK < kill <= KILL",
        (("alert_threshold", _d(rules.alert_threshold)),
         ("block_threshold", _d(rules.block_threshold)),
         ("kill_switch_threshold", _d(rules.kill_switch_threshold)),
         ("buffer_consumed", f"{guard.worst_fraction}")),
        _source(snap, "alert_threshold", "block_threshold", "kill_switch_threshold")))
    traces.append(RuleTrace(
        "sizing_multiplier", _d(guard.sizing_multiplier), "guard-level size scaling",
        "OK->1, ALERT->0.5, BLOCK/KILL->0",
        (("guard_level", guard.level.value),), "DrawdownGuard (live engine)"))

    d_floor = daily_loss_floor(rules, snap.day_start_equity)
    d_distance = (equity - d_floor) if d_floor is not None else None
    if d_floor is not None:
        traces.append(RuleTrace(
            "daily_floor", _d(d_floor), "daily-loss limit",
            "day_start_equity - max_daily_loss",
            (("day_start_equity", _d(snap.day_start_equity)),
             ("max_daily_loss", _d(rules.max_daily_loss))),
            _source(snap, "max_daily_loss")))

    # --- target progress --------------------------------------------------------
    target_equity = target_remaining = target_pct = None
    if rules.profit_target is not None:
        target_equity = snap.initial_balance + rules.profit_target
        target_remaining = target_equity - equity
        target_pct = ((equity - snap.initial_balance) / rules.profit_target
                      if rules.profit_target > _ZERO else None)
        traces.append(RuleTrace(
            "target_equity", _d(target_equity), "phase profit target",
            "initial_balance + profit_target",
            (("initial_balance", _d(snap.initial_balance)),
             ("profit_target", _d(rules.profit_target))),
            _source(snap, "profit_target")))

    withdrawable = None
    if rules.payout_buffer is not None:
        raw = equity - snap.initial_balance - rules.payout_buffer
        withdrawable = raw if raw > _ZERO else _ZERO
        traces.append(RuleTrace(
            "withdrawable", _d(withdrawable), "payout safety net",
            "max(0, equity - initial_balance - payout_buffer)",
            (("payout_buffer", _d(rules.payout_buffer)),),
            _source(snap, "payout_buffer")))

    # --- sizing -----------------------------------------------------------------
    max_safe = prudent = 0
    risk_per_contract: Decimal | None = None
    if stop is not None and stop > _ZERO and snap.point_value > _ZERO:
        risk_per_contract = stop * snap.point_value
        # The most contracts whose worst-case loss still sits at/above the floor,
        # AND above the daily floor when one exists (the binding one wins).
        room = distance if d_distance is None else min(distance, d_distance)
        max_safe = int(max(room, _ZERO) / risk_per_contract)
        risk_budget = min(risk * equity, room)
        prudent = int(max(risk_budget, _ZERO) / risk_per_contract)
        prudent = int(Decimal(prudent) * guard.sizing_multiplier)
        binding = "total-drawdown floor"
        if d_distance is not None and d_distance < distance:
            binding = "daily-loss floor"
        if rules.max_contracts is not None:
            if rules.max_contracts < max_safe:
                binding = "max_contracts cap"
            max_safe = min(max_safe, rules.max_contracts)
            prudent = min(prudent, rules.max_contracts)
        traces.append(RuleTrace(
            "max_safe_contracts", str(max_safe), f"binding limit: {binding}",
            "floor(room_to_binding_floor / (stop_distance * point_value)), "
            "capped by max_contracts",
            (("room", _d(room)), ("stop_distance", _d(stop)),
             ("point_value", _d(snap.point_value)),
             ("max_contracts", _d(rules.max_contracts))),
            _source(snap, "max_contracts") + "; stop/point_value from the snapshot"))
        traces.append(RuleTrace(
            "prudent_contracts", str(prudent),
            "risk budget, then scaled by the guard level",
            "floor(min(risk_pct * equity, room) / risk_per_contract) * sizing_multiplier",
            (("risk_pct", _d(risk)), ("equity", _d(equity)),
             ("risk_per_contract", _d(risk_per_contract)),
             ("sizing_multiplier", _d(guard.sizing_multiplier))),
            "operator risk preference + DrawdownGuard"))

    # --- compliance --------------------------------------------------------------
    largest_pct, consistency_ok = _consistency(snap, rules)
    days_ok = snap.days_traded >= rules.min_trading_days
    winning_days = sum(1 for p in snap.daily_pnls if p > _ZERO)
    winning_days_ok = winning_days >= rules.min_winning_days
    deadline_ok = rules.max_eval_days is None or snap.days_elapsed <= rules.max_eval_days

    checks: list[ComplianceCheck] = [
        ComplianceCheck(
            "Minimum trading days", days_ok,
            f"{snap.days_traded} of {rules.min_trading_days} required",
            "min_trading_days", _source(snap, "min_trading_days")),
    ]
    if rules.min_winning_days:
        checks.append(ComplianceCheck(
            "Minimum winning days", winning_days_ok,
            f"{winning_days} of {rules.min_winning_days} required",
            "min_winning_days", _source(snap, "min_winning_days")))
    if rules.consistency_max_day_pct is not None:
        pct_txt = "n/a (no profit yet)" if largest_pct is None else f"{largest_pct:.0%}"
        checks.append(ComplianceCheck(
            "Consistency rule", consistency_ok,
            f"biggest day {pct_txt} of profit, cap {rules.consistency_max_day_pct:.0%}",
            "consistency_max_day_pct", _source(snap, "consistency_max_day_pct")))
    if rules.max_eval_days is not None:
        checks.append(ComplianceCheck(
            "Evaluation deadline", deadline_ok,
            f"day {snap.days_elapsed} of {rules.max_eval_days} allowed",
            "max_eval_days", _source(snap, "max_eval_days"), blocking=True))
    checks.append(ComplianceCheck(
        "Drawdown headroom", guard.level.value not in {"BLOCK", "KILL"},
        f"{guard.level.value}: ${distance:,.0f} above the ${floor:,.0f} floor",
        f"max_total_drawdown ({dd})",
        _source(snap, "drawdown_type", "max_total_drawdown")))
    if rules.profit_target is not None and target_equity is not None:
        hit = equity >= target_equity
        checks.append(ComplianceCheck(
            "Profit target", hit,
            (f"${equity:,.0f} of ${target_equity:,.0f}"
             + ("" if hit else f" (${target_equity - equity:,.0f} to go)")),
            "profit_target", _source(snap, "profit_target")))

    target_hit = target_equity is not None and equity >= target_equity
    can_pass_now = (target_hit and days_ok and winning_days_ok and consistency_ok
                    and deadline_ok and guard.level.value not in {"BLOCK", "KILL"})
    traces.append(RuleTrace(
        "can_pass_now", str(can_pass_now),
        "every phase gate cleared at once",
        "target_hit AND min_trading_days AND min_winning_days AND consistency "
        "AND deadline AND guard not BLOCK/KILL",
        (("target_hit", str(target_hit)), ("days_ok", str(days_ok)),
         ("winning_days_ok", str(winning_days_ok)),
         ("consistency_ok", str(consistency_ok)), ("deadline_ok", str(deadline_ok)),
         ("guard_level", guard.level.value)),
        _source(snap, "profit_target", "min_trading_days", "min_winning_days",
                "consistency_max_day_pct", "max_eval_days")))

    verdict, reasons = _verdict(
        guard.level.value, distance, floor, target_hit, can_pass_now, days_ok,
        snap.days_traded, rules.min_trading_days, consistency_ok, largest_pct,
        rules.consistency_max_day_pct, target_remaining, deadline_ok,
        snap.days_elapsed, rules.max_eval_days, winning_days_ok, winning_days,
        rules.min_winning_days)
    traces.append(RuleTrace(
        "verdict", verdict, "co-pilot verdict",
        "STOP if guard BLOCK/KILL or deadline blown; GO if passable or healthy; "
        "else CAUTION",
        (("guard_level", guard.level.value), ("can_pass_now", str(can_pass_now)),
         ("deadline_ok", str(deadline_ok)),
         ("consistency_ok", str(consistency_ok))),
        "co-pilot policy over the firm rules above"))

    return CopilotReport(
        total_floor=floor, distance_to_floor=distance,
        buffer_consumed_pct=guard.worst_fraction, guard_level=guard.level.value,
        sizing_multiplier=guard.sizing_multiplier, daily_floor=d_floor,
        daily_distance=d_distance, target_equity=target_equity,
        target_remaining=target_remaining, target_pct=target_pct,
        max_safe_contracts=max_safe, prudent_contracts=prudent,
        days_traded=snap.days_traded, min_trading_days=rules.min_trading_days,
        days_ok=days_ok, largest_day_pct=largest_pct,
        consistency_cap=rules.consistency_max_day_pct, consistency_ok=consistency_ok,
        can_pass_now=can_pass_now, verdict=verdict, reasons=reasons,
        account_id=snap.account_id, label=snap.display_label(), firm_id=snap.firm_id,
        firm_name=firm_name or snap.firm_id, phase=snap.phase.value, symbol=snap.symbol,
        equity=equity, initial_balance=snap.initial_balance,
        triggering_vector=guard.triggering_vector,
        open_pnl=equity - snap.initial_balance, risk_per_contract=risk_per_contract,
        withdrawable=withdrawable, winning_days=winning_days,
        min_winning_days=rules.min_winning_days, winning_days_ok=winning_days_ok,
        days_elapsed=snap.days_elapsed, max_eval_days=rules.max_eval_days,
        deadline_ok=deadline_ok, checks=tuple(checks), traces=tuple(traces))


def assess_fleet(snaps: Iterable[AccountSnapshot], registry: FirmRegistry) -> FleetReport:
    """Assess N accounts across M firms/phases in one pass.

    Each snapshot resolves its own rules from ``registry`` (firm + phase), so a fleet
    may mix trailing and static firms, evaluations and funded accounts. A snapshot
    naming an unknown firm or a phase the firm does not define is collected into
    :attr:`FleetReport.errors` instead of raising — one bad row never blinds the rest.
    """
    reports: list[CopilotReport] = []
    errors: list[tuple[str, str]] = []
    for snap in snaps:
        try:
            firm = registry.firm(snap.firm_id)
            rules = firm.phase(snap.phase).rules
        except KeyError as exc:
            errors.append((snap.display_label(), str(exc).strip("'")))
            continue
        reports.append(assess_account(snap, rules, firm_name=firm.name))
    return FleetReport(reports=tuple(reports), errors=tuple(errors))


def _verdict(level: str, distance: Decimal, floor: Decimal,
             target_hit: bool, can_pass_now: bool, days_ok: bool, days_traded: int,
             min_days: int, consistency_ok: bool, largest_pct: Decimal | None,
             cap: Decimal | None, target_remaining: Decimal | None, deadline_ok: bool,
             days_elapsed: int, max_eval_days: int | None, winning_days_ok: bool,
             winning_days: int, min_winning: int) -> tuple[str, tuple[str, ...]]:
    """Roll the signals into a GO / CAUTION / STOP verdict with human reasons."""
    reasons: list[str] = []
    if level in {"BLOCK", "KILL"}:
        reasons.append(f"drawdown at {level}: stop trading — floor ${floor:,.0f} is "
                       f"only ${distance:,.0f} away")
        return "STOP", tuple(reasons)
    if not deadline_ok:
        reasons.append(f"evaluation window expired: day {days_elapsed} of "
                       f"{max_eval_days} — this account can no longer pass")
        return "STOP", tuple(reasons)
    if can_pass_now:
        reasons.append("target reached AND compliance met — bank the pass, stop risking it")
        return "GO", tuple(reasons)
    if target_hit and not days_ok:
        reasons.append(f"target reached but only {days_traded}/{min_days} trading days — "
                       "trade tiny to clear the day count, do not risk the cushion")
        return "CAUTION", tuple(reasons)
    if target_hit and not winning_days_ok:
        reasons.append(f"target reached but only {winning_days}/{min_winning} winning days — "
                       "clear the winning-day count before risking the cushion")
        return "CAUTION", tuple(reasons)
    if level == "ALERT":
        reasons.append(f"drawdown ALERT: ${distance:,.0f} to the floor — half size, tighten up")
    if not consistency_ok and largest_pct is not None and cap is not None:
        reasons.append(f"consistency at risk: biggest day is {largest_pct:.0%} of profit "
                       f"(cap {cap:.0%}) — spread profit across more days")
    if target_remaining is not None and target_remaining > _ZERO:
        reasons.append(f"${target_remaining:,.0f} to target")
    if not reasons:
        reasons.append("healthy — keep trading the plan toward target")
    return ("CAUTION" if level == "ALERT" or not consistency_ok else "GO"), tuple(reasons)


def guard_worsened(previous: str, current: str) -> bool:
    """True when the guard level moved to a strictly more severe state."""
    return _GUARD_ORDER.get(current, 0) > _GUARD_ORDER.get(previous, 0)


def fleet_from_specs(specs: Sequence[dict[str, object]],
                     registry: FirmRegistry) -> FleetReport:
    """Build and assess a fleet from plain dicts (JSON/YAML rows).

    Each row needs ``firm`` and ``balance``; everything else is optional and mirrors
    :class:`AccountSnapshot`. Values may be strings (they are coerced to ``Decimal``),
    which is what a config file or an API payload will send.
    """
    return assess_fleet([_snapshot_from_spec(s) for s in specs], registry)


def _snapshot_from_spec(spec: dict[str, object]) -> AccountSnapshot:
    def dec(key: str, default: str | None = None) -> Decimal | None:
        raw = spec.get(key, default)
        return None if raw is None else Decimal(str(raw))

    balance = dec("balance") or dec("initial_balance")
    if balance is None:
        raise ValueError("account spec needs a 'balance'")
    equity = dec("equity") or balance
    hwm = dec("hwm") or max(balance, equity)
    phase_raw = str(spec.get("phase", AccountPhase.EVAL_F1.value))
    raw_pnls = spec.get("daily_pnls") or ()
    pnls = (tuple(Decimal(str(p)) for p in raw_pnls)
            if isinstance(raw_pnls, list | tuple) else ())
    return AccountSnapshot(
        firm_id=str(spec["firm"]), initial_balance=balance, current_equity=equity,
        high_watermark=hwm, day_start_equity=dec("day_start") or equity,
        point_value=dec("point_value", "1") or _ONE,
        days_traded=int(str(spec.get("days_traded", 0))), daily_pnls=pnls,
        account_id=str(spec.get("id", "account")), label=str(spec.get("label", "")),
        phase=AccountPhase(phase_raw), symbol=str(spec.get("symbol", "")),
        stop_distance=dec("stop"), risk_pct=dec("risk_pct", "0.01") or Decimal("0.01"),
        days_elapsed=int(str(spec.get("days_elapsed", 0))))
