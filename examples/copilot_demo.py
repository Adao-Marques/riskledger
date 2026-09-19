"""Prop-account risk co-pilot — a runnable demo of the risk kernel on account state.

Give it a firm and an account snapshot and it prints the liquidation line, how big
the next trade can safely be, the compliance checklist and how close the account is
to its profit target — the same limits the risk engine enforces, with every number
traceable to the firm rule that produced it.

    # zero-setup demo: a synthetic multi-account fleet + an HTML dashboard
    python examples/copilot_demo.py --html copilot_demo.html

    # one account from a manual snapshot (--symbol sets the point value)
    python examples/copilot_demo.py --single --firm phidias --balance 50000 \\
        --equity 51200 --hwm 51800 --day-start 50900 --days-traded 1 \\
        --symbol MES --stop 8 --explain

    # several accounts from a YAML/JSON file (N accounts across M firms)
    python examples/copilot_demo.py --fleet accounts.yaml --html fleet.html

This is a risk and compliance calculator. It never places an order, it does not
predict results, and it is only as correct as the firm rules in
``riskledger/firms/configs/*.yaml`` — verify those against the firm's current rulebook.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

if __package__ in (None, ""):
    # Run straight from a clone (`python examples/copilot_demo.py`) without installing.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskledger.business.copilot import (  # noqa: E402
    AccountSnapshot,
    CopilotReport,
    FleetReport,
    assess_account,
    assess_fleet,
    fleet_from_specs,
)
from riskledger.core.enums import AccountPhase  # noqa: E402
from riskledger.firms.firm_registry import BUNDLED_CONFIGS, FirmRegistry  # noqa: E402
from riskledger.firms.instruments import POINT_VALUES, point_value  # noqa: E402
from riskledger.reporting.copilot_html import render_fleet  # noqa: E402

_BANNER = {"GO": "✅ GO", "CAUTION": "⚠️  CAUTION", "STOP": "⛔ STOP"}
_DEFAULT_HTML = "copilot_demo.html"


def _out(msg: str = "") -> None:
    print(msg)  # noqa: T201


# ------------------------------------------------------------------- demo fleet

# (firm, phase, label, symbol, stop, balance, daily PnLs as fractions of the firm's
#  max_total_drawdown, how far the high-watermark sits above today's equity — also
#  as a fraction of the drawdown buffer, which for a trailing/eod firm IS the
#  fraction of the buffer consumed). Only real, published firm programs appear here.
_DEMO: tuple[tuple[str, AccountPhase, str, str, str, str,
                   tuple[str, ...], str], ...] = (
    ("apex", AccountPhase.EVAL_F1, "Apex 50K — eval", "MES", "8", "50000",
     ("0.24", "0.20", "-0.12", "0.24"), "0.08"),
    # Deliberately a CLEARED account: target reached with compliance met, so the
    # demo shows the "bank the pass, stop risking it" advice.
    ("phidias", AccountPhase.EVAL_F1, "Phidias Swing 50K — eval", "MES", "12", "50000",
     ("0.66", "0.56", "0.44", "0.46"), "0.06"),
    ("topstep", AccountPhase.EVAL_F1, "Topstep 50K — eval", "MNQ", "30", "50000",
     ("0.20", "-0.52", "-0.20"), "0.72"),
    ("mffu", AccountPhase.EVAL_F1, "MyFundedFutures 50K — eval", "MES", "10", "50000",
     ("0.85", "0.075", "0.075"), "0.05"),
    ("topstep", AccountPhase.EVAL_F1, "Topstep 50K — second eval", "MES", "9", "50000",
     ("0.22", "-0.10", "0.18"), "0.30"),
    ("apex", AccountPhase.FUNDED, "Apex 50K — funded", "MES", "8", "50000",
     ("0.12", "-0.40", "-0.56"), "0.94"),
    ("ftmo", AccountPhase.EVAL_F2, "FTMO 100K — phase 2", "EURUSD", "0.0030", "100000",
     ("0.12", "0.09", "-0.05", "0.14"), "0.05"),
)


def demo_fleet(registry: FirmRegistry) -> FleetReport:
    """Build a synthetic fleet and assess it — no broker, no credentials, no network.

    The accounts are expressed as fractions of each firm's own drawdown buffer, so
    the demo stays coherent with whatever the firm YAMLs say. It deliberately spans
    every verdict: healthy evaluations, one at the drawdown ALERT line, one tripping
    the consistency rule, one that has cleared its target, and one funded account
    past the kill threshold.
    """
    snaps: list[AccountSnapshot] = []
    for i, (firm_id, phase, label, symbol, stop, balance, pnl_r, peak_r) in enumerate(_DEMO):
        buf = registry.firm(firm_id).phase(phase).rules.max_total_drawdown
        pnls = tuple((Decimal(r) * buf).quantize(Decimal("0.01")) for r in pnl_r)
        bal = Decimal(balance)
        equity = bal + sum(pnls, Decimal("0"))
        hwm = equity + Decimal(peak_r) * buf
        snaps.append(AccountSnapshot(
            firm_id=firm_id, initial_balance=bal, current_equity=equity,
            high_watermark=max(hwm, bal), day_start_equity=equity - pnls[-1],
            point_value=point_value(symbol), days_traded=len(pnls), daily_pnls=pnls,
            account_id=f"DEMO-{i + 1:02d}", label=label, phase=phase, symbol=symbol,
            stop_distance=Decimal(stop), days_elapsed=len(pnls) + 2))
    return assess_fleet(snaps, registry)


def _load_specs(path: str) -> list[dict[str, object]]:
    """Read a fleet file: a JSON/YAML list of account dicts, or ``{accounts: [...]}``."""
    text = Path(path).read_text(encoding="utf-8")
    raw: object
    if path.endswith((".yaml", ".yml")):
        import yaml
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if isinstance(raw, dict):
        raw = raw.get("accounts", [])
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of accounts")
    return [dict(row) for row in raw]


# --------------------------------------------------------------------- printing


def _print(report: CopilotReport, *, explain: bool = False) -> None:
    r = report
    _out(f"\n{_BANNER.get(r.verdict, r.verdict)}  —  equity ${r.equity:,.0f}\n")
    for reason in r.reasons:
        _out(f"   • {reason}")
    _out("\n  Drawdown")
    _out(f"    liquidation floor   ${r.total_floor:,.0f}")
    _out(f"    distance to floor   ${r.distance_to_floor:,.0f}  "
         f"(buffer used {r.buffer_consumed_pct:.0%}, guard {r.guard_level} "
         f"on the {r.triggering_vector} vector)")
    if r.daily_floor is not None and r.daily_distance is not None:
        _out(f"    daily floor         ${r.daily_floor:,.0f}  "
             f"(${r.daily_distance:,.0f} away)")
    if r.target_equity is not None and r.target_remaining is not None:
        pct = "n/a" if r.target_pct is None else f"{r.target_pct:.0%}"
        _out("\n  Target")
        _out(f"    target equity       ${r.target_equity:,.0f}  "
             f"(${r.target_remaining:,.0f} to go, {pct} there)")
    if r.withdrawable is not None:
        _out(f"    withdrawable        ${r.withdrawable:,.0f}")
    if r.max_safe_contracts or r.prudent_contracts:
        _out("\n  Sizing (next trade)")
        _out(f"    prudent size        {r.prudent_contracts} contract(s)  "
             f"(hard max before the floor: {r.max_safe_contracts})")
    _out("\n  Compliance")
    for c in r.checks:
        mark = "ok " if c.ok else "NO "
        _out(f"    [{mark}] {c.name:<22} {c.detail}")
    _out(f"    can pass now        {'YES — bank it' if r.can_pass_now else 'not yet'}")
    if explain:
        _explain(r)
    _out()


def _explain(report: CopilotReport) -> None:
    _out("\n  Why these numbers (rule -> formula -> inputs)")
    for t in report.traces:
        _out(f"    {t.render()}")
        _out(f"       source: {t.source}")


def _fleet_table(fleet: FleetReport) -> None:
    """Print the fleet as one compact table: every account on one screen."""
    counts = fleet.counts()
    _out(f"\n  Fleet — {len(fleet)} account(s): "
         f"{counts['GO']} GO / {counts['CAUTION']} CAUTION / {counts['STOP']} STOP  "
         f"| equity ${fleet.total_equity():,.0f} "
         f"| riskable before liquidation ${fleet.total_at_risk():,.0f}\n")
    head = (f"  {'ACCOUNT':<28}{'PHASE':<9}{'EQUITY':>11}{'FLOOR':>11}"
            f"{'TO FLOOR':>11}{'BUF':>6}{'TARGET':>8}{'SIZE':>6}  VERDICT")
    _out(head)
    _out("  " + "-" * (len(head) - 2))
    for r in fleet.reports:
        tgt = "n/a" if r.target_pct is None else f"{r.target_pct:.0%}"
        _out(f"  {r.display_label()[:27]:<28}{r.phase:<9}"
             f"{f'${r.equity:,.0f}':>11}{f'${r.total_floor:,.0f}':>11}"
             f"{f'${r.distance_to_floor:,.0f}':>11}"
             f"{r.buffer_consumed_pct:>6.0%}{tgt:>8}{r.prudent_contracts:>6}  "
             f"{_BANNER.get(r.verdict, r.verdict)}")
    for r in fleet.reports:
        if r.verdict != "GO" or r.can_pass_now:
            _out(f"\n  {r.display_label()}: {r.reasons[0]}")
    for label, msg in fleet.errors:
        _out(f"\n  ! {label}: {msg}")
    _out()


def _write_fleet_html(path: str, fleet: FleetReport, *,
                      title: str = "Prop risk co-pilot") -> None:
    """Write a fleet to a self-contained HTML dashboard (no external requests)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    out.write_text(render_fleet(fleet, generated=stamp, title=title), encoding="utf-8")


# -------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prop-account risk co-pilot demo. With no mode flag it assesses "
                    "a synthetic multi-account fleet.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true",
                      help="assess the built-in synthetic fleet (the default)")
    mode.add_argument("--fleet", default=None,
                      help="YAML/JSON file holding a list of accounts to assess together")
    mode.add_argument("--single", action="store_true",
                      help="assess one account from the snapshot flags below")
    p.add_argument("--firm", default="apex")
    p.add_argument("--phase", default=AccountPhase.EVAL_F1.value,
                   choices=[ph.value for ph in AccountPhase])
    p.add_argument("--balance", default="50000", help="initial account balance")
    p.add_argument("--equity", default=None, help="current equity (default: balance)")
    p.add_argument("--hwm", default=None,
                   help="peak equity reached (default: max(balance, equity))")
    p.add_argument("--day-start", default=None, help="equity at the start of today")
    p.add_argument("--days-traded", type=int, default=0)
    p.add_argument("--days-elapsed", type=int, default=0,
                   help="days since the evaluation started (vs max_eval_days)")
    p.add_argument("--daily-pnls", default="", help="comma-separated realised daily PnLs")
    p.add_argument("--symbol", default=None, choices=sorted(POINT_VALUES))
    p.add_argument("--point-value", default=None, help="override the symbol's point value")
    p.add_argument("--stop", default=None,
                   help="price distance entry->stop, to size the next trade")
    p.add_argument("--risk-pct", default="0.01", help="fraction of equity risked per trade")
    p.add_argument("--explain", action="store_true",
                   help="print the rule, formula and firm-config inputs behind every number")
    p.add_argument("--firms-dir", default=str(BUNDLED_CONFIGS),
                   help="directory of firm YAML configs (default: the bundled ones)")
    p.add_argument("--html", default=None,
                   help=f"write a self-contained HTML dashboard here "
                        f"(fleet modes default to {_DEFAULT_HTML})")
    return p


def _single(args: argparse.Namespace, registry: FirmRegistry) -> FleetReport:
    phase = AccountPhase(args.phase)
    firm = registry.firm(args.firm)
    balance = Decimal(args.balance)
    equity = Decimal(args.equity) if args.equity is not None else balance
    if args.point_value is not None:
        pv = Decimal(args.point_value)
    elif args.symbol:
        pv = point_value(args.symbol)
    else:
        pv = Decimal("1")
    snap = AccountSnapshot(
        firm_id=args.firm, initial_balance=balance, current_equity=equity,
        high_watermark=Decimal(args.hwm) if args.hwm else max(balance, equity),
        day_start_equity=Decimal(args.day_start) if args.day_start else equity,
        point_value=pv, days_traded=args.days_traded,
        daily_pnls=tuple(Decimal(x) for x in args.daily_pnls.split(",") if x.strip()),
        phase=phase, symbol=args.symbol or "",
        stop_distance=Decimal(args.stop) if args.stop else None,
        risk_pct=Decimal(args.risk_pct), days_elapsed=args.days_elapsed)
    report = assess_account(snap, firm.phase(phase).rules, firm_name=firm.name)
    _print(report, explain=args.explain)
    return FleetReport((report,))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = FirmRegistry.from_dir(args.firms_dir)

    if args.single:
        fleet = _single(args, registry)
        if args.html:
            _write_fleet_html(args.html, fleet)
            _out(f"  HTML dashboard -> {args.html}\n")
        return 0

    if args.fleet:
        fleet = fleet_from_specs(_load_specs(args.fleet), registry)
        title = "Prop risk co-pilot"
    else:
        fleet = demo_fleet(registry)
        title = "Prop risk co-pilot — demo fleet"
    _fleet_table(fleet)
    html_path = args.html or _DEFAULT_HTML
    _write_fleet_html(html_path, fleet, title=title)
    _out(f"  HTML dashboard -> {html_path}  (self-contained; open it in any browser)\n")
    if args.explain:
        for r in fleet.reports:
            _out(f"  {r.display_label()}")
            _explain(r)
            _out()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
