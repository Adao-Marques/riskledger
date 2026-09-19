"""The co-pilot as a product: boundaries, explainability, fleets and the demo.

The verdict logic IS the product, so these tests pin the edges rather than the happy
path: exactly at the floor, exactly on each guard threshold, the consistency rule
exactly at its cap, unmet day counts, an expired evaluation window, and the same
account assessed under a different phase's rules.
"""

from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

from riskledger.business.copilot import (
    AccountSnapshot,
    FleetReport,
    assess_account,
    assess_fleet,
    fleet_from_specs,
    guard_worsened,
    resolve_rules,
)
from riskledger.config.models import PropFirmRules
from riskledger.core.enums import AccountPhase, DrawdownType
from riskledger.firms.firm_registry import FirmRegistry
from riskledger.reporting.copilot_html import render_fleet

BALANCE = Decimal("50000")
BUFFER = Decimal("1000")
FLOOR = BALANCE - BUFFER


@pytest.fixture(scope="module")
def registry() -> FirmRegistry:
    return FirmRegistry.from_dir("riskledger/firms/configs")


def _rules(**over: object) -> PropFirmRules:
    base: dict[str, object] = {
        "drawdown_type": DrawdownType.STATIC,
        "max_total_drawdown": BUFFER,
        "profit_target": Decimal("3000"),
        "alert_threshold": Decimal("0.70"),
        "block_threshold": Decimal("0.80"),
        "kill_switch_threshold": Decimal("0.90"),
    }
    base.update(over)
    return PropFirmRules(**base)  # type: ignore[arg-type]


def _snap(equity: str, **over: object) -> AccountSnapshot:
    kw: dict[str, object] = {
        "firm_id": "test", "initial_balance": BALANCE,
        "current_equity": Decimal(equity), "high_watermark": BALANCE,
        "day_start_equity": Decimal(equity), "point_value": Decimal("5"),
    }
    kw.update(over)
    return AccountSnapshot(**kw)  # type: ignore[arg-type]


# --------------------------------------------------------------- drawdown boundaries


@pytest.mark.parametrize(("equity", "level"), [
    ("49300.01", "OK"),      # 69.999% consumed
    ("49300", "ALERT"),      # exactly on the 70% alert threshold
    ("49200.01", "ALERT"),
    ("49200", "BLOCK"),      # exactly on the 80% block threshold
    ("49100.01", "BLOCK"),
    ("49100", "KILL"),       # exactly on the 90% kill threshold
    ("49000.01", "KILL"),    # one cent ABOVE the floor is still a kill
    ("49000", "KILL"),       # exactly AT the floor
    ("48999.99", "KILL"),    # breached
])
def test_guard_levels_are_exact_on_every_threshold(equity: str, level: str) -> None:
    rep = assess_account(_snap(equity), _rules())
    assert rep.guard_level == level
    assert rep.total_floor == FLOOR
    assert rep.distance_to_floor == Decimal(equity) - FLOOR


def test_at_the_floor_is_stop_and_forbids_new_size() -> None:
    rep = assess_account(_snap("49000", stop_distance=Decimal("2")), _rules())
    assert rep.verdict == "STOP"
    assert rep.prudent_contracts == 0
    assert rep.buffer_consumed_pct >= Decimal("1")


def test_one_tick_above_the_floor_still_stops_but_reports_the_gap() -> None:
    rep = assess_account(_snap("49000.25"), _rules())
    assert rep.verdict == "STOP"
    assert rep.distance_to_floor == Decimal("0.25")


def test_trailing_floor_ratchets_with_the_high_watermark() -> None:
    rules = _rules(drawdown_type=DrawdownType.TRAILING)
    rep = assess_account(_snap("51000", high_watermark=Decimal("51500")), rules)
    assert rep.total_floor == Decimal("51500") - BUFFER


def test_trailing_lock_caps_the_floor_at_the_initial_balance() -> None:
    rules = _rules(drawdown_type=DrawdownType.TRAILING, trailing_locks_at_initial=True)
    rep = assess_account(_snap("54000", high_watermark=Decimal("54000")), rules)
    assert rep.total_floor == BALANCE


def test_daily_floor_binds_sizing_when_it_is_the_tighter_limit() -> None:
    rules = _rules(max_daily_loss=Decimal("300"))
    snap = _snap("50000", day_start_equity=Decimal("50000"),
                 stop_distance=Decimal("10"), point_value=Decimal("5"))
    rep = assess_account(snap, rules)
    assert rep.daily_floor == Decimal("49700")
    assert rep.max_safe_contracts == 6       # 300 room / $50 per contract, not 1000/50
    assert "daily-loss floor" in (rep.explain("max_safe_contracts") or _missing()).rule


def _missing() -> object:
    raise AssertionError("expected a rule trace")


# ------------------------------------------------------------------ compliance edges


def test_consistency_exactly_at_the_cap_passes_and_one_cent_over_fails() -> None:
    rules = _rules(consistency_max_day_pct=Decimal("0.5"))
    at_cap = assess_account(
        _snap("51000", daily_pnls=(Decimal("500"), Decimal("500"))), rules)
    assert at_cap.largest_day_pct == Decimal("0.5")
    assert at_cap.consistency_ok is True

    over = assess_account(
        _snap("51000", daily_pnls=(Decimal("501"), Decimal("499"))), rules)
    assert over.consistency_ok is False
    assert over.verdict == "CAUTION"


def test_losing_days_never_trip_the_consistency_rule() -> None:
    rules = _rules(consistency_max_day_pct=Decimal("0.3"))
    rep = assess_account(
        _snap("49500", daily_pnls=(Decimal("-300"), Decimal("-200"))), rules)
    assert rep.largest_day_pct is None and rep.consistency_ok is True


def test_min_trading_days_unmet_holds_a_reached_target_at_caution() -> None:
    rules = _rules(min_trading_days=4)
    rep = assess_account(_snap("53000", days_traded=3), rules)
    assert rep.days_ok is False and rep.can_pass_now is False
    assert rep.verdict == "CAUTION"
    assert any("trading days" in x for x in rep.reasons)

    cleared = assess_account(_snap("53000", days_traded=4), rules)
    assert cleared.can_pass_now is True and cleared.verdict == "GO"


def test_min_winning_days_is_enforced_from_the_daily_pnls() -> None:
    rules = _rules(min_winning_days=3, min_trading_days=3)
    pnls = (Decimal("1500"), Decimal("1600"), Decimal("-100"))
    rep = assess_account(_snap("53000", days_traded=3, daily_pnls=pnls), rules)
    assert rep.winning_days == 2 and rep.winning_days_ok is False
    assert rep.can_pass_now is False and rep.verdict == "CAUTION"


def test_expired_evaluation_window_is_a_hard_stop() -> None:
    rules = _rules(max_eval_days=30)
    rep = assess_account(_snap("50500", days_elapsed=31), rules)
    assert rep.deadline_ok is False and rep.verdict == "STOP"
    assert any(c.name == "Evaluation deadline" and c.blocking for c in rep.checks)

    inside = assess_account(_snap("50500", days_elapsed=30), rules)
    assert inside.deadline_ok is True and inside.verdict != "STOP"


def test_target_exactly_met_counts_as_hit() -> None:
    rules = _rules(min_trading_days=1)
    assert assess_account(_snap("53000", days_traded=1), rules).can_pass_now is True
    assert assess_account(_snap("52999.99", days_traded=1), rules).can_pass_now is False


def test_funded_phase_has_no_target_and_reports_withdrawable() -> None:
    rules = _rules(profit_target=None, payout_buffer=Decimal("500"))
    rep = assess_account(_snap("51200"), rules)
    assert rep.target_equity is None and rep.can_pass_now is False
    assert rep.withdrawable == Decimal("700")


def test_phase_transition_changes_the_rules_applied(registry: FirmRegistry) -> None:
    """The same equity assessed under F1, F2 and FUNDED of one firm differs."""
    base = {"firm_id": "ftmo", "initial_balance": Decimal("100000"),
            "current_equity": Decimal("106000"), "high_watermark": Decimal("106000"),
            "day_start_equity": Decimal("106000"), "point_value": Decimal("1"),
            "days_traded": 5}
    out = {}
    for phase in (AccountPhase.EVAL_F1, AccountPhase.EVAL_F2, AccountPhase.FUNDED):
        snap = AccountSnapshot(**base, phase=phase)  # type: ignore[arg-type]
        out[phase] = assess_account(snap, resolve_rules(registry, snap))
    assert out[AccountPhase.EVAL_F1].can_pass_now is False       # needs +10,000
    assert out[AccountPhase.EVAL_F2].can_pass_now is True        # needs +5,000
    assert out[AccountPhase.FUNDED].target_equity is None        # payout phase


# ------------------------------------------------------------------ explainability


def test_every_trace_names_a_real_field_a_rule_and_its_firm_config_source() -> None:
    rules = _rules(drawdown_type=DrawdownType.TRAILING, min_trading_days=2,
                   consistency_max_day_pct=Decimal("0.4"), max_contracts=5)
    rep = assess_account(
        _snap("50600", high_watermark=Decimal("50900"), days_traded=2,
              daily_pnls=(Decimal("300"), Decimal("300")),
              stop_distance=Decimal("4")), rules)
    assert rep.traces
    for t in rep.traces:
        assert hasattr(rep, t.field), t.field
        assert t.rule and t.formula and t.source and t.inputs
        assert t.render().startswith(t.field)
    floor = rep.explain("total_floor")
    assert floor is not None
    assert floor.formula == "high_watermark - max_total_drawdown"
    assert ("max_total_drawdown", "1000") in floor.inputs
    assert "rules.max_total_drawdown" in floor.source
    assert rep.explain("no_such_field") is None


def test_checks_cover_every_gate_and_compliance_ok_agrees() -> None:
    rules = _rules(min_trading_days=2, min_winning_days=1,
                   consistency_max_day_pct=Decimal("0.4"), max_eval_days=45)
    rep = assess_account(_snap("53000", days_traded=3, days_elapsed=10,
                               daily_pnls=(Decimal("1200"), Decimal("1000"),
                                           Decimal("800"))), rules)
    names = {c.name for c in rep.checks}
    assert names == {"Minimum trading days", "Minimum winning days", "Consistency rule",
                     "Evaluation deadline", "Drawdown headroom", "Profit target"}
    assert rep.compliance_ok() is True and rep.can_pass_now is True


def test_guard_worsened_only_fires_on_a_more_severe_level() -> None:
    assert guard_worsened("OK", "ALERT") and guard_worsened("BLOCK", "KILL")
    assert not guard_worsened("ALERT", "OK") and not guard_worsened("KILL", "KILL")


# ------------------------------------------------------------------------- the fleet


def _fleet_snaps() -> list[AccountSnapshot]:
    return [
        AccountSnapshot(firm_id="apex", initial_balance=Decimal("50000"),
                        current_equity=Decimal("51400"),
                        high_watermark=Decimal("51600"),
                        day_start_equity=Decimal("51000"), point_value=Decimal("5"),
                        days_traded=4, account_id="A1", phase=AccountPhase.EVAL_F1,
                        stop_distance=Decimal("8")),
        AccountSnapshot(firm_id="ftmo", initial_balance=Decimal("100000"),
                        current_equity=Decimal("102000"),
                        high_watermark=Decimal("102000"),
                        day_start_equity=Decimal("102000"), point_value=Decimal("5"),
                        days_traded=3, account_id="S1",
                        daily_pnls=(Decimal("1700"), Decimal("150"), Decimal("150"))),
        AccountSnapshot(firm_id="apex", initial_balance=Decimal("50000"),
                        current_equity=Decimal("47900"),
                        high_watermark=Decimal("50250"),
                        day_start_equity=Decimal("48400"), point_value=Decimal("5"),
                        days_traded=3, account_id="A2", phase=AccountPhase.FUNDED),
    ]


def test_fleet_assesses_many_accounts_across_many_firms(registry: FirmRegistry) -> None:
    fleet = assess_fleet(_fleet_snaps(), registry)
    assert len(fleet) == 3 and not fleet.errors
    assert {r.firm_id for r in fleet.reports} == {"apex", "ftmo"}
    assert {r.phase for r in fleet.reports} == {"EVAL_F1", "FUNDED"}
    # Each account carries its OWN firm's floor — a fleet is not one shared rule set.
    floors = {r.account_id: r.total_floor for r in fleet.reports}
    assert floors["S1"] == Decimal("100000") - registry.firm("ftmo").phase(
        AccountPhase.EVAL_F1).rules.max_total_drawdown
    assert floors["A1"] == Decimal("51600") - registry.firm("apex").phase(
        AccountPhase.EVAL_F1).rules.max_total_drawdown
    assert fleet.fleet_verdict() == "STOP"
    worst = fleet.worst()
    assert worst is not None and worst.account_id == "A2"
    assert fleet.total_equity() == Decimal("201300")
    assert fleet.total_at_risk() == sum(r.distance_to_floor for r in fleet.reports)
    assert sum(fleet.counts().values()) == 3


def test_fleet_isolates_a_bad_account_instead_of_blinding_the_rest(
        registry: FirmRegistry) -> None:
    snaps = [*_fleet_snaps()]
    snaps.append(AccountSnapshot(firm_id="nosuchfirm", initial_balance=Decimal("50000"),
                                 current_equity=Decimal("50000"),
                                 high_watermark=Decimal("50000"),
                                 day_start_equity=Decimal("50000"),
                                 point_value=Decimal("1"), account_id="BAD"))
    snaps.append(AccountSnapshot(firm_id="apex", initial_balance=Decimal("50000"),
                                 current_equity=Decimal("50000"),
                                 high_watermark=Decimal("50000"),
                                 day_start_equity=Decimal("50000"),
                                 point_value=Decimal("1"), account_id="NOPHASE",
                                 phase=AccountPhase.EVAL_F2))
    fleet = assess_fleet(snaps, registry)
    assert len(fleet) == 3
    assert {label for label, _ in fleet.errors} == {"nosuchfirm BAD", "apex NOPHASE"}


def test_empty_fleet_is_safe(registry: FirmRegistry) -> None:
    fleet = assess_fleet([], registry)
    assert len(fleet) == 0 and fleet.worst() is None
    assert fleet.fleet_verdict() == "GO" and fleet.total_equity() == Decimal("0")
    assert "No accounts" in render_fleet(fleet)


def test_fleet_from_plain_dicts_coerces_strings(registry: FirmRegistry) -> None:
    fleet = fleet_from_specs([
        {"firm": "apex", "balance": "50000", "equity": "51000", "hwm": "51200",
         "days_traded": 2, "stop": "8", "point_value": "5", "symbol": "MES",
         "label": "Apex #1"},
        {"firm": "phidias", "balance": 50000, "equity": 53200, "days_traded": 3,
         "daily_pnls": [1000, 1200, 1000]},
    ], registry)
    assert len(fleet) == 2 and not fleet.errors
    assert fleet.reports[0].display_label() == "Apex #1"
    assert fleet.reports[0].prudent_contracts >= 1
    assert fleet.reports[1].can_pass_now is True


# ---------------------------------------------------------------------- the dashboard


def test_dashboard_is_self_contained_responsive_and_theme_aware(
        registry: FirmRegistry) -> None:
    fleet = assess_fleet(_fleet_snaps(), registry)
    html = render_fleet(fleet, refresh_secs=20, generated="2026-09-16 10:00 UTC")
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "http://" not in html and "https://" not in html      # no network at all
    assert "<script" not in html and "src=" not in html
    assert 'name="viewport"' in html                              # phone-readable
    assert "prefers-color-scheme:dark" in html                    # light + dark
    assert "@media (max-width:420px)" in html                     # collapses to 1 column
    assert 'http-equiv="refresh"' in html
    assert "distance to liquidation" in html                      # the hero number
    for r in fleet.reports:
        assert r.display_label() in html
        assert f"${r.distance_to_floor:,.0f}" in html
    assert "STOP" in html and "GO" in html
    assert "Why these numbers?" in html                            # rule traces shipped
    assert "does not predict results" in html                      # honest disclaimer


def test_dashboard_escapes_account_labels(registry: FirmRegistry) -> None:
    snap = AccountSnapshot(firm_id="apex", initial_balance=Decimal("50000"),
                           current_equity=Decimal("50000"),
                           high_watermark=Decimal("50000"),
                           day_start_equity=Decimal("50000"), point_value=Decimal("1"),
                           label="<script>alert(1)</script>")
    html = render_fleet(assess_fleet([snap], registry))
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_fleet_report_defaults_are_empty() -> None:
    empty = FleetReport()
    assert len(empty) == 0 and empty.counts() == {"GO": 0, "CAUTION": 0, "STOP": 0}


# ----------------------------------------------------------------------- the demo


def _load_demo() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "examples" / "copilot_demo.py"
    spec = importlib.util.spec_from_file_location("copilot_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_runs_with_no_broker_and_writes_a_dashboard(tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str],
                                                         ) -> None:
    risk_monitor = _load_demo()
    out = tmp_path / "demo.html"
    assert risk_monitor.main(["--demo", "--html", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    printed = capsys.readouterr().out
    assert "Fleet —" in printed and "VERDICT" in printed

    registry = FirmRegistry.from_dir("riskledger/firms/configs")
    fleet = risk_monitor.demo_fleet(registry)
    assert len(fleet) >= 5 and not fleet.errors
    # A prospect must see the whole range of outcomes, not just green tiles.
    assert {r.verdict for r in fleet.reports} == {"GO", "CAUTION", "STOP"}
    assert any(r.can_pass_now for r in fleet.reports)
    assert len({r.firm_id for r in fleet.reports}) >= 4
    assert len({r.phase for r in fleet.reports}) >= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
