"""The prop risk co-pilot: floor math, sizing, compliance and verdicts are honest."""

from __future__ import annotations

from decimal import Decimal

import pytest

from riskledger.business.copilot import AccountSnapshot, assess_account
from riskledger.core.enums import AccountPhase
from riskledger.firms.firm_registry import FirmRegistry


@pytest.fixture(scope="module")
def registry() -> FirmRegistry:
    return FirmRegistry.from_dir("riskledger/firms/configs")


def _rules(registry: FirmRegistry, firm: str):
    return registry.firm(firm).phase(AccountPhase.EVAL_F1).rules


def _snap(firm: str, *, equity: str, hwm: str, balance: str = "50000",
          day_start: str | None = None, days: int = 0,
          pnls: tuple[str, ...] = ()) -> AccountSnapshot:
    return AccountSnapshot(
        firm_id=firm, initial_balance=Decimal(balance), current_equity=Decimal(equity),
        high_watermark=Decimal(hwm), day_start_equity=Decimal(day_start or equity),
        point_value=Decimal("5"), days_traded=days,
        daily_pnls=tuple(Decimal(p) for p in pnls))


def test_static_floor_is_initial_minus_max_drawdown(registry: FirmRegistry) -> None:
    rules = _rules(registry, "ftmo")  # the real static-drawdown program
    assert rules.drawdown_type.value == "static"
    # profit must NOT raise the floor
    snap = _snap("ftmo", balance="100000", equity="100500", hwm="100800")
    rep = assess_account(snap, rules)
    assert rep.total_floor == Decimal("100000") - rules.max_total_drawdown


def test_trailing_floor_follows_high_watermark_and_breach_stops(registry: FirmRegistry) -> None:
    rules = _rules(registry, "apex")  # trailing
    # Equity below hwm - maxDD => breached => KILL => STOP verdict.
    snap = _snap("apex", equity="47600", hwm="50200", day_start="48000", days=3)
    rep = assess_account(snap, rules, stop_distance=Decimal("8"))
    assert rep.total_floor == Decimal("50200") - rules.max_total_drawdown
    assert rep.guard_level in {"BLOCK", "KILL"}
    assert rep.verdict == "STOP"
    assert rep.prudent_contracts == 0          # no new risk when blocked/killed


def test_sizing_respects_floor_and_max_contracts(registry: FirmRegistry) -> None:
    rules = _rules(registry, "mffu")  # eod floor + a max_contracts cap
    assert rules.max_contracts is not None
    snap = _snap("mffu", equity="51000", hwm="51000", day_start="51000")
    rep = assess_account(snap, rules, stop_distance=Decimal("8"))  # $40 risk/contract @ pv 5
    assert rep.prudent_contracts <= rep.max_safe_contracts
    # worst case of max_safe must not cross the floor.
    worst = Decimal(rep.max_safe_contracts) * Decimal("8") * Decimal("5")
    assert snap.current_equity - worst >= rep.total_floor
    if rules.max_contracts is not None:
        assert rep.max_safe_contracts <= rules.max_contracts


def test_consistency_rule_flags_a_too_large_day(registry: FirmRegistry) -> None:
    rules = _rules(registry, "mffu")  # eval carries consistency_max_day_pct
    assert rules.consistency_max_day_pct is not None
    # One day is the vast majority of profit -> over the cap -> not ok.
    snap = _snap("mffu", equity="52600", hwm="52600", days=3,
                 pnls=("2400", "100", "100"))
    rep = assess_account(snap, rules)
    assert rep.largest_day_pct is not None and rep.largest_day_pct > rules.consistency_max_day_pct
    assert rep.consistency_ok is False


def test_target_and_compliance_met_says_pass_now(registry: FirmRegistry) -> None:
    rules = _rules(registry, "phidias")  # eod, no consistency rule on the eval
    snap = _snap("phidias", equity="53100", hwm="53100", day_start="52000", days=3)
    rep = assess_account(snap, rules)
    assert rep.can_pass_now is True
    assert rep.verdict == "GO"
    assert any("bank the pass" in r for r in rep.reasons)


def test_target_hit_but_days_short_is_caution(registry: FirmRegistry) -> None:
    rules = _rules(registry, "mffu")  # min_trading_days >= 2
    assert rules.min_trading_days >= 2
    snap = _snap("mffu", equity="53100", hwm="53100", days=1)
    rep = assess_account(snap, rules)
    assert rep.can_pass_now is False
    assert rep.verdict == "CAUTION"


def test_copilot_html_renders_a_self_contained_tile(registry: FirmRegistry) -> None:
    from riskledger.reporting.copilot_html import render_copilot
    rules = _rules(registry, "phidias")
    snap = _snap("phidias", equity="51000", hwm="51200", days=1)
    rep = assess_account(snap, rules, stop_distance=Decimal("8"))
    html = render_copilot([("acct", Decimal("51000"), rep)], refresh_secs=30, generated="t")
    assert "<html" in html and rep.verdict in html
    assert "51,000" in html                         # equity rendered
    assert 'http-equiv="refresh"' in html           # auto-refresh tile
    assert "http://" not in html and "https://" not in html   # no external assets


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
