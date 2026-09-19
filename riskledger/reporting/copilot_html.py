"""Render the prop risk co-pilot as a self-contained HTML dashboard.

One tile per account, laid out as a responsive grid that collapses to a single
column on a phone: a colour-coded GO / CAUTION / STOP badge, **distance to the
liquidation floor as the hero number**, the drawdown-buffer and target progress
bars, the prudent next-trade size, a tick/cross compliance checklist and — folded
away behind a disclosure — the rule trace behind every number.

Everything is inlined (CSS + markup, no images, no fonts, no scripts, no network
requests at all), so the file opens offline in any browser, in an IDE preview, or
as an email attachment. Light and dark are both handled via
``prefers-color-scheme``. An optional ``<meta refresh>`` turns it into a live tile
when the co-pilot rewrites the file on a watch loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal
from html import escape

from ..business.copilot import CopilotReport, FleetReport

_ACCENT = {"GO": "ok", "CAUTION": "warn", "STOP": "bad"}

_CSS = """
:root{color-scheme:light dark;--bg:#f6f8fa;--fg:#1f2328;--muted:#656d76;--card:#fff;
--line:#d0d7de;--track:#e6e9ee;--ok:#1a7f37;--ok-bg:#dafbe1;--warn:#9a6700;
--warn-bg:#fff8c5;--bad:#cf222e;--bad-bg:#ffebe9;--accent:#0969da}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;
--card:#161b22;--line:#30363d;--track:#21262d;--ok:#3fb950;--ok-bg:#12261a;
--warn:#d29922;--warn-bg:#2b2211;--bad:#f85149;--bad-bg:#2d1213;--accent:#4493f8}}
*{box-sizing:border-box}
body{margin:0;padding:16px 14px 40px;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
h1{font-size:19px;margin:0 0 2px}
.sub{color:var(--muted);font-size:12px;margin-bottom:14px}
.summary{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}
.pill{border:1px solid var(--line);background:var(--card);border-radius:999px;
padding:5px 12px;font-size:12px;white-space:nowrap}
.pill b{font-size:13px}
.pill.ok{color:var(--ok);border-color:var(--ok)}
.pill.warn{color:var(--warn);border-color:var(--warn)}
.pill.bad{color:var(--bad);border-color:var(--bad)}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:16px;overflow:hidden}
.card.ok{border-top:4px solid var(--ok)}
.card.warn{border-top:4px solid var(--warn)}
.card.bad{border-top:4px solid var(--bad)}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.name{font-weight:650;font-size:15px;word-break:break-word}
.meta{color:var(--muted);font-size:12px;margin-top:2px}
.badge{font-weight:700;font-size:12px;letter-spacing:.04em;padding:4px 11px;
border-radius:999px;white-space:nowrap}
.badge.ok{background:var(--ok-bg);color:var(--ok)}
.badge.warn{background:var(--warn-bg);color:var(--warn)}
.badge.bad{background:var(--bad-bg);color:var(--bad)}
.hero{margin:14px 0 2px}
.hero .label{color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.06em}
.hero .val{font-size:34px;font-weight:700;line-height:1.1;word-break:break-all}
.hero .val.ok{color:var(--ok)}.hero .val.warn{color:var(--warn)}
.hero .val.bad{color:var(--bad)}
.hero .note{color:var(--muted);font-size:12px}
.bar{background:var(--track);border-radius:5px;height:8px;overflow:hidden;margin:6px 0 2px}
.bar span{display:block;height:8px;border-radius:5px}
.rows{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:13px;
margin:12px 0 0}
.rows dt{color:var(--muted)}
.rows dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
.checks{list-style:none;margin:12px 0 0;padding:0;font-size:13px}
.checks li{display:flex;gap:8px;align-items:baseline;padding:2px 0}
.checks .mark{font-weight:700;width:1em;flex:none}
.checks .pass .mark{color:var(--ok)}
.checks .fail .mark{color:var(--bad)}
.checks .detail{color:var(--muted);font-size:12px}
.reasons{margin:12px 0 0;padding-left:18px;font-size:12.5px;color:var(--fg)}
.reasons li{margin:3px 0}
details{margin-top:12px;font-size:12px}
summary{cursor:pointer;color:var(--accent)}
.trace{margin:8px 0 0;padding:0;list-style:none;color:var(--muted)}
.trace li{padding:4px 0;border-top:1px solid var(--line);word-break:break-word}
.trace code{font:11.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.err{background:var(--bad-bg);color:var(--bad);border-radius:8px;padding:10px 14px;
margin-bottom:14px;font-size:13px}
.foot{color:var(--muted);font-size:11.5px;margin-top:22px;max-width:70ch}
@media (max-width:420px){.grid{grid-template-columns:1fr}.hero .val{font-size:29px}}
"""

_DISCLAIMER = (
    "Risk and compliance calculator. It reports where the firm's rules put your "
    "liquidation floor and whether the phase gates are met, using the same Decimal "
    "drawdown math the execution engine enforces. It does not place orders, does not "
    "predict results, and is only as correct as the firm rules it was given — always "
    "verify against your firm's current rulebook."
)


def _money(v: Decimal | None) -> str:
    return "n/a" if v is None else f"${v:,.0f}"


def _pct(v: Decimal | None) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def _bar(frac: float, accent: str) -> str:
    pct = max(0.0, min(1.0, frac)) * 100.0
    return (f'<div class="bar"><span style="width:{pct:.1f}%;'
            f'background:var(--{accent})"></span></div>')


def _checks(r: CopilotReport) -> str:
    if not r.checks:
        return ""
    items = "".join(
        f'<li class="{"pass" if c.ok else "fail"}"><span class="mark">'
        f'{"&#10003;" if c.ok else "&#10007;"}</span><span>{escape(c.name)}'
        f'<br><span class="detail">{escape(c.detail)}</span></span></li>'
        for c in r.checks)
    return f'<ul class="checks">{items}</ul>'


def _traces(r: CopilotReport) -> str:
    if not r.traces:
        return ""
    items = "".join(
        f'<li><b>{escape(t.field)}</b> = {escape(t.value)}<br>{escape(t.rule)}<br>'
        f'<code>{escape(t.formula)}</code><br>'
        f'{escape(", ".join(f"{k}={v}" for k, v in t.inputs))}<br>'
        f'<i>{escape(t.source)}</i></li>' for t in r.traces)
    return (f'<details><summary>Why these numbers? ({len(r.traces)} rules)</summary>'
            f'<ul class="trace">{items}</ul></details>')


def _card(r: CopilotReport) -> str:
    accent = _ACCENT.get(r.verdict, "warn")
    buf_left = 1.0 - min(1.0, float(r.buffer_consumed_pct))
    rows: list[tuple[str, str]] = [
        ("Equity", _money(r.equity)),
        ("Liquidation floor", _money(r.total_floor)),
    ]
    if r.daily_floor is not None:
        rows.append(("Daily floor", f"{_money(r.daily_floor)} "
                                    f"({_money(r.daily_distance)} away)"))
    if r.target_equity is not None:
        rows.append(("Target", f"{_money(r.target_equity)} "
                               f"({_money(r.target_remaining)} to go)"))
    if r.withdrawable is not None:
        rows.append(("Withdrawable", _money(r.withdrawable)))
    if r.max_safe_contracts or r.prudent_contracts:
        sym = f" {r.symbol}" if r.symbol else ""
        rows.append(("Next trade", f"{r.prudent_contracts} prudent / "
                                   f"{r.max_safe_contracts} max{escape(sym)}"))
    body = "".join(f"<dt>{escape(k)}</dt><dd>{v}</dd>" for k, v in rows)

    target_bar = ""
    if r.target_pct is not None:
        target_bar = (f'<div class="meta" style="margin-top:10px">target progress '
                      f'{_pct(r.target_pct)}</div>{_bar(float(r.target_pct), "accent")}')
    reasons = "".join(f"<li>{escape(x)}</li>" for x in r.reasons)
    meta = " &middot; ".join(
        escape(x) for x in (r.firm_name or r.firm_id, r.phase, r.symbol) if x)
    return (
        f'<section class="card {accent}">'
        f'<div class="head"><div><div class="name">{escape(r.display_label())}</div>'
        f'<div class="meta">{meta}</div></div>'
        f'<span class="badge {accent}">{escape(r.verdict)}</span></div>'
        f'<div class="hero"><div class="label">distance to liquidation</div>'
        f'<div class="val {accent}">{_money(r.distance_to_floor)}</div>'
        f'<div class="note">{_pct(r.buffer_consumed_pct)} of the drawdown buffer used '
        f'&middot; guard {escape(r.guard_level)} ({escape(r.triggering_vector)})</div>'
        f'</div>{_bar(buf_left, accent)}{target_bar}'
        f'<dl class="rows">{body}</dl>{_checks(r)}'
        f'<ul class="reasons">{reasons}</ul>{_traces(r)}</section>')


def _summary(fleet: FleetReport) -> str:
    counts = fleet.counts()
    pills = [f'<span class="pill"><b>{len(fleet)}</b> account'
             f'{"" if len(fleet) == 1 else "s"}</span>']
    for verdict, accent in (("GO", "ok"), ("CAUTION", "warn"), ("STOP", "bad")):
        if counts.get(verdict):
            pills.append(f'<span class="pill {accent}"><b>{counts[verdict]}</b> '
                         f'{verdict}</span>')
    pills.append(f'<span class="pill">equity <b>{_money(fleet.total_equity())}</b></span>')
    pills.append('<span class="pill">riskable before liquidation '
                 f'<b>{_money(fleet.total_at_risk())}</b></span>')
    passable = fleet.passable()
    if passable:
        pills.append(f'<span class="pill ok"><b>{len(passable)}</b> can pass now</span>')
    return f'<div class="summary">{"".join(pills)}</div>'


def render_fleet(fleet: FleetReport, *, refresh_secs: int = 0, generated: str = "",
                 title: str = "Prop risk co-pilot") -> str:
    """Render a whole fleet to one self-contained, responsive HTML page.

    ``refresh_secs`` > 0 adds a meta-refresh so a browser tile updates itself while
    the co-pilot rewrites the file.
    """
    meta = (f'<meta http-equiv="refresh" content="{refresh_secs}">'
            if refresh_secs > 0 else "")
    sub = (f"updated {escape(generated)}" if generated else "")
    if refresh_secs > 0:
        sub += f"{' &middot; ' if sub else ''}auto-refresh {refresh_secs}s"
    errs = "".join(f'<div class="err">{escape(label)}: {escape(msg)}</div>'
                   for label, msg in fleet.errors)
    cards = "".join(_card(r) for r in fleet.reports) or (
        '<section class="card"><div class="name">No accounts</div></section>')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'{meta}<title>{escape(title)}</title><style>{_CSS}</style></head><body>'
        f'<h1>{escape(title)}</h1><div class="sub">{sub}</div>'
        f'{_summary(fleet)}{errs}<div class="grid">{cards}</div>'
        f'<p class="foot">{escape(_DISCLAIMER)}</p></body></html>')


def render_copilot(cards: Sequence[tuple[str, Decimal, CopilotReport]], *,
                   refresh_secs: int = 0, generated: str = "") -> str:
    """Render one or more account cards to a self-contained HTML page.

    ``cards`` is a list of ``(label, equity, report)`` — the label and equity override
    whatever the report carries, which keeps the older single-account call sites and
    the live ``--watch`` loop working. Prefer :func:`render_fleet` for new code.
    """
    fleet = FleetReport(tuple(
        replace(rep, label=label, equity=equity) for label, equity, rep in cards))
    return render_fleet(fleet, refresh_secs=refresh_secs, generated=generated)
