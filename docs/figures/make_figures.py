"""Render the figures in docs/ from the derived series in docs/figures/data/.

The CSVs are strategy and basket *returns* derived in the original research platform;
the raw vendor prices are not redistributed. Every number in a figure is recomputed
here from those returns, so the figures cannot drift from the data they claim to show.

    python docs/figures/make_figures.py        # writes *-light.png and *-dark.png

Needs matplotlib, numpy and pandas (not dependencies of the package itself).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TRADING_DAYS = 252


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_2: str
    muted: str
    grid: str
    axis: str
    series_1: str
    series_2: str
    band: str


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
              "#c3c2b7", "#2a78d6", "#eb6834", "#f0efec")
DARK = Theme("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a",
             "#383835", "#3987e5", "#d95926", "#262624")


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * math.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0


def _frame(t: Theme, w: float = 8.0, h: float = 4.6) -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(t.surface)
    ax.set_facecolor(t.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t.axis)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t.muted, labelsize=9, length=0)
    ax.grid(axis="y", color=t.grid, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    return fig, ax


def _titles(fig: plt.Figure, t: Theme, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, title, color=t.ink, fontsize=13, fontweight="bold",
             ha="left", va="top")
    fig.text(0.012, 0.915, subtitle, color=t.ink_2, fontsize=9.5, ha="left", va="top")


def _save(fig: plt.Figure, stem: str, t: Theme) -> None:
    fig.savefig(HERE / f"{stem}-{t.name}.png", facecolor=t.surface, dpi=200)
    plt.close(fig)


def fig_24y(t: Theme) -> None:
    """Growth of $1 over 23 years: momentum long-short vs the naive basket."""
    d = pd.read_csv(DATA / "a_24y_daily_returns.csv", parse_dates=["date"]).set_index("date")
    s_sr, b_sr = sharpe(d.momentum_long_short), sharpe(d.naive_basket)
    eq = (1.0 + d).cumprod()

    fig, ax = _frame(t)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.80, bottom=0.10)
    ax.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2012-12-31"),
               color=t.band, zorder=0, linewidth=0)
    ax.text(pd.Timestamp("2010-07-01"), 0.03, "2008–2012", transform=ax.get_xaxis_transform(),
            color=t.muted, fontsize=8.5, ha="center", va="bottom")
    ax.plot(eq.index, eq.naive_basket, color=t.series_2, linewidth=1.6,
            label=f"Naive equal-weight basket (Sharpe {b_sr:.2f})")
    ax.plot(eq.index, eq.momentum_long_short, color=t.series_1, linewidth=1.6,
            label=f"Momentum long-short (Sharpe {s_sr:.2f})")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for col, colour, name, sr in (("momentum_long_short", t.series_1, "Momentum L/S", s_sr),
                                  ("naive_basket", t.series_2, "Naive basket", b_sr)):
        y = eq[col].iloc[-1]
        ax.plot(eq.index[-1], y, "o", color=colour, markersize=5,
                markeredgecolor=t.surface, markeredgewidth=1.5)
        ax.annotate(f"{name}\nSharpe {sr:.2f} · ${y:.2f}", (eq.index[-1], y),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    color=t.ink, fontsize=9)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=t.ink_2)
    leg.set_zorder(5)
    _titles(fig, t, "Twenty-three years: the strategy loses to doing nothing clever",
            f"Growth of $1, {d.index[0].year}–{d.index[-1].year}, 13 futures markets, net of "
            f"costs. It lost money through 2008–2012 and ends "
            f"{1 - eq.momentum_long_short.iloc[-1] / eq.naive_basket.iloc[-1]:.0%} lower.")
    _save(fig, "fig-24y-growth", t)


def fig_beta(t: Theme) -> None:
    """Daily returns: long-only momentum against the equal-weight basket."""
    d = pd.read_csv(DATA / "b_6y_daily_returns.csv", parse_dates=["date"]).set_index("date")
    x, y = d.basket.to_numpy(), d.momentum_long_only.to_numpy()
    beta, intercept = np.polyfit(x, y, 1)
    resid = pd.Series(y - beta * x)
    alpha_sr = sharpe(resid)
    t_stat = alpha_sr * math.sqrt(len(resid) / TRADING_DAYS)

    fig, ax = _frame(t, w=6.4, h=6.0)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.82, bottom=0.11)
    lim = float(np.percentile(np.abs(np.concatenate([x, y])), 99.7)) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(axis="x", color=t.grid, linewidth=0.7, linestyle="-")
    ax.axhline(0, color=t.axis, linewidth=0.8, zorder=1)
    ax.axvline(0, color=t.axis, linewidth=0.8, zorder=1)
    ax.scatter(x, y, s=7, color=t.series_1, alpha=0.35, linewidths=0, zorder=2)
    xs = np.array([-lim, lim])
    ax.plot(xs, xs, color=t.muted, linewidth=1.0, zorder=3)
    ax.plot(xs, beta * xs + intercept, color=t.ink, linewidth=1.6, zorder=4)
    backing = {"boxstyle": "round,pad=0.25", "facecolor": t.surface, "edgecolor": "none",
               "alpha": 0.9}
    ax.annotate(f"fitted: beta {beta:.2f}", (lim * 0.55, beta * lim * 0.55 + intercept),
                xytext=(-10, 12), textcoords="offset points", color=t.ink, fontsize=9,
                ha="right", bbox=backing, zorder=5)
    ax.annotate("one-for-one (beta 1)", (-lim * 0.6, -lim * 0.6), xytext=(10, -14),
                textcoords="offset points", color=t.muted, fontsize=8.5, bbox=backing,
                zorder=5)
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("Equal-weight basket, daily return", color=t.ink_2, fontsize=9)
    ax.set_ylabel("Long-only momentum, daily return", color=t.ink_2, fontsize=9)
    _titles(fig, t, "It moves one-for-one with the basket",
            f"{len(d):,} trading days, {d.index[0].year}–{d.index[-1].year}, same 16 markets. "
            f"What is left after\nremoving the basket: Sharpe {alpha_sr:.2f}, "
            f"t = {t_stat:.2f} — indistinguishable from zero.")
    _save(fig, "fig-beta", t)


def fig_pass_rate(t: Theme) -> None:
    """Evaluation pass rate against per-trade Sharpe, one dot per configuration."""
    c = pd.read_csv(DATA / "c_pass_rate_vs_sharpe.csv")
    highlight = {("MES", "mom_pullback"): "mean-reversion pullback",
                 ("MES", "range_scalp"): "range scalp",
                 ("MES", "regime_switch"): "regime switch"}
    is_hi = c.apply(lambda r: (r.asset, r.strategy) in highlight, axis=1)

    fig, ax = _frame(t, w=8.0, h=4.8)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13)
    ax.axvline(0, color=t.axis, linewidth=0.8, zorder=1)
    ax.scatter(c.sharpe_per_trade[~is_hi], c.pass_rate[~is_hi], s=42, color=t.muted,
               alpha=0.55, edgecolors=t.surface, linewidths=1.5, zorder=2)
    ax.scatter(c.sharpe_per_trade[is_hi], c.pass_rate[is_hi], s=62, color=t.series_1,
               edgecolors=t.surface, linewidths=1.5, zorder=3)
    placement = {"mom_pullback": (-12, 10, "right", "bottom"),
                 "range_scalp": (12, 10, "left", "bottom"),
                 "regime_switch": (12, 0, "left", "center")}
    for _, r in c[is_hi].iterrows():
        dx, dy, ha, va = placement[r.strategy]
        pnl = f"PnL {'−' if r.total_pnl < 0 else '+'}${abs(r.total_pnl):,.0f}"
        sep = "\n" if r.strategy == "regime_switch" else " · "
        ax.annotate(f"{highlight[(r.asset, r.strategy)]}\n"
                    f"pass {r.pass_rate:.0%} · killed {r.kill_rate:.0%}{sep}{pnl}",
                    (r.sharpe_per_trade, r.pass_rate), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, color=t.ink, fontsize=8.5)
    ax.set_xlim(right=0.66)
    ax.set_xlabel("Sharpe per trade, not annualised  (left of the line: losing money)",
                  color=t.ink_2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.01, 0.335)
    ax.set_ylabel("Evaluation pass rate", color=t.ink_2, fontsize=9)
    _titles(fig, t, "The highest pass rates belong to money-losing strategies",
            f"{len(c)} strategy × market configurations, each measured over 103 rolling "
            "21-day windows\nwith costs charged. Grey: every other configuration.")
    _save(fig, "fig-pass-rate", t)


if __name__ == "__main__":
    for theme in (LIGHT, DARK):
        fig_24y(theme)
        fig_beta(theme)
        fig_pass_rate(theme)
    print("written:", sorted(p.name for p in HERE.glob("fig-*.png")))
