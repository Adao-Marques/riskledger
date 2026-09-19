# riskledger

A `Decimal`-exact risk kernel and an honest-measurement library for systematic trading,
extracted from a six-month research platform after an adversarial audit of that platform
found that its one claimed edge was a bug plus market beta.

This is the part that survived — because it tells the truth.

```bash
pip install -e ".[dev]"
pytest                                     # unit + Hypothesis property tests
python examples/copilot_demo.py --html fleet.html
```

---

## What is in here

### A risk kernel you can audit

Money never touches a `float`. Equity, PnL, drawdown and position size are `Decimal` end
to end, so a floor computed here is the floor to the cent, not to the nearest
binary-representable neighbour of it.

- **Drawdown in three regimes** — `trailing` (the watermark follows every tick),
  `eod` (it steps once per day, at the close) and `static`. Implemented in
  [`risk/drawdown.py`](riskledger/risk/drawdown.py) and enforced by
  [`risk/drawdown_guard.py`](riskledger/risk/drawdown_guard.py).
- **A kill switch that is actually atomic.** One latch per account, a lock around the
  claim only (so a hung broker cannot wedge every other caller inside the mutex), losers
  that block and receive the winner's real result, and suspension that happens in a
  `finally` so a failing audit sink cannot leave an account live.
  [`risk/kill_switch.py`](riskledger/risk/kill_switch.py)
- **Position sizing from the stop.** How many contracts can this trade carry, given this
  stop distance, without a loss at the stop crossing the floor?
  [`risk/position_sizer.py`](riskledger/risk/position_sizer.py)
- **Rules as data.** Each firm's program is a YAML file, not code
  ([`firms/configs/`](riskledger/firms/configs/)), and a hash-chained, append-only
  compliance log records every decision ([`audit/compliance_log.py`](riskledger/audit/compliance_log.py)).
- **Every number carries its derivation.** The co-pilot
  ([`business/copilot.py`](riskledger/business/copilot.py)) returns, for each figure,
  the formula, its inputs and the configuration field it came from — a risk number
  nobody can audit is a number nobody should trust.

### A measurement library that assumes you are fooling yourself

- **Walk-forward with purging and embargo** (López de Prado, *Advances in Financial
  Machine Learning*, ch. 7), so overlapping labels cannot leak the future into training.
  [`research/ml.py`](riskledger/research/ml.py)
- **Combinatorially-symmetric cross-validation and the Probability of Backtest
  Overfitting** (Bailey, Borwein, López de Prado & Zhu).
  [`validation/overfitting.py`](riskledger/validation/overfitting.py),
  [`validation/cpcv.py`](riskledger/validation/cpcv.py)
- **White's Reality Check and Hansen's SPA test** with a moving-block bootstrap, to deflate
  the best of *N* searched configurations rather than reporting it as if it were the only
  one tried. [`validation/reality_check.py`](riskledger/validation/reality_check.py)
- **Probabilistic and Deflated Sharpe ratios** with sample skew and kurtosis, deflated by
  the true search size. [`validation/sharpe.py`](riskledger/validation/sharpe.py)
- **Named performance conventions.** "Sharpe" means at least three incompatible things in
  most codebases; here each is a separate, named function, and the one that silently
  inflates sparse strategies has a documented correction.
  [`validation/conventions.py`](riskledger/validation/conventions.py)

---

## Four things worth reading the code for

**1. A race condition a normal test cannot find.** The original kill switch was a
check-then-set on a bare boolean. Racing 32 threads against it for 25 rounds produced
**zero** double-fires, and 200 further rounds at a one-nanosecond switch interval produced
zero more: on a GIL build the window is about three bytecodes. So the test *widens the
seam* — a subclass turns the latch flag into a property that yields on every read and
write, leaving the production `trigger()` untouched. Old latch: 14 of 15 rounds
double-fire. New: 0 of 15.
[`tests/unit/test_kill_switch_concurrency.py`](tests/unit/test_kill_switch_concurrency.py)

**2. "End-of-day trailing drawdown" is a misnomer everywhere.** Four firms state in their
own documentation that the *floor* steps at the close but the *breach* is tested on every
tick, against open equity. A guard that only checks at the close is optimistic; one that
trails the intraday peak is pessimistic. The guard here assesses the breach at each bar's
intra-bar trough and ratchets the watermark to the day's close — and a test pins the
second half, because an earlier version ratcheted to the trough instead and left the floor
below the firm's. [`tests/unit/test_drawdown_guard.py`](tests/unit/test_drawdown_guard.py)

**3. Property tests on the invariants, not examples of them.** Hypothesis generates
arbitrary equity paths and asserts that the reported floor, guard level, consumed fraction
and sizing multiplier are *identical* to what the enforcing guard computes; that anything
at or below the floor is a stop at size zero; and that the maximum safe size never crosses
the floor at the stop. [`tests/property/`](tests/property/)

**4. A drawdown function that refuses to lie.** `max_drawdown_fraction` raises on a
cumulative-PnL series instead of returning a number. Writing its test caught an earlier
version reporting a **150%** drawdown for `[0, -5, -3, 2, -1]` — the result of dividing
by a near-zero peak. A function that returns something shaped like a percentage that is
not one is worse than a function that raises.
[`tests/unit/test_conventions.py`](tests/unit/test_conventions.py)

---

## What this is not

- **Not a trading strategy.** No edge is claimed here, because none was found. The
  measurement library is what established that; see
  [`docs/what-the-audit-found.md`](docs/what-the-audit-found.md).
- **Not a statement of what any firm will decide.** The co-pilot reports what a program's
  *published* rules imply. Firms reserve the right to change those rules without notice
  and to apply discretion; a calculator does not bind them. The firm configurations here
  are best-effort transcriptions of public documentation — verify against the firm's
  current terms before relying on any number.
- **Not financial advice.**

## The story

[`docs/what-the-audit-found.md`](docs/what-the-audit-found.md) — how a platform built to
find an edge ended up proving it had none, and which engineering decisions made that
possible to see.

## Licence

MIT. See [`LICENSE`](LICENSE).
