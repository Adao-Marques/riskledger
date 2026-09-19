"""BoldPlaySizer — Dubins-Savage barrier-proportional sizing, bounded by survival.

Locks the core properties: the bet is ``boldness`` × the distance to the drawdown
floor, it is strictly bolder than the fixed sizer when there is room, it shrinks to
zero as the floor nears (so one stop-out can't cross it), and the boldness fraction
is clamped to (0, 1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from riskledger.config.models import PropFirmRules
from riskledger.core.enums import Direction, DrawdownType
from riskledger.core.types import Account, Signal
from riskledger.risk.position_sizer import BoldPlaySizer, FixedFractionalSizer

# Static DD 2000: floor = initial - 2000. With equity 50000 -> buffer 2000.
RULES = PropFirmRules(drawdown_type=DrawdownType.STATIC, max_total_drawdown=Decimal("2000"))


def _acc(equity: str = "50000") -> Account:
    a = Account.open("a", "f", Decimal("50000"))
    a.update_equity(Decimal(equity))
    return a


def _sig(entry: str = "100", stop: str = "98") -> Signal:
    return Signal(id="s", strategy_id="b", ts=datetime(2024, 1, 1, tzinfo=UTC),
                  symbol="ES", direction=Direction.LONG,
                  entry_price=Decimal(entry), stop_price=Decimal(stop))


def test_size_is_a_fraction_of_the_drawdown_buffer() -> None:
    # buffer 2000, boldness 0.25 -> budget 500; risk/contract 2 -> 250 contracts.
    sizer = BoldPlaySizer(boldness=Decimal("0.25"), max_contracts=100000)
    assert sizer.size(_sig(), _acc(), RULES) == Decimal("250")


def test_bolder_than_fixed_when_there_is_room() -> None:
    state = (_sig(), _acc(), RULES)
    bold = BoldPlaySizer(boldness=Decimal("0.25"), max_contracts=100000).size(*state)
    fixed = FixedFractionalSizer(max_contracts=100000).size(*state)
    assert bold > fixed                       # 250 vs 100 (timid 0.5%/equity capped)


def test_shrinks_toward_zero_as_floor_nears_and_never_crosses() -> None:
    sizer = BoldPlaySizer(boldness=Decimal("0.25"), max_contracts=100000)
    far = sizer.size(_sig(), _acc("50000"), RULES)      # buffer 2000
    near = sizer.size(_sig(), _acc("48040"), RULES)     # buffer 40
    assert near < far
    # At the floor (buffer 0) and below it, no position is taken.
    assert sizer.size(_sig(), _acc("48000"), RULES) == Decimal("0")
    assert sizer.size(_sig(), _acc("47500"), RULES) == Decimal("0")


def test_single_trade_cannot_risk_the_whole_buffer() -> None:
    # boldness clamps below 1, so budget = boldness*buffer < buffer for any state:
    # one stop-out always leaves equity above the floor.
    sizer = BoldPlaySizer(boldness=Decimal("0.95"), max_contracts=100000)
    contracts = sizer.size(_sig(entry="100", stop="99"), _acc("50000"), RULES)  # rpc 1
    risked = contracts * Decimal("1")
    assert risked < Decimal("2000")           # < the full buffer


def test_boldness_is_clamped_to_open_unit_interval() -> None:
    assert BoldPlaySizer(boldness=Decimal("5")).boldness == Decimal("0.95")
    assert BoldPlaySizer(boldness=Decimal("0")).boldness == Decimal("0.01")
    assert BoldPlaySizer(boldness=Decimal("-3")).boldness == Decimal("0.01")


def test_max_contracts_caps_the_bold_bet() -> None:
    sizer = BoldPlaySizer(boldness=Decimal("0.25"), max_contracts=6)
    assert sizer.size(_sig(), _acc(), RULES) == Decimal("6")


def test_degenerate_inputs_return_zero() -> None:
    sizer = BoldPlaySizer()
    assert sizer.size(_sig(entry="100", stop="100"), _acc(), RULES) == Decimal("0")  # no risk
    assert sizer.size(_sig(), _acc("0"), RULES) == Decimal("0")                      # no equity
