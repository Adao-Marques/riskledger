"""Contract point values: dollars per one full point of price, per contract or lot.

Sizing a trade needs one number per instrument — how many dollars a one-point move
is worth — and it has to be the same ``Decimal`` the risk kernel does its money
arithmetic in. This table is deliberately small, explicit and typed; add a symbol
only with the exchange's published contract multiplier.

Futures values are the CME contract multipliers. ``EURUSD`` is one standard lot
(100,000 units of base currency), so a 0.0001 move is $10 and a full 1.0000 move is
$100,000.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType

POINT_VALUES: Mapping[str, Decimal] = MappingProxyType({
    "ES": Decimal("50"),       # E-mini S&P 500
    "MES": Decimal("5"),       # Micro E-mini S&P 500
    "NQ": Decimal("20"),       # E-mini Nasdaq-100
    "MNQ": Decimal("2"),       # Micro E-mini Nasdaq-100
    "EURUSD": Decimal("100000"),  # spot FX, one standard lot
})


def point_value(symbol: str) -> Decimal:
    """Dollars per full point for ``symbol`` (KeyError if the symbol is not listed)."""
    try:
        return POINT_VALUES[symbol]
    except KeyError:
        raise KeyError(f"no point value for {symbol!r}; known: "
                       f"{', '.join(sorted(POINT_VALUES))}") from None
