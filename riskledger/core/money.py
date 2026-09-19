"""Money helpers.

All monetary and price values here are :class:`decimal.Decimal`. Using
``float`` for equity/PnL/drawdown is forbidden: rounding error in an engine that
decides whether to close the account at 90% drawdown is unacceptable.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

Number = int | str | Decimal

# Default quantization for currency amounts (2 dp). Prices may need more.
CENTS = Decimal("0.01")


def to_decimal(value: Number) -> Decimal:
    """Coerce ``int``/``str``/``Decimal`` to ``Decimal``.

    ``float`` is intentionally rejected to prevent silent precision loss. Pass a
    string literal (``"1.5"``) instead of a float.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):  # bool is a subclass of int — reject explicitly
        raise TypeError("bool is not a valid monetary value")
    if isinstance(value, (int, str)):
        return Decimal(value)
    raise TypeError(f"unsupported type for money: {type(value).__name__} (use str/int/Decimal)")


def money(value: Number, quantum: Decimal = CENTS) -> Decimal:
    """Coerce and round a value to a currency quantum (default cents)."""
    return to_decimal(value).quantize(quantum, rounding=ROUND_HALF_UP)
