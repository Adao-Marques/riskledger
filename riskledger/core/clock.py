"""Time helpers.

Everything here is UTC. Session windows for the strategies (London
Open, Opening Range) are defined in UTC; DST handling for the underlying venues
is the strategy's responsibility, never an implicit local-time assumption.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Return ``dt`` as a tz-aware UTC datetime.

    Naive datetimes are rejected: an ambiguous timestamp must never enter the
    risk or audit path.
    """
    if dt.tzinfo is None:
        raise ValueError("naive datetime is not allowed; provide a tz-aware UTC datetime")
    return dt.astimezone(UTC)
