"""FirmRegistry — prop firms modelled as data, never hard-coded.

Each firm program is a validated YAML document -- drawdown structure, limits, targets,
compliance rules, fees and payout terms -- so a rule change is a data change, never a
code change. The configurations shipped here are transcriptions of each firm's public
documentation; firms change terms without notice, so verify against the firm's current
rules before relying on any number derived from them.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..config.models import PropFirmRules
from ..core.enums import AccountPhase


class FirmPhaseConfig(BaseModel):
    """Rules and objective for one phase of a firm's program."""

    model_config = ConfigDict(frozen=True)

    phase: AccountPhase
    objective: Literal["pass", "payout"]
    rules: PropFirmRules


class FirmConfig(BaseModel):
    """A prop firm program modelled as data."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    evaluation_type: str
    fee: Decimal = Field(ge=0)
    profit_split: Decimal = Field(gt=0, le=1)
    payout_reliability: Decimal = Field(ge=0, le=1)
    phases: tuple[FirmPhaseConfig, ...] = Field(min_length=1)

    # Funded-account lifecycle terms, optional. Lets an expected-value model count
    # payout cycles from the firm's cadence instead of assuming a number.
    payout_per_cycle: Decimal | None = Field(default=None, ge=0)
    """Gross profit typically withdrawn per payout cycle on the funded account."""

    days_between_payouts: int | None = Field(default=None, gt=0)
    """Minimum days between payouts on the funded account."""

    funded_horizon_days: int = Field(default=180, gt=0)
    """Modelled funded-account lifespan (days) used to count payout cycles."""

    allowed_classes: tuple[str, ...] = ()
    """Instrument classes this firm actually offers (e.g. ``index``, ``micro``,
    ``rates``, ``energy``, ``metals``, ``forex``). Empty means unrestricted. Evaluating
    a firm only on instruments it lists prevents the category error of testing, say,
    a CFD firm on futures it does not sell."""

    def allows_kind(self, kind: str) -> bool:
        """True if this firm offers instruments of ``kind`` (empty list = any)."""
        return not self.allowed_classes or kind in self.allowed_classes

    def phase(self, phase: AccountPhase) -> FirmPhaseConfig:
        """Return the config for ``phase`` (KeyError if absent)."""
        for p in self.phases:
            if p.phase is phase:
                return p
        raise KeyError(f"firm {self.id!r} has no phase {phase}")

    def funded_lifecycle(self, horizon_days: int | None = None) -> tuple[Decimal, int]:
        """Return ``(payout_per_cycle, max_cycles)`` for the funded account.

        ``max_cycles`` is ``max(0, horizon // days_between_payouts)`` over the
        given ``horizon_days`` (falling back to ``self.funded_horizon_days``).

        When the firm has no funded terms (``payout_per_cycle`` or
        ``days_between_payouts`` is ``None``) this returns ``(Decimal("0"), 0)``
        so a firm with no modelled payout terms yields zero funded payouts rather
        than assumed ones.
        """
        if self.payout_per_cycle is None or self.days_between_payouts is None:
            return Decimal("0"), 0
        horizon = horizon_days if horizon_days is not None else self.funded_horizon_days
        max_cycles = max(0, horizon // self.days_between_payouts)
        return self.payout_per_cycle, max_cycles


BUNDLED_CONFIGS = Path(__file__).resolve().parent / "configs"
"""The firm configs shipped with the package (real, published programs only)."""


class FirmRegistry:
    """In-memory registry of firms loaded from YAML."""

    def __init__(self, firms: dict[str, FirmConfig] | None = None) -> None:
        self._firms: dict[str, FirmConfig] = dict(firms or {})

    @classmethod
    def from_dir(cls, path: str | Path) -> FirmRegistry:
        """Load every ``*.yaml``/``*.yml`` file in ``path`` into the registry."""
        directory = Path(path)
        firms: dict[str, FirmConfig] = {}
        for file in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
            firm = FirmConfig.model_validate(data)
            if firm.id in firms:
                raise ValueError(f"duplicate firm id {firm.id!r} in {file}")
            firms[firm.id] = firm
        return cls(firms)

    @classmethod
    def bundled(cls) -> FirmRegistry:
        """Load the firm configs that ship with the package."""
        return cls.from_dir(BUNDLED_CONFIGS)

    def firm(self, firm_id: str) -> FirmConfig:
        """Return the firm config (KeyError if unknown)."""
        return self._firms[firm_id]

    def rules_for(self, firm_id: str, phase: AccountPhase) -> PropFirmRules:
        """Return the PropFirmRules for a firm/phase (KeyError if unknown)."""
        return self.firm(firm_id).phase(phase).rules

    def list_firms(self) -> list[str]:
        """Return the ids of all loaded firms."""
        return list(self._firms)
