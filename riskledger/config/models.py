"""Validated configuration models (Pydantic v2).

Configuration over code: prop firm rules, accounts and strategies live in
versioned YAML and are validated here. Monetary fields are ``Decimal``.
"""

from __future__ import annotations

from datetime import time
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.enums import AccountPhase, DrawdownType


class PropFirmRules(BaseModel):
    """Risk rules for one prop firm *phase*.

    Semantics (all amounts are absolute account currency unless noted):

    * ``drawdown_type`` — how ``max_total_drawdown`` is measured:
      - ``static``:   limit floor = ``initial_balance - max_total_drawdown`` (fixed).
      - ``trailing``: limit floor = ``high_watermark - max_total_drawdown``; the
        high-watermark must be updated on every tick. If
        ``trailing_locks_at_initial`` is set, the floor stops trailing once it
        reaches ``initial_balance`` (common Topstep-style rule).
      - ``eod``:      like trailing but the high-watermark only ratchets at the
        end of day (close).
    * ``max_daily_loss`` — max loss from the day's starting equity (None = no rule).
    * ``profit_target`` — equity gain required to pass this phase (None = funded).
    * ``kill_switch_threshold`` — fraction (0..1) of the drawdown buffer that, once
      consumed, triggers the Kill Switch. 0.90 means fire at 90% of the way to the
      limit floor (i.e. before an actual breach).

    Real prop-firm constraints also modelled:

    * ``max_contracts`` — hard position/contract cap for this phase (e.g. Topstep
      50k = 5). Scaling plans are expressed as different phases. Bounds sizing.
    * ``max_eval_days`` — days allowed to pass the evaluation (None = unlimited).
    * ``min_trading_days`` — minimum days traded before passing / withdrawing.
    * ``min_winning_days`` — minimum profitable days required (funded consistency).
    * ``consistency_max_day_pct`` — no single day may exceed this fraction of total
      profit (funded consistency rule).
    * ``payout_buffer`` — equity above the starting balance that must be preserved
      to withdraw (the firm's "safety net").
    * ``overnight_allowed`` / ``weekend_flat`` — whether positions may be held past
      the session close / over the weekend (futures evals are usually flat).
    * ``mandatory_flat_utc`` — time-of-day by which all positions must be closed.
    * ``news_blackout`` — trading restricted around high-impact news.
    """

    model_config = ConfigDict(frozen=True)

    drawdown_type: DrawdownType
    max_total_drawdown: Decimal = Field(gt=0)
    max_daily_loss: Decimal | None = Field(default=None, gt=0)
    profit_target: Decimal | None = Field(default=None, gt=0)
    min_trading_days: int = Field(default=0, ge=0)
    trailing_locks_at_initial: bool = False
    news_blackout: bool = False
    mandatory_flat_utc: time | None = None
    consistency_max_day_pct: Decimal | None = Field(default=None, gt=0, le=1)
    # Real prop-firm constraints (all optional; defaults preserve prior behaviour).
    max_contracts: int | None = Field(default=None, gt=0)
    max_eval_days: int | None = Field(default=None, gt=0)
    min_winning_days: int = Field(default=0, ge=0)
    payout_buffer: Decimal | None = Field(default=None, ge=0)
    overnight_allowed: bool = False
    weekend_flat: bool = True
    kill_switch_threshold: Decimal = Field(default=Decimal("0.90"), gt=0, le=1)
    alert_threshold: Decimal = Field(default=Decimal("0.70"), gt=0, le=1)
    block_threshold: Decimal = Field(default=Decimal("0.80"), gt=0, le=1)

    @model_validator(mode="after")
    def _check_threshold_order(self) -> PropFirmRules:
        if not (self.alert_threshold < self.block_threshold < self.kill_switch_threshold):
            raise ValueError("thresholds must satisfy alert < block < kill_switch")
        return self


class AccountConfig(BaseModel):
    """A single managed account."""

    model_config = ConfigDict(frozen=True)

    id: str
    prop_firm: str
    phase: AccountPhase = AccountPhase.EVAL_F1
    initial_balance: Decimal = Field(gt=0)
    rules: PropFirmRules


class StrategyConfig(BaseModel):
    """Configuration of a strategy instance."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    timeframe: str
    symbols: tuple[str, ...]
    preferred_regime: str | None = None
    active: bool = True
    params: dict[str, object] = Field(default_factory=dict)
