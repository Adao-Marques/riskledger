"""Append-only, hash-chained compliance log.

Every decision (signal, rejection, risk event, kill switch, fill, phase
transition) is recorded as an immutable entry. Each entry stores the hash of the
previous entry, so any tampering with history is detectable via
:meth:`ComplianceLog.verify`.

Storage is a JSON-lines file (one entry per line), opened in append mode. This
is deliberately simple and dependency-free; a SQL-backed implementation can be
added later behind the same interface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.clock import utcnow

GENESIS_HASH = "0" * 64


def _canonical(payload: dict[str, Any]) -> str:
    """Deterministic JSON serialization for hashing (sorted keys, no spaces)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One immutable record in the chain."""

    seq: int
    ts: str
    event_type: str
    payload: dict[str, Any]
    prev_hash: str
    hash: str

    @staticmethod
    def compute_hash(seq: int, ts: str, event_type: str,
                     payload: dict[str, Any], prev_hash: str) -> str:
        material = _canonical(
            {"seq": seq, "ts": ts, "event_type": event_type,
             "payload": payload, "prev_hash": prev_hash}
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ComplianceLog:
    """Append-only hash-chained audit log persisted as JSON lines."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = GENESIS_HASH
        self._seq = -1
        self._load_tail()

    def _load_tail(self) -> None:
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            self._seq = rec["seq"]
            self._last_hash = rec["hash"]

    def append(self, event_type: str, payload: dict[str, Any],
               ts: datetime | None = None) -> AuditEntry:
        """Append an entry and return it. Idempotency is the caller's concern."""
        seq = self._seq + 1
        ts_str = (ts or utcnow()).isoformat()
        entry_hash = AuditEntry.compute_hash(seq, ts_str, event_type, payload, self._last_hash)
        entry = AuditEntry(
            seq=seq, ts=ts_str, event_type=event_type,
            payload=payload, prev_hash=self._last_hash, hash=entry_hash,
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(_canonical(asdict(entry)) + "\n")
        self._seq = seq
        self._last_hash = entry_hash
        return entry

    def entries(self) -> list[AuditEntry]:
        """Read all entries in order."""
        out: list[AuditEntry] = []
        if not self._path.exists():
            return out
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(AuditEntry(**json.loads(line)))
        return out

    def verify(self) -> bool:
        """Return True iff the hash chain is intact and unbroken."""
        prev = GENESIS_HASH
        for i, entry in enumerate(self.entries()):
            if entry.seq != i or entry.prev_hash != prev:
                return False
            expected = AuditEntry.compute_hash(
                entry.seq, entry.ts, entry.event_type, entry.payload, entry.prev_hash
            )
            if expected != entry.hash:
                return False
            prev = entry.hash
        return True
