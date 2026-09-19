"""Unit tests for the append-only hash-chained compliance log."""

from __future__ import annotations

import json
from pathlib import Path

from riskledger.audit.compliance_log import GENESIS_HASH, ComplianceLog


def test_append_and_verify(tmp_path: Path) -> None:
    log = ComplianceLog(tmp_path / "audit.jsonl")
    e0 = log.append("SIGNAL", {"id": "s1", "state": "VALIDATED"})
    e1 = log.append("KILL_90", {"account": "acc-1", "drawdown": "1800"})
    assert e0.seq == 0 and e0.prev_hash == GENESIS_HASH
    assert e1.seq == 1 and e1.prev_hash == e0.hash
    assert log.verify() is True
    assert len(log.entries()) == 2


def test_chain_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = ComplianceLog(path)
    log.append("A", {"x": 1})
    log.append("B", {"x": 2})
    # reopen
    reopened = ComplianceLog(path)
    e = reopened.append("C", {"x": 3})
    assert e.seq == 2
    assert reopened.verify() is True


def test_tamper_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = ComplianceLog(path)
    log.append("A", {"x": 1})
    log.append("B", {"x": 2})
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["payload"] = {"x": 999}  # tamper with the payload, keep the old hash
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert ComplianceLog(path).verify() is False
