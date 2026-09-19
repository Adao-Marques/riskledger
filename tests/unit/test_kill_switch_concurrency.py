"""The KillSwitch latch must be atomic, not merely idempotent.

``tests/unit/test_kill_switch.py`` covers sequential idempotency (call twice, get
the same result). That is a weaker property than the one the design actually promises: exactly ONE
switch is shared per account so that concurrent callers — the execution engine
and the worker, a guard breach and a stale watchdog — cannot both flatten. A
check-then-set on a plain bool does not give that; these tests pin that it holds
when the callers really do race.
"""

from __future__ import annotations

import asyncio
import threading
import time
from decimal import Decimal
from pathlib import Path

from riskledger.audit.compliance_log import ComplianceLog
from riskledger.core.clock import utcnow
from riskledger.core.enums import AccountStatus, Direction
from riskledger.core.types import Account, Fill, Order, Position
from riskledger.risk.kill_switch import KillResult, KillSwitch


class CountingBroker:
    """Thread-safe call-counting broker; can park inside the flatten sequence."""

    def __init__(self, *, gate: threading.Event | None = None,
                 entered: threading.Event | None = None) -> None:
        self._lock = threading.Lock()
        self._positions = [Position("ES", Direction.LONG, Decimal("2"), Decimal("100"))]
        self._gate = gate
        self._entered = entered
        self.cancel_calls = 0
        self.flatten_calls = 0

    def submit(self, order: Order) -> Fill:  # pragma: no cover - unused
        raise NotImplementedError

    def cancel_all(self) -> int:
        with self._lock:
            self.cancel_calls += 1
        if self._entered is not None:
            self._entered.set()
        if self._gate is not None:
            # Park the winner mid-sequence so every other caller is guaranteed to
            # arrive while the fire is genuinely in flight.
            self._gate.wait(timeout=5)
        return 2

    def flatten(self) -> list[Fill]:
        with self._lock:
            self.flatten_calls += 1
            self._positions = []
        return [Fill("o1", "ES", Direction.SHORT, Decimal("1"), Decimal("100"), utcnow())]

    def get_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions)

    def get_equity(self) -> Decimal:  # pragma: no cover - unused
        return Decimal("50000")

    def set_mark(self, symbol: str, price: Decimal) -> None:  # pragma: no cover
        pass


def _account() -> Account:
    return Account.open("acc-1", "topstep", Decimal("50000"))


def _race(ks: KillSwitch, n: int) -> list[KillResult]:
    """Fire ``ks`` from ``n`` threads released simultaneously by a barrier."""
    results: list[KillResult | None] = [None] * n
    barrier = threading.Barrier(n)

    def run(i: int) -> None:
        barrier.wait(timeout=5)
        results[i] = ks.trigger(f"racer-{i}")

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "trigger() deadlocked"
    out = [r for r in results if r is not None]
    assert len(out) == n
    return out


class WidenedLatchSwitch(KillSwitch):
    """The production :meth:`KillSwitch.trigger`, with the latch seam widened.

    On a GIL build the window between reading ``_firing`` and setting it is ~3
    bytecodes, so a plain thread race almost never lands in it: measured 0 hits
    in 200 rounds against a deliberately unlocked implementation. A test that
    cannot fail on the buggy code is not a regression test.

    So we widen the seam instead of hoping for it. ``_firing`` becomes a property
    whose accessors yield to the scheduler; ``trigger()`` itself is untouched
    production code. An unlocked check-then-set then double-fires in ~14 of 15
    rounds, while the mutex-guarded one never does — because under the mutex the
    widened read happens *inside* the critical section, where a yield is harmless.
    """

    @property
    def _firing(self) -> bool:
        time.sleep(0)  # force a scheduler switch at the check-and-set seam
        return self.__firing

    @_firing.setter
    def _firing(self, value: bool) -> None:
        time.sleep(0)
        self.__firing = value


def test_racing_threads_flatten_exactly_once(tmp_path: Path) -> None:
    """16 threads firing at once: ONE flatten, ONE cancel, ONE audit record.

    This is the property the shared-per-account switch exists to guarantee. N
    concurrent flattens on a killed account means duplicate orders — the failure
    the single shared switch exists to prevent. Run over several rounds because the
    interleaving is still scheduler-dependent even with the seam widened.
    """
    for round_ in range(12):
        broker = CountingBroker()
        acc = _account()
        log = ComplianceLog(tmp_path / f"audit-{round_}.jsonl")
        ks = WidenedLatchSwitch(broker, acc, log)

        results = _race(ks, 16)

        assert broker.flatten_calls == 1, (
            f"round {round_}: {broker.flatten_calls} concurrent flattens — the "
            "latch is not atomic (duplicate orders on a killed account)")
        assert broker.cancel_calls == 1, f"round {round_}: {broker.cancel_calls} cancels"
        assert [e.event_type for e in log.entries()] == ["KILL_90"], (
            f"round {round_}: {len(log.entries())} audit records for one kill")
        assert log.verify() is True
        assert acc.status is AccountStatus.KILL_SWITCH_TRIGGERED
        assert len({id(r) for r in results}) == 1, (
            f"round {round_}: racers saw different kill results")


def test_plain_thread_race_is_also_clean(tmp_path: Path) -> None:
    """The same race without the widened seam, on the unmodified class."""
    broker = CountingBroker()
    acc = _account()
    log = ComplianceLog(tmp_path / "audit.jsonl")
    ks = KillSwitch(broker, acc, log)
    results = _race(ks, 32)
    assert broker.flatten_calls == 1
    assert [e.event_type for e in log.entries()] == ["KILL_90"]
    assert len({id(r) for r in results}) == 1
    assert acc.status is AccountStatus.KILL_SWITCH_TRIGGERED


def test_in_flight_callers_get_the_winners_result(tmp_path: Path) -> None:
    """Callers arriving mid-fire wait for, and receive, the real KillResult.

    Deterministic: the winner is parked inside ``cancel_all`` until every other
    caller has arrived, so they are guaranteed to hit the in-flight path.
    """
    gate, entered = threading.Event(), threading.Event()
    broker = CountingBroker(gate=gate, entered=entered)
    acc = _account()
    log = ComplianceLog(tmp_path / "audit.jsonl")
    ks = KillSwitch(broker, acc, log)

    winner: list[KillResult] = []
    first = threading.Thread(target=lambda: winner.append(ks.trigger("winner")))
    first.start()
    assert entered.wait(timeout=5), "winner never entered the flatten sequence"
    assert ks.triggered is True  # latched the moment the fire was claimed

    losers: list[KillResult | None] = [None] * 8
    started = threading.Barrier(9)

    def loser(i: int) -> None:
        started.wait(timeout=5)
        losers[i] = ks.trigger(f"loser-{i}")

    threads = [threading.Thread(target=loser, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    started.wait(timeout=5)
    gate.set()  # release the winner; the losers must now unblock with its result
    first.join(timeout=10)
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in [first, *threads]), "trigger() deadlocked"

    assert broker.flatten_calls == 1
    assert broker.cancel_calls == 1
    assert [e.event_type for e in log.entries()] == ["KILL_90"]
    assert all(r is winner[0] for r in losers), (
        "an in-flight caller got a placeholder instead of the real kill result")
    assert acc.status is AccountStatus.KILL_SWITCH_TRIGGERED


def test_reentrant_trigger_from_the_firing_thread_does_not_deadlock() -> None:
    """A broker callback re-entering trigger() must return, not block on itself."""
    acc = _account()
    seen: list[KillResult] = []

    class ReentrantBroker(CountingBroker):
        def __init__(self) -> None:
            super().__init__()
            self.switch: KillSwitch | None = None

        def cancel_all(self) -> int:
            assert self.switch is not None
            seen.append(self.switch.trigger("re-entrant"))  # same thread
            return super().cancel_all()

    broker = ReentrantBroker()
    ks = KillSwitch(broker, acc)
    broker.switch = ks
    done = threading.Thread(target=lambda: ks.trigger("outer"))
    done.start()
    done.join(timeout=5)
    assert not done.is_alive(), "re-entrant trigger() deadlocked on its own latch"
    assert broker.flatten_calls == 1
    assert len(seen) == 1 and seen[0].cancelled == 0  # in-flight marker, no side effects
    assert acc.status is AccountStatus.KILL_SWITCH_TRIGGERED


def test_audit_sink_failure_still_suspends_the_account() -> None:
    """A failing audit sink must not abort the suspension or raise out of trigger()."""
    class BrokenLog:
        def append(self, *a: object, **k: object) -> None:
            raise OSError("audit disk full")

    broker = CountingBroker()
    acc = _account()
    ks = KillSwitch(broker, acc, BrokenLog())  # type: ignore[arg-type]
    res = ks.trigger("x")
    assert res.verified is True
    assert acc.status is AccountStatus.KILL_SWITCH_TRIGGERED


def test_shared_switch_across_two_event_loops_fires_once(tmp_path: Path) -> None:
    """The async path: two loops in two threads sharing ONE switch still fire once.

    ``Orchestrator.run_all`` gathers accounts on a single cooperative loop, so
    tasks there cannot interleave inside the synchronous ``trigger()``. The
    guarantee that matters is the one the shared-switch design makes: if the same
    account is ever driven from more than one loop/thread, the latch still holds.
    """
    broker = CountingBroker()
    acc = _account()
    log = ComplianceLog(tmp_path / "audit.jsonl")
    ks = KillSwitch(broker, acc, log)
    barrier = threading.Barrier(2)
    results: list[KillResult] = []
    lock = threading.Lock()

    async def fire(tag: str) -> None:
        await asyncio.sleep(0)
        barrier.wait(timeout=5)
        res = ks.trigger(tag)
        with lock:
            results.append(res)

    def loop(tag: str) -> None:
        asyncio.run(fire(tag))

    threads = [threading.Thread(target=loop, args=(f"loop-{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads)
    assert broker.flatten_calls == 1
    assert [e.event_type for e in log.entries()] == ["KILL_90"]
    assert len({id(r) for r in results}) == 1  # both loops saw the same single kill
