# SPDX-License-Identifier: Apache-2.0
"""Native gather and bounded bounce lifetime contracts, without a GPU."""

# Standard
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from unittest.mock import Mock
import gc
import logging
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.dax_coordinated_l1.staged_retrieve import (
    StagedRetrievePool,
    StagedRetrieveRetiredError,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
import lmcache.v1.distributed.dax_coordinated_l1.staged_retrieve as mod

pytestmark = pytest.mark.no_shared_allocator


def _object(size: int, value: int = 37) -> TensorMemoryObj:
    raw = torch.full((size,), value, dtype=torch.uint8)
    return TensorMemoryObj(
        raw,
        MemoryObjMetadata(
            shape=torch.Size([size]),
            dtype=torch.uint8,
            address=(1 << 38) + 17,
            phy_size=size,
            ref_count=1,
            fmt=MemoryFormat.KV_2LTD,
            shapes=[torch.Size([size])],
            dtypes=[torch.uint8],
        ),
        None,
    )


class _Handle:
    """Completion handle double: done at once, or once an Event is set."""

    def __init__(self, done: threading.Event | bool = True) -> None:
        self._done = done

    def query(self) -> bool:
        if isinstance(self._done, bool):
            return self._done
        return self._done.is_set()

    def synchronize(self) -> None:
        if not isinstance(self._done, bool):
            self._done.wait()


class _RaisingHandle:
    """Completion handle whose query raises a fresh error after ``after`` polls."""

    def __init__(self, after: int = 0) -> None:
        self._after = after
        self.calls = 0

    def query(self) -> bool:
        self.calls += 1
        if self.calls > self._after:
            raise RuntimeError("lost GPU")
        return False

    def synchronize(self) -> None:
        raise RuntimeError("lost GPU")


def _done() -> _Handle:
    return _Handle()


def _capture(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # LMCache loggers do not propagate; attach the capture handler directly.
    monkeypatch.setattr(mod.logger, "handlers", [*mod.logger.handlers, caplog.handler])


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.001)


def _exhausted_errors(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
        and "every staged retrieve slot is retired" in record.getMessage()
    ]


@pytest.fixture(autouse=True)
def cpu_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only pinning is mocked; every gather uses the real native extension."""
    empty = torch.empty

    def allocate(*args, **kwargs):
        assert kwargs.pop("pin_memory") is True
        return empty(*args, **kwargs)

    monkeypatch.setattr(mod.torch, "empty", allocate)


def test_native_bytes_layout_offsets_and_reuse() -> None:
    objects = [_object(65, i + 1) for i in range(7)]
    before = [replace(obj.meta) for obj in objects]
    pointers = []
    observed = []
    records = []

    def enqueue(start, bounce):
        pointers.append(bounce[0].data_ptr)
        assert [obj.meta.address for obj in bounce] == [0, 128][: len(bounce)]
        for obj, source in zip(bounce, objects[start:], strict=False):
            assert torch.equal(obj.raw_tensor, source.raw_tensor)
            assert obj.meta.shapes == source.meta.shapes
            assert obj.meta.dtypes == source.meta.dtypes
            assert obj.meta.fmt == source.meta.fmt
            assert obj.meta is not source.meta
            assert obj.meta.shapes is not source.meta.shapes
            observed.append(source)

    with closing(StagedRetrievePool(1, 256, 2)) as pool:
        with pool.session() as transfer:
            transfer(objects, enqueue, lambda: records.append(True) or _Handle())
        # A completed handle is reclaimed at the next acquisition, so every
        # batch reused the single slot.
        assert len(set(pointers)) == 1
        assert len(records) == 4
        assert observed == objects
        status = pool.report_status()
        assert status["allocated_bytes"] == status["capacity_bytes"] == 256
        assert status["gather_bytes"] == 7 * 65
        assert status["slots_in_use"] == 0
        assert status["in_flight_slots"] == 0
        # One completion check per acquisition that found the slot in flight
        # (three), plus the one this status report made for the last batch.
        assert status["event_queries"] == 4
        assert status["slot_waits"] == 0
    assert [obj.meta for obj in objects] == before


def test_transfer_returns_before_dma_completes() -> None:
    completions = [threading.Event() for _ in range(3)]
    handles = iter(completions)
    pointers = []

    def enqueue(_start, bounce):
        pointers.append(bounce[0].data_ptr)

    with closing(StagedRetrievePool(3, 128, 1)) as pool:
        with pool.session() as transfer:
            transfer(
                [_object(128, i + 1) for i in range(3)],
                enqueue,
                lambda: _Handle(next(handles)),
            )
            # Back from the transfer while every DMA is still pending.
            status = pool.report_status()
            assert status["active_sessions"] == 1
            assert status["in_flight_slots"] == 3
            assert status["slots_in_use"] == 3
            assert status["available_slots"] == 0
        assert len(set(pointers)) == 3
        status = pool.report_status()
        assert status["active_sessions"] == 0
        assert status["in_flight_slots"] == 3
        queries = status["event_queries"]
        # Each observed completion is reclaimed by the next status report.
        for expected, complete in zip((2, 1, 0), completions, strict=True):
            complete.set()
            status = pool.report_status()
            assert status["in_flight_slots"] == expected
            assert status["slots_in_use"] == expected
            assert status["available_slots"] == 3 - expected
        assert status["event_queries"] > queries
        assert status["event_timeouts"] == 0
        assert status["event_timeout_s"] == 30.0
        assert status["close_timeout_s"] == 30.0
    status = pool.report_status()
    assert status["closed"] is True
    assert status["allocated_bytes"] == 0
    assert status["quarantined_at_close"] == 0


def test_delayed_dma_saturation_and_close_make_progress() -> None:
    pool = StagedRetrievePool(1, 128, 2)
    complete = threading.Event()
    second_enqueued = threading.Event()

    def first():
        with pool.session() as transfer:
            transfer([_object(128)], lambda *_: None, lambda: _Handle(complete))

    def second():
        with pool.session() as transfer:
            transfer([_object(128, 99)], lambda *_: second_enqueued.set(), _done)

    with ThreadPoolExecutor(3) as executor:
        a = executor.submit(first)
        # The first pipeline returns while its DMA is still pending.
        a.result(5)
        assert not complete.is_set()
        status = pool.report_status()
        assert status["active_sessions"] == 0
        assert status["in_flight_slots"] == 1
        assert status["slots_in_use"] == 1
        # The second pipeline waits for that completion inside its acquire.
        b = executor.submit(second)
        _wait_until(lambda: pool.report_status()["slot_waits"] == 1)
        assert not second_enqueued.wait(0.05)
        # Close waits for the admitted session, which waits for the DMA.
        c = executor.submit(pool.close)
        assert not second_enqueued.wait(0.05)
        assert not c.done()
        assert pool.report_status()["closing"] is True
        complete.set()
        b.result(5)
        c.result(5)
    assert second_enqueued.is_set()
    status = pool.report_status()
    assert status["peak_slots"] == 1
    assert status["slot_waits"] == 1
    assert status["event_timeouts"] == 0
    assert status["quarantined_at_close"] == 0
    assert status["allocated_bytes"] == 0
    with pytest.raises(RuntimeError, match="closing"):
        with pool.session():
            pass


@pytest.mark.parametrize("completes", [True, False])
def test_close_waits_for_in_flight_dma_or_quarantines_it(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    completes: bool,
) -> None:
    _capture(monkeypatch, caplog)
    # The registry is process-global; isolate this test's entries.
    monkeypatch.setattr(mod, "_RETIRED_BUFFERS", [])
    pool = StagedRetrievePool(1, 128, 1, close_timeout_s=0.2)
    complete = threading.Event()
    pointers = []

    def enqueue(_start, bounce):
        pointers.append(bounce[0].data_ptr)

    with pool.session() as transfer:
        transfer([_object(128)], enqueue, lambda: _Handle(complete))
    assert pool.report_status()["in_flight_slots"] == 1
    with ThreadPoolExecutor(1) as executor:
        began = time.monotonic()
        closing_pool = executor.submit(pool.close)
        # No session is active; close is waiting for the in-flight DMA.
        with pytest.raises(TimeoutError):
            closing_pool.result(0.05)
        if completes:
            complete.set()
        closing_pool.result(5)
        elapsed = time.monotonic() - began
    status = pool.report_status()
    assert status["closed"] is True
    assert status["in_flight_slots"] == 0
    if completes:
        assert status["quarantined_at_close"] == 0
        assert status["quarantined_slots"] == 0
        assert status["allocated_bytes"] == 0
        assert mod._RETIRED_BUFFERS == []
        assert "still in flight" not in caplog.text
        return
    assert elapsed >= 0.2
    assert status["quarantined_at_close"] == 1
    assert status["quarantined_slots"] == 1
    assert status["slots_in_use"] == 1
    assert status["allocated_bytes"] == 128
    assert [buffer.data_ptr() for buffer in mod._RETIRED_BUFFERS] == pointers
    assert "1 bounce slot(s) still in flight after 0.2 s" in caplog.text
    assert "keeping 1 retired bounce buffer(s), 128 bytes" in caplog.text
    # Quarantine is not the exhausted transition: no ERROR, the pool closes.
    assert not [r for r in caplog.records if r.levelno == logging.ERROR]
    pool.close()  # idempotent: nothing left in flight, buffer registered once
    status = pool.report_status()
    assert status["quarantined_at_close"] == 1
    assert status["allocated_bytes"] == 128
    assert len(mod._RETIRED_BUFFERS) == 1


@pytest.mark.parametrize("slots", [1, 2])
def test_event_timeout_retires_in_flight_slot(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, slots: int
) -> None:
    _capture(monkeypatch, caplog)
    monkeypatch.setattr(mod, "_RETIRED_BUFFERS", [])
    pool = StagedRetrievePool(slots, 128, 1, event_timeout_s=0.2)
    complete = threading.Event()
    pointers = []

    def enqueue(_start, bounce):
        pointers.append(bounce[0].data_ptr)

    with pool.session() as transfer:
        # The first DMA never completes; with two slots the second completes
        # while the next acquisition is already waiting on the first.
        transfer([_object(128)], enqueue, lambda: _Handle(False))
        if slots == 2:
            transfer([_object(128)], enqueue, lambda: _Handle(complete))
    assert pool.report_status()["in_flight_slots"] == slots
    threading.Timer(0.05, complete.set).start()
    with pool.session() as transfer:
        if slots == 2:
            # Waiting on the oldest in-flight slot times out and retires it;
            # the acquisition continues on the other slot.
            transfer([_object(128)], enqueue, _done)
            assert pointers[2] == pointers[1] != pointers[0]
        else:
            with pytest.raises(StagedRetrieveRetiredError, match="every staged"):
                transfer([_object(128)], enqueue, _done)
    status = pool.report_status()
    assert status["event_timeouts"] == 1
    assert status["quarantined_slots"] == 1
    assert status["slot_waits"] == 1
    assert status["in_flight_slots"] == 0
    assert status["available_slots"] == slots - 1
    assert status["slots_in_use"] == 1
    assert pool.exhausted is (slots == 1)
    assert [buffer.data_ptr() for buffer in mod._RETIRED_BUFFERS] == [pointers[0]]
    assert "staged DMA did not complete within 0.2 s; retiring bounce slot" in (
        caplog.text
    )
    assert f"(1 of {slots} retired)" in caplog.text
    assert len(_exhausted_errors(caplog)) == (1 if slots == 1 else 0)
    pool.close()
    status = pool.report_status()
    assert status["quarantined_at_close"] == 0
    assert status["allocated_bytes"] == 128  # only the retired buffer


@pytest.mark.parametrize("where", ["reclaim", "wait"])
def test_query_exception_retires_slot(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, where: str
) -> None:
    _capture(monkeypatch, caplog)
    monkeypatch.setattr(mod, "_RETIRED_BUFFERS", [])
    pool = StagedRetrievePool(2, 128, 1)
    complete = threading.Event()
    pointers = []

    def enqueue(_start, bounce):
        pointers.append(bounce[0].data_ptr)

    # "reclaim": the first poll, made while reclaiming, raises. "wait": the
    # reclaim poll sees a pending DMA and the waiting poll raises.
    handle = _RaisingHandle(after=0 if where == "reclaim" else 1)
    with pool.session() as transfer:
        transfer([_object(128)], enqueue, lambda: handle)
        if where == "wait":
            transfer([_object(128)], enqueue, lambda: _Handle(complete))
    if where == "reclaim":
        assert pool.report_status()["quarantined_slots"] == 1
    else:
        threading.Timer(0.05, complete.set).start()
    # The other slot keeps serving; the retired buffer is never handed out.
    with pool.session() as transfer:
        transfer([_object(128)], enqueue, _done)
    assert pointers[-1] != pointers[0]
    status = pool.report_status()
    assert status["quarantined_slots"] == 1
    assert status["event_timeouts"] == 0
    assert status["in_flight_slots"] == 0
    assert status["available_slots"] == 1
    assert status["slot_waits"] == (1 if where == "wait" else 0)
    assert pool.exhausted is False
    assert [buffer.data_ptr() for buffer in mod._RETIRED_BUFFERS] == [pointers[0]]
    assert "staged DMA completion query failed; retiring bounce slot" in caplog.text
    assert "(1 of 2 retired)" in caplog.text
    assert "lost GPU" in caplog.text  # the query error's traceback is logged
    assert not _exhausted_errors(caplog)
    pool.close()


def test_cpu_completion_precedes_h2d(monkeypatch: pytest.MonkeyPatch) -> None:
    copied = threading.Event()
    continue_copy = threading.Event()
    enqueue = Mock()
    real_pool = mod.native.DevDaxCopyPool

    class DelayedPool:
        num_threads = 1

        def __init__(self, _threads):
            self.pool = real_pool(1)

        def submit(self, *args):
            batch = self.pool.submit(*args)

            def wait():
                copied.set()
                assert continue_copy.wait(5)
                batch.wait()

            return type("Batch", (), {"wait": staticmethod(wait)})()

        def shutdown(self):
            self.pool.shutdown()

    monkeypatch.setattr(mod.native, "DevDaxCopyPool", DelayedPool)
    with closing(StagedRetrievePool(1, 128, 1)) as pool:

        def run():
            with pool.session() as transfer:
                transfer([_object(128)], enqueue, _done)

        with ThreadPoolExecutor(1) as executor:
            result = executor.submit(run)
            assert copied.wait(5)
            enqueue.assert_not_called()
            continue_copy.set()
            result.result(5)
        enqueue.assert_called_once()


@pytest.mark.parametrize("failure", ["submit", "wait", "enqueue"])
def test_failure_after_first_batch_settles_before_release(monkeypatch, failure) -> None:
    real_pool = mod.native.DevDaxCopyPool
    recorded = []
    enqueued = []
    pending = threading.Event()

    class FailingPool:
        num_threads = 1

        def __init__(self, _threads):
            self.pool = real_pool(1)
            self.calls = 0

        def submit(self, *args):
            self.calls += 1
            if self.calls == 2 and failure == "submit":
                raise RuntimeError("injected submit")
            batch = self.pool.submit(*args)
            fail_wait = self.calls == 2 and failure == "wait"

            def wait():
                batch.wait()
                if fail_wait:
                    raise RuntimeError("injected wait")

            return type("Batch", (), {"wait": staticmethod(wait)})()

        def shutdown(self):
            self.pool.shutdown()

    def enqueue(start, bounce):
        enqueued.append(start)
        assert torch.all(bounce[0].raw_tensor == 37)
        if start == 1 and failure == "enqueue":
            raise RuntimeError("injected enqueue after partial DMA")

    def record():
        recorded.append(list(enqueued))
        # The first batch completes at once; a later batch stays pending.
        return _Handle(True if len(recorded) == 1 else pending)

    monkeypatch.setattr(mod.native, "DevDaxCopyPool", FailingPool)
    with closing(StagedRetrievePool(1, 128, 1)) as pool:
        with pool.session() as transfer:
            with pytest.raises(RuntimeError, match="injected"):
                transfer([_object(128) for _ in range(3)], enqueue, record)
        status = pool.report_status()
        if failure == "enqueue":
            # The partial enqueue was recorded: the slot stays in flight
            # until the GPU is done with it, then returns to the pool.
            assert recorded == [[0], [0, 1]]
            assert status["in_flight_slots"] == 1
            assert status["slots_in_use"] == 1
            assert status["available_slots"] == 0
            pending.set()
            status = pool.report_status()
        else:
            # A copy failure before record returns the slot at once.
            assert recorded == [[0]]
        assert status["in_flight_slots"] == 0
        assert status["slots_in_use"] == 0
        assert status["quarantined_slots"] == 0


def test_failed_record_retires_slot_and_close_keeps_its_buffer(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _capture(monkeypatch, caplog)
    monkeypatch.setattr(mod, "_RETIRED_BUFFERS", [])
    pool = StagedRetrievePool(2, 128, 1)
    pointers = []

    def enqueue(_start, bounce):
        pointers.append(bounce[0].data_ptr)

    record = Mock(side_effect=RuntimeError("lost GPU"))
    with pool.session() as transfer:
        with pytest.raises(RuntimeError, match="retired"):
            transfer([_object(128)], enqueue, record)
    status = pool.report_status()
    assert status["quarantined_slots"] == 1
    assert status["slots_in_use"] == 1
    assert status["in_flight_slots"] == 0
    assert status["allocated_bytes"] == 128
    assert "completion could not be recorded; retiring bounce slot" in caplog.text
    assert "(1 of 2 retired)" in caplog.text
    # The retired buffer went to the module registry at retire time.
    assert [buffer.data_ptr() for buffer in mod._RETIRED_BUFFERS] == [pointers[0]]
    # The remaining slot keeps serving; the retired one is never handed out.
    for _ in range(2):
        with pool.session() as transfer:
            transfer([_object(128)], enqueue, _done)
    assert pointers[1] == pointers[2] != pointers[0]
    status = pool.report_status()
    assert status["slots_in_use"] == 1  # the retired slot
    assert status["available_slots"] == 1  # the serving slot, returned
    pool.close()
    status = pool.report_status()
    assert status["closed"] is True
    assert status["allocated_bytes"] == 128
    assert status["quarantined_at_close"] == 0
    assert "keeping 1 retired bounce buffer(s), 128 bytes" in caplog.text
    with pytest.raises(RuntimeError, match="closing"):
        with pool.session():
            pass
    # The registry, not the pool, keeps the buffer alive past the pool.
    del pool
    gc.collect()
    assert len(mod._RETIRED_BUFFERS) == 1


def test_retire_in_later_batch_releases_once_and_close_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three 128 B objects in 128 B slots: only the third batch's record fails."""
    # First Party
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.storage_manager import StorageManager

    monkeypatch.setattr(mod, "_RETIRED_BUFFERS", [])
    keys = ["k0", "k1", "k2"]
    storage = StorageManager.__new__(StorageManager)
    storage._l1_manager = Mock()
    storage._event_bus = Mock()
    storage._l1_manager.unsafe_read.return_value = {
        key: (L1Error.SUCCESS, _object(128, index + 1))
        for index, key in enumerate(keys)
    }
    pool = StagedRetrievePool(2, 128, 1)
    enqueued: list[int] = []
    recorded: list[int] = []

    def enqueue(_start: int, bounce: list) -> None:
        enqueued.append(bounce[0].data_ptr)

    def record() -> _Handle:
        recorded.append(len(enqueued))
        if len(enqueued) == 3:
            raise RuntimeError("lost GPU on batch 3")
        return _Handle()

    with pool.session() as transfer:
        with pytest.raises(RuntimeError, match="bounce slot retired"):
            with storage.read_prefetched_results(keys) as objects:
                transfer(objects, enqueue, record)
    # Two batches landed before the third record failed; the read context
    # releases the readers of every key exactly once.
    assert recorded == [1, 2, 3]
    storage._l1_manager.finish_read.assert_called_once_with(keys)
    status = pool.report_status()
    assert status["quarantined_slots"] == 1
    assert status["slots_in_use"] == 1
    assert status["active_sessions"] == 0
    assert [buffer.data_ptr() for buffer in mod._RETIRED_BUFFERS] == [enqueued[2]]
    # The other slot keeps serving; the retired buffer is never handed out again.
    with pool.session() as transfer:
        transfer([_object(128)], enqueue, _done)
    assert enqueued[3] != enqueued[2]
    assert pool.report_status()["slots_in_use"] == 1
    pool.close()
    pool.close()  # idempotent: native shutdown joins once, retired buffer kept
    status = pool.report_status()
    assert status["closed"] is True
    assert status["quarantined_slots"] == 1
    assert status["allocated_bytes"] == 128
    with pytest.raises(RuntimeError, match="closing"):
        with pool.session():
            pass


def test_all_slots_retired_fails_acquire_promptly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _capture(monkeypatch, caplog)
    pool = StagedRetrievePool(1, 128, 1)
    in_record = threading.Event()
    lose_gpu = threading.Event()
    enqueue = Mock()

    def failing_record():
        in_record.set()
        assert lose_gpu.wait(5)
        raise RuntimeError("lost GPU")

    def run(record):
        with pool.session() as transfer:
            transfer([_object(128)], enqueue, record)

    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(run, failing_record)
        assert in_record.wait(5)
        # The slot is held by a session that has not recorded yet, so the
        # waiter polls the condition until it is registered or returned.
        waiting = executor.submit(run, _done)
        _wait_until(lambda: pool.report_status()["slot_waits"] == 1)
        lose_gpu.set()
        with pytest.raises(RuntimeError, match="bounce slot retired"):
            first.result(5)
        # A pipeline already waiting for the slot is woken and fails.
        with pytest.raises(StagedRetrieveRetiredError, match="every staged"):
            waiting.result(5)
        # New pipelines fail at once instead of waiting forever.
        with pytest.raises(StagedRetrieveRetiredError, match="every staged"):
            executor.submit(run, _done).result(5)
    assert enqueue.call_count == 1
    assert pool.exhausted is True
    # The transition to "every slot retired" is logged as exactly one ERROR.
    assert len(_exhausted_errors(caplog)) == 1
    status = pool.report_status()
    assert status["quarantined_slots"] == 1
    assert status["active_sessions"] == 0
    # Only the pipeline that really waited counts; the new pipeline's
    # immediate failure is not slot pressure.
    assert status["slot_waits"] == 1
    assert status["available_slots"] == 0
    pool.close()


def test_exhausted_pool_fails_before_any_native_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pool = mod.native.DevDaxCopyPool
    pools: list = []

    class CountingPool:
        num_threads = 1

        def __init__(self, _threads):
            self.pool = real_pool(1)
            self.submits = 0
            pools.append(self)

        def submit(self, *args):
            self.submits += 1
            return self.pool.submit(*args)

        def shutdown(self):
            self.pool.shutdown()

    monkeypatch.setattr(mod.native, "DevDaxCopyPool", CountingPool)
    with closing(StagedRetrievePool(1, 128, 1)) as pool:
        with pool.session() as transfer:
            with pytest.raises(RuntimeError, match="bounce slot retired"):
                transfer(
                    [_object(128)],
                    lambda *_: None,
                    Mock(side_effect=RuntimeError("lost GPU")),
                )
        assert pool.exhausted is True
        assert pools[0].submits == 1
        # The failure happens in the slot acquire, before any native copy.
        with pool.session() as transfer:
            with pytest.raises(StagedRetrieveRetiredError, match="every staged"):
                transfer([_object(128)], lambda *_: None, _done)
        assert pools[0].submits == 1


def test_oversize_and_unavailable_native_fail_explicitly(monkeypatch) -> None:
    with closing(StagedRetrievePool(1, 128, 1)) as pool:
        enqueue = Mock()
        with pool.session() as transfer:
            with pytest.raises(ValueError, match="slot_bytes"):
                transfer([_object(129)], enqueue, _done)
        enqueue.assert_not_called()
        assert pool.report_status()["allocated_bytes"] == 0
    monkeypatch.delattr(mod.native, "DevDaxCopyPool")
    with pytest.raises(RuntimeError, match="rebuild"):
        StagedRetrievePool(1, 128, 1)


@pytest.mark.parametrize("timeout", [0, -1.0, True, "1"])
def test_timeouts_must_be_positive_numbers(timeout) -> None:
    for name in ("event_timeout_s", "close_timeout_s"):
        with pytest.raises(ValueError, match="timeouts must be positive"):
            StagedRetrievePool(1, 128, 1, **{name: timeout})
    with closing(
        StagedRetrievePool(1, 128, 1, event_timeout_s=1, close_timeout_s=0.5)
    ) as pool:
        status = pool.report_status()
        assert status["event_timeout_s"] == 1.0
        assert status["close_timeout_s"] == 0.5


def test_session_callable_cannot_outlive_reservation() -> None:
    with closing(StagedRetrievePool(1, 128, 1)) as pool:
        with pool.session() as transfer:
            pass
        with pytest.raises(RuntimeError, match="owning session"):
            transfer([_object(128)], lambda *_: None, _done)


def test_record_failure_releases_shared_reader_via_context() -> None:
    # First Party
    from lmcache.v1.distributed.error import L1Error
    from lmcache.v1.distributed.storage_manager import StorageManager

    storage = StorageManager.__new__(StorageManager)
    storage._l1_manager = Mock()
    storage._event_bus = Mock()
    storage._l1_manager.unsafe_read.return_value = {
        "key": (L1Error.SUCCESS, _object(64))
    }
    with closing(StagedRetrievePool(1, 128, 1)) as pool:
        with pool.session() as transfer:
            with pytest.raises(RuntimeError, match="retired"):
                with storage.read_prefetched_results(["key"]) as objects:
                    transfer(
                        objects,
                        lambda *_: None,
                        Mock(side_effect=RuntimeError("lost GPU")),
                    )
    storage._l1_manager.finish_read.assert_called_once_with(["key"])
