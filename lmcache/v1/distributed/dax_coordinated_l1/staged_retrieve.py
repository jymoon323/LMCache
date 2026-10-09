# SPDX-License-Identifier: Apache-2.0
"""Bounded, transient pinned buffers for DAX-Coordinated L1 retrieval.

A pipeline gathers one batch into one slot at a time: it waits for the native
gather, enqueues the H2D copy and records a completion handle, then takes
another free slot for the next batch without waiting for the DMA, so one
pipeline can have every slot in flight. The session ends once the last batch
is gathered, enqueued and recorded. A slot returns to the pool when its
completion is observed at a later acquisition, a status report or close. When
no slot is free the pipeline waits for the oldest in-flight completion;
nothing reserves slots ahead, so the wait cannot deadlock. The caller holds
the shared read reservation through the session, which does not span the DMA:
the GPU reads only the bounce, so the DAX payload is not needed once the
gather is done.

A slot is retired in four cases: its completion cannot be recorded, its query
raises, its completion is still pending past the event timeout while a
pipeline waits for the slot, or it is still in flight when the close-time
drain window ends (quarantine). A retired slot's buffer is kept for the
process lifetime in a module-level registry and never reused; a record
failure fails its retrieve, whose shared read reservation is released as for
any other read failure.
"""

# Standard
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Protocol
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj, TensorMemoryObj
import lmcache.lmcache_native as native

logger = init_logger(__name__)
_RETIRED_BUFFERS: list[torch.Tensor] = []
"""Bounce buffers whose last DMA could not be proven complete.

Appended at retire time and never cleared, so a retired buffer outlives its
pool: a pinned block returned to the PyTorch host allocator could otherwise be
handed to the next pinned allocation while the GPU may still read it.
"""
_POLL_INTERVAL_S = 0.0001
"""100 us completion polling: bounded without adding millisecond DMA latency."""


class CompletionHandle(Protocol):
    """A recorded device completion: ``query`` polls it, ``synchronize`` waits.

    ``torch.cuda.Event`` satisfies this once recorded. The pool only polls
    ``query`` so that every wait stays bounded by the event timeout.
    """

    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


Enqueue = Callable[[int, list[MemoryObj]], None]
Record = Callable[[], CompletionHandle]
Transfer = Callable[[Sequence[MemoryObj], Enqueue, Record], None]


@dataclass(slots=True)
class _InFlight:
    """A bounce slot whose last DMA has been enqueued but not yet observed."""

    handle: CompletionHandle
    recorded_at: float
    deadline: float


class StagedRetrieveRetiredError(RuntimeError):
    """Every bounce slot is retired after unproven DMA completions.

    Staged retrieves fail until the server restarts.
    """


class StagedRetrievePool:
    """Own a fixed number of lazily allocated pinned DRAM slots and CPU workers.

    Args:
        slots: Maximum concurrent bounce allocations.
        slot_bytes: Bytes per allocation; a single object must fit in a slot.
        copy_threads: Number of native libc copy workers (no Python fallback).
        event_timeout_s: How long a recorded completion may stay pending once
            an acquisition waits on it; past that the slot is retired as
            unknown instead of blocking the pipeline forever.
        close_timeout_s: How long close waits for in-flight DMA before
            quarantining the remaining slots.

    Raises:
        ValueError: If a limit is not a positive integer or a timeout is not
            a positive number.
        RuntimeError: If the native copy extension is unavailable.

    Pinned payload capacity is bounded by ``slots * slot_bytes`` before the
    PyTorch host allocator rounds each slot up to a power of two. Each
    allocation is one pinned region; descriptor addresses are offsets within
    it, never DAX logical offsets. Buffers are not a KV cache tier. A slot
    whose completion could not be recorded, whose query raised, or whose DMA
    was not observed complete within ``event_timeout_s`` (or
    ``close_timeout_s`` at close) is retired and its buffer kept until process
    exit in the module-level ``_RETIRED_BUFFERS`` registry, which outlives the
    pool.
    """

    def __init__(
        self,
        slots: int,
        slot_bytes: int,
        copy_threads: int,
        *,
        event_timeout_s: float = 30.0,
        close_timeout_s: float = 30.0,
    ) -> None:
        for value in (slots, slot_bytes, copy_threads):
            if type(value) is not int or value <= 0:
                raise ValueError("staged retrieve limits must be positive integers")
        for timeout in (event_timeout_s, close_timeout_s):
            if type(timeout) not in (int, float) or timeout <= 0:
                raise ValueError("staged retrieve timeouts must be positive numbers")
        if not hasattr(native, "DevDaxCopyPool"):
            raise RuntimeError(
                "staged retrieve requires native DevDaxCopyPool; rebuild"
            )
        self._copy_pool = native.DevDaxCopyPool(copy_threads)
        self._slot_bytes = slot_bytes
        self._event_timeout_s = float(event_timeout_s)
        self._close_timeout_s = float(close_timeout_s)
        self._buffers: list[torch.Tensor | None] = [None] * slots
        self._available = list(range(slots))
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._active = 0
        # Slots whose DMA has been enqueued and recorded but not yet observed
        # complete. They are neither available nor held by a session; a later
        # acquisition, status report or close reclaims them.
        self._in_flight: dict[int, _InFlight] = {}
        # Slots whose DMA completion could not be proven. Their buffers stay
        # in _buffers for the process lifetime (a DMA may still read them) and
        # never return to _available. DAX sources are not retained: the GPU
        # reads only bounce. Retire is absorbing and never undone.
        self._retired: set[int] = set()
        self._peak_slots = 0
        self._bytes = 0
        self._slot_waits = 0
        self._event_queries = 0
        self._event_timeouts = 0
        self._quarantined_at_close = 0
        self._copy_seconds = 0.0
        logger.info(
            "DAX staged retrieve: native libc, %d workers, %d pinned slots of %d "
            "bytes, event timeout %g s, close timeout %g s",
            copy_threads,
            slots,
            slot_bytes,
            self._event_timeout_s,
            self._close_timeout_s,
        )

    @contextmanager
    def session(self) -> Iterator[Transfer]:
        """Admit a pipeline until its shared read reservation has been settled.

        Yields a callable ``transfer(objects, enqueue, record)``. ``enqueue``
        receives the source index and packed bounce objects. ``record`` must
        return a completion handle covering all work submitted by enqueue,
        including partial submissions; it runs even when enqueue raises.
        ``transfer`` returns right after the last batch is recorded, without
        waiting for the DMA. Use the callable only within this context, on
        the entering thread. Acquire/read/finish shared reservations inside
        this context so close cannot unmap an active reader; the DMA may
        still run after the context ends, reading only the bounce.

        Raises:
            RuntimeError: If the pool is closing, on use outside the context,
                or if a completion cannot be recorded (the slot is retired).
            StagedRetrieveRetiredError: If every slot is retired; a
                ``RuntimeError`` subclass raised before any copy is submitted.
                A slot retired while this pipeline waits for it (a query that
                raises or an event timeout) only affects that slot; the wait
                continues on the others.
        """
        with self._condition:
            if self._closing:
                raise RuntimeError("staged retrieve is closing")
            self._active += 1
        active = True
        owner = threading.get_ident()

        def transfer(
            objects: Sequence[MemoryObj], enqueue: Enqueue, record: Record
        ) -> None:
            if not active or threading.get_ident() != owner:
                raise RuntimeError("staged transfer must run inside its owning session")
            self._transfer(objects, enqueue, record)

        try:
            yield transfer
        finally:
            active = False
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def report_status(self) -> dict:
        """Return bounded allocation, pressure, gather and shutdown counters.

        Completed in-flight slots are reclaimed first, as an acquisition
        would, so a slot whose DMA has just finished counts as available.
        """
        with self._condition:
            retired = self._reclaim_locked()
            status = {
                "copy_backend": "native",
                "copy_mode": "libc",
                "copy_threads": self._copy_pool.num_threads,
                "slot_bytes": self._slot_bytes,
                "slots": len(self._buffers),
                "capacity_bytes": len(self._buffers) * self._slot_bytes,
                "allocated_bytes": sum(b is not None for b in self._buffers)
                * self._slot_bytes,
                "slots_in_use": len(self._buffers) - len(self._available),
                "available_slots": len(self._available),
                "in_flight_slots": len(self._in_flight),
                "peak_slots": self._peak_slots,
                "slot_waits": self._slot_waits,
                "event_queries": self._event_queries,
                "event_timeouts": self._event_timeouts,
                "event_timeout_s": self._event_timeout_s,
                "close_timeout_s": self._close_timeout_s,
                "active_sessions": self._active,
                "gather_bytes": self._bytes,
                "gather_seconds": self._copy_seconds,
                "quarantined_slots": len(self._retired),
                "quarantined_at_close": self._quarantined_at_close,
                "closing": self._closing,
                "closed": self._closed,
            }
        self._log_retired(retired)
        return status

    @property
    def exhausted(self) -> bool:
        """Return whether every slot is retired, so no transfer can succeed."""
        with self._condition:
            return len(self._retired) == len(self._buffers)

    def close(self) -> None:
        """Reject new sessions, drain CPU/GPU work for a bounded time, release.

        Active sessions are waited for without a timeout: a session now ends
        right after its last batch is recorded, and its only wait, a slot
        acquisition, is bounded by the event timeout. In-flight DMA is then
        polled for up to ``close_timeout_s``; slots still in flight are
        quarantined (retired, buffer kept, registry) with a warning, so close
        cannot hang on a device that never fires its completion. Retired
        slots keep their buffers for the process lifetime because their DMA
        completion was never proven; this is logged, not raised. The
        module-level registry keeps them after the pool itself is gone. A
        second close is idempotent.
        """
        with self._close_lock:
            with self._condition:
                self._closing = True
                while self._active:
                    self._condition.wait()
            deadline = time.monotonic() + self._close_timeout_s
            while True:
                with self._condition:
                    retired = self._reclaim_locked()
                    pending = list(self._in_flight)
                self._log_retired(retired)
                if not pending:
                    break
                if time.monotonic() >= deadline:
                    # Quarantine: the same absorbing retire, without the
                    # per-slot error. The pool is closing, so no later
                    # retrieve is affected; one warning names the count.
                    with self._condition:
                        quarantined = [
                            slot
                            for slot in pending
                            if slot in self._in_flight
                            and self._retire_locked(slot) is not None
                        ]
                        self._quarantined_at_close += len(quarantined)
                    if quarantined:
                        logger.warning(
                            "DAX staged retrieve close: %d bounce slot(s) still in "
                            "flight after %g s; their buffers are quarantined for "
                            "the process lifetime",
                            len(quarantined),
                            self._close_timeout_s,
                        )
                    break
                time.sleep(_POLL_INTERVAL_S * 10)
            self._copy_pool.shutdown()
            with self._condition:
                self._buffers = [
                    buffer if slot in self._retired else None
                    for slot, buffer in enumerate(self._buffers)
                ]
                self._closed = True
                retired_count = len(self._retired)
            if retired_count:
                logger.warning(
                    "DAX staged retrieve closed, keeping %d retired bounce "
                    "buffer(s), %d bytes, for the process lifetime; DMA "
                    "completion was never proven",
                    retired_count,
                    retired_count * self._slot_bytes,
                )

    def _transfer(
        self, objects: Sequence[MemoryObj], enqueue: Enqueue, record: Record
    ) -> None:
        # Validate the whole group before any raw-pointer submission.
        sizes = []
        for obj in objects:
            raw = obj.raw_tensor
            size = obj.get_size()
            if (
                raw is None
                or raw.device.type != "cpu"
                or not raw.is_contiguous()
                or size <= 0
                or size > raw.numel() * raw.element_size()
                or size > self._slot_bytes
            ):
                raise ValueError(
                    "staged retrieve needs contiguous CPU objects fitting "
                    "staged_retrieve_slot_bytes"
                )
            sizes.append(size)

        start = 0
        while start < len(objects):
            offsets = []
            used = 0
            end = start
            while end < len(objects):
                offset = (used + 63) // 64 * 64
                if offset + sizes[end] > self._slot_bytes:
                    break
                offsets.append(offset)
                used = offset + sizes[end]
                end += 1
            slot = self._acquire()
            registered = False
            try:
                buffer = self._buffers[slot]
                if buffer is None:
                    buffer = torch.empty(
                        self._slot_bytes,
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=True,
                    )
                    self._buffers[slot] = buffer
                sources = list(objects[start:end])
                bounce: list[MemoryObj] = []
                for obj, offset, size in zip(
                    sources, offsets, sizes[start:end], strict=True
                ):
                    metadata = replace(
                        obj.meta,
                        address=offset,
                        phy_size=size,
                        ref_count=1,
                        pin_count=0,
                        shapes=list(obj.meta.shapes) if obj.meta.shapes else None,
                        dtypes=list(obj.meta.dtypes) if obj.meta.dtypes else None,
                    )
                    view = TensorMemoryObj(
                        buffer[offset : offset + size], metadata, None
                    )
                    view.set_used_size(size)
                    bounce.append(view)
                began = time.monotonic()
                # Native submit is atomic; wait settles every copy before raising.
                # No later submit can strand an earlier batch on the exception path.
                batch = self._copy_pool.submit(
                    [
                        (dst.data_ptr, src.data_ptr, src.get_size())
                        for dst, src in zip(bounce, sources, strict=True)
                    ],
                    native.DevDaxCopyStores.LIBC,
                )
                batch.wait()
                with self._condition:
                    self._bytes += sum(sizes[start:end])
                    self._copy_seconds += time.monotonic() - began
                try:
                    enqueue(start, bounce)
                finally:
                    # Record after every enqueue attempt, including partial
                    # failures, so the slot is recycled once the GPU is done.
                    try:
                        handle = record()
                    except BaseException as error:
                        # Only the bounce can still be read by the GPU. Keep it
                        # out of service; the caller releases DAX readers.
                        with self._condition:
                            retired = self._retire_locked(slot)
                        self._log_retired(
                            [(slot, retired, "completion could not be recorded", error)]
                        )
                        raise RuntimeError(
                            "staged DMA completion could not be recorded; "
                            "bounce slot retired"
                        ) from error
                    with self._condition:
                        now = time.monotonic()
                        self._in_flight[slot] = _InFlight(
                            handle, now, now + self._event_timeout_s
                        )
                        registered = True
                        self._condition.notify_all()
            finally:
                if not registered:
                    # Nothing was enqueued for this slot (a copy or validation
                    # failure before record), or record failed and retired it.
                    with self._condition:
                        if slot not in self._retired:
                            self._available.append(slot)
                        self._condition.notify_all()
            start = end

    def _acquire(self) -> int:
        """Return a free slot, reclaiming or waiting on in-flight DMA first.

        The wait polls the oldest in-flight slot outside the condition, so a
        completion that never fires retires that slot after the event timeout
        instead of blocking every pipeline; the loop then continues on the
        remaining slots.
        """
        waited = False
        while True:
            with self._condition:
                retired = self._reclaim_locked()
            self._log_retired(retired)
            with self._condition:
                if self._available:
                    slot = self._available.pop()
                    self._peak_slots = max(
                        self._peak_slots, len(self._buffers) - len(self._available)
                    )
                    return slot
                if len(self._retired) == len(self._buffers):
                    raise StagedRetrieveRetiredError(
                        "every staged retrieve slot is retired after unproven "
                        "DMA completions; restart the server"
                    )
                if not waited:
                    # Count each acquisition once, and only when it really
                    # waits: an immediate failure above is not slot pressure.
                    self._slot_waits += 1
                    waited = True
                oldest = min(
                    self._in_flight.items(),
                    key=lambda item: item[1].recorded_at,
                    default=None,
                )
                if oldest is None:
                    # Every slot is held by a session still gathering or
                    # enqueueing; it registers or returns its slot under this
                    # condition and notifies.
                    self._condition.wait(0.0005)
                    continue
                slot, flight = oldest
            self._wait_for_completion(slot, flight)

    def _wait_for_completion(self, slot: int, flight: _InFlight) -> None:
        """Poll ``flight`` outside the condition until done, raised or expired.

        A completion that raises or stays pending past its deadline retires
        the slot, unless another thread already reclaimed or retired it. The
        caller re-checks the pool afterwards either way.
        """
        while True:
            try:
                done = flight.handle.query()
            except Exception as error:
                self._retire_in_flight(slot, flight, "completion query failed", error)
                return
            if done:
                return
            if time.monotonic() >= flight.deadline:
                self._retire_in_flight(
                    slot,
                    flight,
                    f"did not complete within {self._event_timeout_s:g} s",
                    None,
                )
                return
            time.sleep(_POLL_INTERVAL_S)

    def _retire_in_flight(
        self,
        slot: int,
        flight: _InFlight,
        reason: str,
        error: BaseException | None,
    ) -> None:
        """Retire ``slot`` if ``flight`` is still the registration waited on."""
        with self._condition:
            if self._in_flight.get(slot) is not flight:
                return
            if error is None:
                self._event_timeouts += 1
            retired = self._retire_locked(slot)
        self._log_retired([(slot, retired, reason, error)])

    def _reclaim_locked(self) -> list[tuple[int, int | None, str, BaseException]]:
        """Move in-flight slots whose DMA completed back to the free list.

        Caller holds the condition. A ``query`` that raises (a sticky device
        error) retires its slot instead of failing every later acquisition;
        the returned retire events are logged by the caller outside the
        condition.
        """
        retired: list[tuple[int, int | None, str, BaseException]] = []
        for slot, flight in list(self._in_flight.items()):
            self._event_queries += 1
            try:
                done = flight.handle.query()
            except Exception as error:
                retired.append(
                    (slot, self._retire_locked(slot), "completion query failed", error)
                )
                continue
            if done:
                del self._in_flight[slot]
                self._available.append(slot)
                self._condition.notify_all()
        return retired

    def _retire_locked(self, slot: int) -> int | None:
        """Take ``slot`` out of circulation; caller holds the condition.

        Returns the retired count after this call, or None when the slot was
        already retired: retire is absorbing, so a second retire of the same
        slot changes nothing and registers nothing. The buffer joins the
        module-level registry here, before any later close could free it.
        """
        if slot in self._retired:
            return None
        self._retired.add(slot)
        self._in_flight.pop(slot, None)
        buffer = self._buffers[slot]
        if buffer is not None:
            _RETIRED_BUFFERS.append(buffer)
        # Waiters re-check: a retire can be the last one (every slot gone).
        self._condition.notify_all()
        return len(self._retired)

    def _log_retired(
        self, events: Sequence[tuple[int, int | None, str, BaseException | None]]
    ) -> None:
        """Log retire events outside the condition; one ERROR at the last one.

        Each event is ``(slot, retired_count, reason, error)``; a None count
        means the slot had already been retired and is not logged again.
        """
        for slot, retired, reason, error in events:
            if retired is None:
                continue
            logger.error(
                "staged DMA %s; retiring bounce slot %d (%d of %d retired)",
                reason,
                slot,
                retired,
                len(self._buffers),
                exc_info=error,
            )
            if retired == len(self._buffers):
                # Retire is absorbing, so this transition happens at most
                # once per process: one ERROR here, not one traceback per
                # later request.
                logger.error(
                    "every staged retrieve slot is retired after unproven DMA "
                    "completions; staged retrieves fail until the server restarts"
                )
