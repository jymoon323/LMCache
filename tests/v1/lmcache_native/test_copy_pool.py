# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the native Device-DAX copy pool (``DevDaxCopyPool``).

The pool is driven directly, without the transfer engine: non-temporal
stores on unaligned and overlapping ranges, submissions racing a shutdown,
the drain run by the destructor, batches that outlive their pool, several
waiters on one batch, and the sticky copy error.
"""

# Standard
import ctypes
import gc
import threading

# Third Party
import pytest

lmcache_native = pytest.importorskip(
    "lmcache.lmcache_native",
    reason="lmcache_native extension not built",
)

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        not hasattr(lmcache_native, "DevDaxCopyPool"),
        reason="lmcache_native lacks DevDaxCopyPool (source-only or older build)",
    ),
]

MIB = 1 << 20


def _allocation(nbytes: int) -> tuple[ctypes.Array, int]:
    allocation = ctypes.create_string_buffer(nbytes)
    return allocation, ctypes.addressof(allocation)


def _payload(nbytes: int) -> bytes:
    pattern = bytes(range(251))
    return (pattern * (nbytes // len(pattern) + 1))[:nbytes]


def _stores(name: str):
    return getattr(lmcache_native.DevDaxCopyStores, name)


def _nt_min_bytes() -> int:
    return int(lmcache_native.DEVDAX_MEMCPY_NT_MIN_BYTES)


@pytest.mark.parametrize("stores", ["LIBC", "NON_TEMPORAL"])
def test_stores_copy_unaligned_pieces_above_and_below_the_nt_minimum(stores):
    """Three unaligned pieces above the non-temporal minimum and one small
    piece below it, in one submission: every byte lands, the guard bytes
    around the destinations are untouched, and the batch reports its size."""
    piece = _nt_min_bytes() + 337
    nbytes = 3 * piece
    src_owner, src_base = _allocation(nbytes + 64)
    dst_owner, dst_base = _allocation(nbytes + 64)
    src, dst = src_base + 13, dst_base + 7
    payload = _payload(nbytes)
    ctypes.memmove(src, payload, nbytes)
    ctypes.memset(dst_base, 0xEE, nbytes + 64)
    small = 513
    small_src_owner, small_src = _allocation(small + 8)
    small_dst_owner, small_dst = _allocation(small + 8)
    ctypes.memmove(small_src + 1, payload, small)
    ctypes.memset(small_dst, 0xEE, small + 8)
    copies = [(dst + k * piece, src + k * piece, piece) for k in range(3)]
    copies.append((small_dst + 3, small_src + 1, small))

    pool = lmcache_native.DevDaxCopyPool(3)
    try:
        batch = pool.submit(copies, stores=_stores(stores))
        batch.wait()
        assert batch.done()
        assert batch.count == 4 and batch.nbytes == nbytes + small
        assert ctypes.string_at(dst, nbytes) == payload
        assert ctypes.string_at(dst_base, 7) == b"\xee" * 7
        assert ctypes.string_at(dst + nbytes, 57) == b"\xee" * 57
        assert ctypes.string_at(small_dst + 3, small) == payload[:small]
        assert ctypes.string_at(small_dst, 3) == b"\xee" * 3
        assert ctypes.string_at(small_dst + 3 + small, 5) == b"\xee" * 5
        assert pool.completed_copies == 4
    finally:
        pool.shutdown()
    # Keep the raw allocations alive until after the native accesses.
    assert src_owner is not None and dst_owner is not None
    assert small_src_owner is not None and small_dst_owner is not None


def test_non_temporal_overlapping_copy_keeps_memmove_semantics():
    nbytes = _nt_min_bytes() + 31
    owner, base = _allocation(nbytes + 128)
    initial = _payload(nbytes + 64)
    ctypes.memmove(base, initial, len(initial))
    expected = bytearray(initial)
    expected[33 : 33 + nbytes] = expected[1 : 1 + nbytes]

    pool = lmcache_native.DevDaxCopyPool(1)
    try:
        pool.submit(
            [(base + 33, base + 1, nbytes)], stores=_stores("NON_TEMPORAL")
        ).wait()
    finally:
        pool.shutdown()
    assert ctypes.string_at(base, len(initial)) == bytes(expected)
    assert owner is not None


def test_submissions_racing_shutdown_are_completed_or_refused():
    """Three threads submit without pause while the pool is shut down: each
    is eventually refused (``RuntimeError``), every submission accepted
    before that completes -- the completed count equals the accepted count,
    nothing stays queued -- and every destination holds the data of its last
    accepted copy. Each thread copies within its own ring of source and
    destination pairs, reusing a pair only after the copy that last used it
    has completed, and every submission carries its own byte pattern, so
    two copies never touch one range at once and the final comparison
    exposes a lost or misplaced write."""
    pool = lmcache_native.DevDaxCopyPool(4)
    nbytes = 128 * 1024
    ring = 16
    rings = [
        [(_allocation(nbytes), _allocation(nbytes)) for _ in range(ring)]
        for _ in range(3)
    ]
    accepted: list[tuple[object, int, int, bytes]] = []
    refusals: list[str] = []
    lock = threading.Lock()
    primed = [threading.Event() for _ in range(3)]

    def _submit_until_refused(index: int) -> None:
        handles: list[object] = []
        count = 0
        while True:
            slot = count % ring
            if count >= ring:
                handles[count - ring].wait()  # the pair's previous copy is done
            (_, src), (_, dst) = rings[index][slot]
            pattern = (bytes([index]) + count.to_bytes(3, "big")) * (nbytes // 4)
            ctypes.memmove(src, pattern, nbytes)
            stores = _stores("NON_TEMPORAL" if count % 2 else "LIBC")
            try:
                handle = pool.submit([(dst, src, nbytes)], stores=stores)
            except RuntimeError as exc:
                with lock:
                    refusals.append(str(exc))
                return
            handles.append(handle)
            with lock:
                accepted.append((handle, index, slot, pattern))
            count += 1
            if count == 16:
                primed[index].set()

    threads = [
        threading.Thread(target=_submit_until_refused, args=(index,))
        for index in range(3)
    ]
    for thread in threads:
        thread.start()
    for event in primed:
        assert event.wait(30)
    pool.shutdown()
    for thread in threads:
        thread.join(30)
    assert not any(thread.is_alive() for thread in threads)

    assert len(refusals) == 3 and all("shut down" in text for text in refusals)
    assert len(accepted) >= 3 * 16
    assert all(handle.done() for handle, *_ in accepted)
    for handle, *_ in accepted:
        handle.wait()
    assert pool.queued == 0
    assert pool.completed_copies == len(accepted)
    # Each thread appended its submissions in order, so the last entry per
    # pair is the copy whose pattern the destination must hold.
    last_pattern: dict[tuple[int, int], bytes] = {}
    for _, index, slot, pattern in accepted:
        last_pattern[(index, slot)] = pattern
    for (index, slot), pattern in last_pattern.items():
        assert ctypes.string_at(rings[index][slot][1][1], nbytes) == pattern
    with pytest.raises(RuntimeError, match="shut down"):
        pool.submit([(rings[0][0][1][1], rings[0][0][0][1], nbytes)])
    pool.shutdown()  # idempotent
    # Keep the raw allocations alive until after the native accesses.
    assert rings is not None


def test_dropping_the_last_reference_drains_the_queue():
    """Without an explicit shutdown, the destructor finishes every queued
    copy before the pool goes away; handles held by the caller stay valid."""
    pool = lmcache_native.DevDaxCopyPool(1)
    nbytes = 2 * MIB
    src_owner, src = _allocation(nbytes)
    ctypes.memset(src, 0x33, nbytes)
    destinations = [_allocation(nbytes) for _ in range(8)]
    handles = [pool.submit([(dst, src, nbytes)]) for _, dst in destinations]

    del pool
    gc.collect()

    assert all(handle.done() for handle in handles)
    for handle in handles:
        handle.wait()
    assert all(
        ctypes.string_at(dst, nbytes) == b"\x33" * nbytes for _, dst in destinations
    )
    assert src_owner is not None


def test_every_waiter_on_one_batch_wakes():
    """Three threads wait on the same batch behind a long queue on a single
    worker; all of them return once it completes."""
    pool = lmcache_native.DevDaxCopyPool(1)
    nbytes = 16 * MIB
    src_owner, src = _allocation(nbytes)
    dst_owner, dst = _allocation(nbytes)
    ctypes.memset(src, 0x77, nbytes)
    blocker = pool.submit([(dst, src, nbytes)] * 4)  # keeps the worker busy
    batch = pool.submit([(dst, src, nbytes)])
    woke: list[int] = []
    lock = threading.Lock()
    start = threading.Barrier(4)

    def _wait() -> None:
        start.wait(5)
        batch.wait()
        with lock:
            woke.append(threading.get_ident())

    threads = [threading.Thread(target=_wait) for _ in range(3)]
    for thread in threads:
        thread.start()
    start.wait(5)
    for thread in threads:
        thread.join(60)
    assert not any(thread.is_alive() for thread in threads), "a waiter hung"
    assert len(woke) == 3 and batch.done()
    blocker.wait()
    pool.shutdown()
    assert ctypes.string_at(dst, nbytes) == b"\x77" * nbytes
    assert src_owner is not None and dst_owner is not None


def test_copy_error_is_reported_on_every_wait():
    pool = lmcache_native.DevDaxCopyPool(1)
    src_owner, src = _allocation(2 * MIB)
    try:
        batch = pool.submit([(0, src, 2 * MIB)], stores=_stores("NON_TEMPORAL"))
        for _ in range(2):
            with pytest.raises(RuntimeError, match="null address"):
                batch.wait()
        assert batch.done()
        assert pool.completed_copies == 1  # the failed copy still counts as run
    finally:
        pool.shutdown()
    assert src_owner is not None
