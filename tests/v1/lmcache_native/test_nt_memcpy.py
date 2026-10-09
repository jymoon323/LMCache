# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the runtime-dispatched Device-DAX gather copy."""

# Standard
import ctypes

# Third Party
import pytest

pytestmark = pytest.mark.no_shared_allocator

lmcache_native = pytest.importorskip(
    "lmcache.lmcache_native",
    reason="lmcache_native extension not built",
)


def _allocation(nbytes: int) -> tuple[ctypes.Array, int]:
    allocation = ctypes.create_string_buffer(nbytes)
    return allocation, ctypes.addressof(allocation)


def _payload(nbytes: int) -> bytes:
    pattern = bytes(range(251))
    return (pattern * (nbytes // len(pattern) + 1))[:nbytes]


def test_devdax_memcpy_nt_copies_large_unaligned_range() -> None:
    nbytes = int(lmcache_native.DEVDAX_MEMCPY_NT_MIN_BYTES) + 137
    src_owner, src_base = _allocation(nbytes + 64)
    dst_owner, dst_base = _allocation(nbytes + 64)
    src = src_base + 5
    dst = dst_base + 3
    payload = _payload(nbytes)
    ctypes.memmove(src, payload, nbytes)

    lmcache_native.devdax_memcpy_nt(dst, src, nbytes)

    assert ctypes.string_at(dst, nbytes) == payload
    # Keep raw allocations alive until after the native access and assertion.
    assert src_owner is not None and dst_owner is not None


def test_devdax_memcpy_nt_handles_small_and_overlapping_ranges() -> None:
    small = 513
    src_owner, src = _allocation(small + 8)
    dst_owner, dst = _allocation(small + 8)
    payload = _payload(small)
    ctypes.memmove(src + 1, payload, small)
    lmcache_native.devdax_memcpy_nt(dst + 3, src + 1, small)
    assert ctypes.string_at(dst + 3, small) == payload

    nbytes = int(lmcache_native.DEVDAX_MEMCPY_NT_MIN_BYTES) + 31
    overlap_owner, base = _allocation(nbytes + 128)
    initial = _payload(nbytes + 64)
    ctypes.memmove(base, initial, len(initial))
    expected = bytearray(initial)
    expected[33 : 33 + nbytes] = expected[1 : 1 + nbytes]

    lmcache_native.devdax_memcpy_nt(base + 33, base + 1, nbytes)

    assert ctypes.string_at(base, len(initial)) == bytes(expected)
    assert src_owner is not None and dst_owner is not None
    assert overlap_owner is not None


def test_devdax_memcpy_nt_validates_addresses_and_reports_backend() -> None:
    lmcache_native.devdax_memcpy_nt(0, 0, 0)
    with pytest.raises(ValueError, match="non-null addresses"):
        lmcache_native.devdax_memcpy_nt(0, 1, 1)
    with pytest.raises(ValueError, match="non-null addresses"):
        lmcache_native.devdax_memcpy_nt(1, 0, 1)
    with pytest.raises(ValueError, match="non-null addresses"):
        lmcache_native.devdax_memcpy_nt(0, 0, 1)

    assert lmcache_native.devdax_memcpy_nt_backend() in {"avx2_nt", "libc"}
